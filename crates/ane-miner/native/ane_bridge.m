#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <CoreFoundation/CFByteOrder.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>
#include "ane_bridge.h"

// Local declarations give ARC the actual ownership and NSError conventions.
@interface NSObject (QuipAnePrivate)
- (instancetype)initWithNetworkText:(NSData *)text weights:(NSDictionary *)weights
                      optionsPlist:(NSData *)options isMILModel:(BOOL)isMIL;
+ (instancetype)inMemoryModelWithDescriptor:(id)descriptor;
- (NSString *)hexStringIdentifier;
- (BOOL)compileWithQoS:(NSInteger)qos options:(NSDictionary *)options error:(NSError **)error;
- (BOOL)loadWithQoS:(NSInteger)qos options:(NSDictionary *)options error:(NSError **)error;
+ (instancetype)objectWithIOSurface:(IOSurfaceRef)surface;
+ (instancetype)requestWithInputs:(NSArray *)inputs inputIndices:(NSArray *)inputIndices
                         outputs:(NSArray *)outputs outputIndices:(NSArray *)outputIndices
                   weightsBuffer:(id)weights perfStats:(id)stats procedureIndex:(NSNumber *)index;
- (BOOL)evaluateWithQoS:(NSInteger)qos options:(NSDictionary *)options request:(id)request error:(NSError **)error;
- (BOOL)unloadWithQoS:(NSInteger)qos error:(NSError **)error;
@end

static int32_t fail(char *error, size_t capacity, NSString *message) {
    if (error != NULL && capacity > 0) {
        const char *text = message.UTF8String;
        snprintf(error, capacity, "%s", text != NULL ? text : "ANE runtime failure");
        error[capacity - 1] = '\0';
    }
    return 1;
}

static NSString *describe(NSString *operation, NSError *error) {
    return [NSString stringWithFormat:@"%@: %@", operation, error != nil ? error.description : @"runtime returned false"];
}

static void requireSelector(id receiver, SEL selector) {
    if (receiver == nil || ![receiver respondsToSelector:selector]) {
        @throw [NSException exceptionWithName:@"QuipAneUnavailable"
            reason:[NSString stringWithFormat:@"ANE selector unavailable: %@", NSStringFromSelector(selector)] userInfo:nil];
    }
}

static BOOL multiply(size_t a, size_t b, size_t *result) {
    if (b != 0 && a > SIZE_MAX / b) return NO;
    *result = a * b;
    return YES;
}

static BOOL surfaceBytes(size_t elements, size_t *bytes) {
    size_t payload;
    if (!multiply(elements, sizeof(_Float16), &payload) || payload > SIZE_MAX - 65535) return NO;
    *bytes = MAX((size_t)65536, (payload + 65535) & ~(size_t)65535);
    return *bytes <= UINT32_MAX;
}

static IOSurfaceRef makeSurface(size_t elements) {
    size_t bytes;
    if (!surfaceBytes(elements, &bytes)) return NULL;
    NSDictionary *properties = @{
        (__bridge NSString *)kIOSurfaceWidth: @(bytes),
        (__bridge NSString *)kIOSurfaceHeight: @1,
        (__bridge NSString *)kIOSurfaceBytesPerElement: @1,
        (__bridge NSString *)kIOSurfaceBytesPerRow: @(bytes),
        (__bridge NSString *)kIOSurfaceAllocSize: @(bytes)
    };
    return IOSurfaceCreate((__bridge CFDictionaryRef)properties);
}

static uint64_t monotonicUS(void) {
    return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) / 1000;
}

@interface QuipAneProgram : NSObject {
@public
    size_t inputElements;
    size_t sweeps;
    size_t current;
    size_t nextThresholdBank;
    BOOL initialized;
    BOOL pending;
    IOSurfaceRef surfaces[18];
    dispatch_queue_t queue;
    dispatch_group_t group;
    uint64_t pendingDispatchUS;
}
@property(nonatomic, strong) id model;
@property(nonatomic, strong) NSArray *requests;
@property(nonatomic, strong) NSArray *wrappers;
@property(nonatomic, strong) NSString *directory;
@property(nonatomic, strong) NSString *pendingError;
@property(nonatomic) BOOL loaded;
- (BOOL)removeDirectory:(NSError **)error;
- (BOOL)unload:(NSError **)error;
@end

@implementation QuipAneProgram
- (BOOL)removeDirectory:(NSError **)error {
    if (self.directory == nil) return YES;
    if (![NSFileManager.defaultManager removeItemAtPath:self.directory error:error]) return NO;
    self.directory = nil;
    return YES;
}
- (BOOL)unload:(NSError **)error {
    if (!self.loaded) return YES;
    // One attempt only, even if the private runtime throws or returns false.
    self.loaded = NO;
    requireSelector(self.model, @selector(unloadWithQoS:error:));
    return [self.model unloadWithQoS:21 error:error];
}
- (void)dealloc {
    if (pending) dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
    @try {
        NSError *error = nil;
        if (![self unload:&error]) NSLog(@"%@", describe(@"ANE cleanup unload", error));
    } @catch (NSException *exception) {
        NSLog(@"ANE cleanup unload exception: %@", exception.reason);
    }
    @try {
        NSError *error = nil;
        if (![self removeDirectory:&error]) NSLog(@"%@", describe(@"ANE staging cleanup", error));
    } @catch (NSException *exception) {
        NSLog(@"ANE staging cleanup exception: %@", exception.reason);
    }
    _requests = nil;
    _wrappers = nil;
    _model = nil;
    for (size_t i = 0; i < 18; ++i) {
        if (surfaces[i] != NULL) CFRelease(surfaces[i]);
    }
}
@end

static NSString *shape(size_t channels, size_t lanes) {
    return [NSString stringWithFormat:@"tensor<fp16, [1, %zu, 1, %zu]>", channels, lanes];
}

static void slice(NSMutableString *mil, NSString *name, NSString *source, size_t begin, size_t count, size_t lanes) {
    [mil appendFormat:@"    tensor<int32, [4]> %@begin = const()[name=string(\"%@begin\"), val=tensor<int32, [4]>([0,%zu,0,0])];\n", name, name, begin];
    [mil appendFormat:@"    tensor<int32, [4]> %@size = const()[name=string(\"%@size\"), val=tensor<int32, [4]>([1,%zu,1,%zu])];\n", name, name, count, lanes];
    [mil appendFormat:@"    %@ %@ = slice_by_size(x=%@, begin=%@begin, size=%@size)[name=string(\"%@\")];\n", shape(count, lanes), name, source, name, name, name];
}

// Coupling weights travel sparse: per tile a one-bit mask over the padded
// [rows, channels] matrix and the nonzero values in mask order, which the
// program expands with constexpr_sparse_to_dense. The engine consumes that
// form directly, so a sweep streams the mask and the values rather than the
// dense fp16 matrix, and the output is bit-identical to the dense program.
// docs/perf/2026-09-18-ane-utilization.md records the measurement.
typedef struct { uint64_t maskOffset, dataOffset; size_t nonzero; } QuipTileWeights;
static const uint32_t kBlobFP16 = 1, kBlobUInt1 = 9;

static size_t align64(size_t bytes) {
    return (bytes + 63) & ~(size_t)63;
}

// Blob layout shared by makeMIL and makeWeightBlob: a 64-byte header, then
// per tile a 64-byte record and payload for the mask, and the same for the
// values, each payload 64-byte aligned. A tile without a nonzero weight
// gets one explicit 0.0 under its first mask bit so that no tensor is empty.
static void layoutTiles(const int8_t *weights, size_t channels, const size_t *lengths, size_t tiles, QuipTileWeights *layout) {
    uint64_t offset = 64;
    size_t weightOffset = 0;
    for (size_t tile = 0; tile < tiles; ++tile) {
        size_t elements = ((lengths[tile] + 31) / 32 * 32) * channels, nonzero = 0;
        for (size_t i = 0; i < elements; ++i) nonzero += weights[weightOffset + i] != 0;
        if (nonzero == 0) nonzero = 1;
        layout[tile].maskOffset = offset;
        offset += 64 + align64(elements / 8);
        layout[tile].dataOffset = offset;
        offset += 64 + align64(nonzero * sizeof(_Float16));
        layout[tile].nonzero = nonzero;
        weightOffset += elements;
    }
}

static NSString *makeMIL(size_t channels, size_t lanes, const size_t *lengths, size_t tiles, size_t sweeps, const int8_t *fields, const int8_t *weights) {
    QuipTileWeights layout[24];
    layoutTiles(weights, channels, lengths, tiles, layout);
    NSMutableString *mil = [NSMutableString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}, {\"quip-ane-msa\", \"%@\"}})]\n{\n  func main<ios18>(%@ a_state", NSUUID.UUID.UUIDString, shape(channels, lanes)];
    for (size_t sweep = 0; sweep < sweeps; ++sweep) [mil appendFormat:@", %@ t%zu", shape(channels, lanes), sweep];
    [mil appendString:@") {\n"
        "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "    int32 axis = const()[name=string(\"axis\"), val=int32(1)];\n"
        "    bool interleave = const()[name=string(\"interleave\"), val=bool(false)];\n"
        "    fp16 zero = const()[name=string(\"zero\"), val=fp16(0.0)];\n"
        "    fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];\n"
        "    fp16 minusTwo = const()[name=string(\"minusTwo\"), val=fp16(-2.0)];\n"];
    size_t begin = 0;
    for (size_t tile = 0; tile < tiles; ++tile) {
        size_t count = lengths[tile], padded = (count + 31) / 32 * 32;
        [mil appendFormat:@"    tensor<uint1, [%zu, %zu, 1, 1]> m%zu = const()[name=string(\"m%zu\"), val=tensor<uint1, [%zu, %zu, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/weight_data.bin\"), offset=uint64(%llu)))];\n", padded, channels, tile, tile, padded, channels, (unsigned long long)layout[tile].maskOffset];
        [mil appendFormat:@"    tensor<fp16, [%zu]> v%zu = const()[name=string(\"v%zu\"), val=tensor<fp16, [%zu]>(BLOBFILE(path=string(\"@model_path/weights/weight_data.bin\"), offset=uint64(%llu)))];\n", layout[tile].nonzero, tile, tile, layout[tile].nonzero, (unsigned long long)layout[tile].dataOffset];
        [mil appendFormat:@"    tensor<fp16, [%zu, %zu, 1, 1]> w%zu = constexpr_sparse_to_dense(nonzero_data=v%zu, mask=m%zu)[name=string(\"w%zu\")];\n", padded, channels, tile, tile, tile, tile];
        NSMutableArray *values = [NSMutableArray new];
        for (size_t row = 0; row < count; ++row) [values addObject:[NSString stringWithFormat:@"%d.0", (int)fields[begin + row]]];
        [mil appendFormat:@"    tensor<fp16, [%zu]> hFlat%zu = const()[name=string(\"hFlat%zu\"), val=tensor<fp16, [%zu]>([%@])];\n", count, tile, tile, count, [values componentsJoinedByString:@","]];
        [mil appendFormat:@"    tensor<int32, [4]> hShape%zu = const()[name=string(\"hShape%zu\"), val=tensor<int32, [4]>([1,%zu,1,1])];\n", tile, tile, count];
        [mil appendFormat:@"    tensor<fp16, [1,%zu,1,1]> h%zu = reshape(x=hFlat%zu, shape=hShape%zu)[name=string(\"h%zu\")];\n", count, tile, tile, tile, tile];
        begin += count;
    }
    NSString *state = @"a_state";
    for (size_t sweep = 0; sweep < sweeps; ++sweep) {
        begin = 0;
        for (size_t tile = 0; tile < tiles; ++tile) {
            size_t count = lengths[tile], padded = (count + 31) / 32 * 32;
            NSString *prefix = [NSString stringWithFormat:@"s%zuc%zu", sweep, tile];
            NSString *own = [prefix stringByAppendingString:@"own"];
            NSString *threshold = [prefix stringByAppendingString:@"threshold"];
            NSString *raw = [prefix stringByAppendingString:@"raw"];
            NSString *js = [prefix stringByAppendingString:@"js"];
            slice(mil, own, state, begin, count, lanes);
            slice(mil, threshold, [NSString stringWithFormat:@"t%zu", sweep], begin, count, lanes);
            [mil appendFormat:@"    %@ %@ = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w%zu, x=%@)[name=string(\"%@\")];\n", shape(padded, lanes), raw, tile, state, raw];
            slice(mil, js, raw, 0, count, lanes);
            NSArray *ops = @[
                @[@"field", [NSString stringWithFormat:@"add(x=%@, y=h%zu)", js, tile]],
                @[@"signed", [NSString stringWithFormat:@"mul(x=%@, y=%@field)", own, prefix]],
                @[@"margin", [NSString stringWithFormat:@"add(x=%@, y=%@signed)", threshold, prefix]],
                @[@"shifted", [NSString stringWithFormat:@"add(x=%@margin, y=one)", prefix]],
                @[@"accept", [NSString stringWithFormat:@"clip(x=%@shifted, alpha=zero, beta=one)", prefix]],
                @[@"negative", [NSString stringWithFormat:@"mul(x=%@accept, y=minusTwo)", prefix]],
                @[@"factor", [NSString stringWithFormat:@"add(x=one, y=%@negative)", prefix]],
                @[@"updated", [NSString stringWithFormat:@"mul(x=%@, y=%@factor)", own, prefix]]
            ];
            for (NSArray *op in ops) [mil appendFormat:@"    %@ %@%@ = %@[name=string(\"%@%@\")];\n", shape(count, lanes), prefix, op[0], op[1], prefix, op[0]];
            NSMutableArray *parts = [NSMutableArray new];
            if (begin > 0) {
                NSString *head = [prefix stringByAppendingString:@"head"];
                slice(mil, head, state, 0, begin, lanes);
                [parts addObject:head];
            }
            [parts addObject:[prefix stringByAppendingString:@"updated"]];
            if (begin + count < channels) {
                NSString *tail = [prefix stringByAppendingString:@"tail"];
                slice(mil, tail, state, begin + count, channels - begin - count, lanes);
                [parts addObject:tail];
            }
            if (parts.count == 1) {
                state = parts[0];
            } else {
                state = [prefix stringByAppendingString:@"state"];
                [mil appendFormat:@"    %@ %@ = concat(values=(%@), axis=axis, interleave=interleave)[name=string(\"%@\")];\n", shape(channels, lanes), state, [parts componentsJoinedByString:@", "], state];
            }
            begin += count;
        }
    }
    [mil appendFormat:@"  } -> (%@);\n}\n", state];
    return mil;
}

// One 64-byte blob record: sentinel, MIL dtype, payload size, payload offset.
static void writeBlobRecord(uint8_t *record, uint32_t dtype, uint64_t size, uint64_t payloadOffset) {
    uint32_t sentinel = CFSwapInt32HostToLittle(0xDEADBEEF), littleType = CFSwapInt32HostToLittle(dtype);
    uint64_t littleSize = CFSwapInt64HostToLittle(size), littleOffset = CFSwapInt64HostToLittle(payloadOffset);
    memcpy(record, &sentinel, sizeof(sentinel));
    memcpy(record + 4, &littleType, sizeof(littleType));
    memcpy(record + 8, &littleSize, sizeof(littleSize));
    memcpy(record + 16, &littleOffset, sizeof(littleOffset));
}

static NSData *makeWeightBlob(const int8_t *weights, size_t channels, const size_t *lengths, size_t tiles) {
    QuipTileWeights layout[24];
    layoutTiles(weights, channels, lengths, tiles, layout);
    QuipTileWeights last = layout[tiles - 1];
    NSMutableData *blob = [NSMutableData dataWithLength:last.dataOffset + 64 + align64(last.nonzero * sizeof(_Float16))];
    uint8_t *bytes = blob.mutableBytes;
    uint32_t chunkCount = CFSwapInt32HostToLittle((uint32_t)(2 * tiles));
    memcpy(bytes, &chunkCount, sizeof(chunkCount));
    bytes[4] = 2;
    size_t weightOffset = 0;
    for (size_t tile = 0; tile < tiles; ++tile) {
        size_t elements = ((lengths[tile] + 31) / 32 * 32) * channels;
        uint8_t *mask = bytes + layout[tile].maskOffset + 64;
        uint8_t *data = bytes + layout[tile].dataOffset + 64;
        size_t nonzero = 0;
        for (size_t i = 0; i < elements; ++i) {
            int8_t weight = weights[weightOffset + i];
            if (weight == 0) continue;
            mask[i / 8] |= (uint8_t)(1 << (i % 8));
            _Float16 value = (_Float16)weight;
            uint16_t bits;
            memcpy(&bits, &value, sizeof(bits));
            bits = CFSwapInt16HostToLittle(bits);
            memcpy(data + nonzero * sizeof(bits), &bits, sizeof(bits));
            nonzero += 1;
        }
        // An empty tile carries one explicit zero, as layoutTiles counted.
        if (nonzero == 0) { mask[0] = 1; nonzero = 1; }
        writeBlobRecord(bytes + layout[tile].maskOffset, kBlobUInt1, elements / 8, layout[tile].maskOffset + 64);
        writeBlobRecord(bytes + layout[tile].dataOffset, kBlobFP16, nonzero * sizeof(_Float16), layout[tile].dataOffset + 64);
        weightOffset += elements;
    }
    return blob;
}

int32_t quip_ane_create(size_t input_channels, size_t lanes, const size_t *lengths, size_t tile_count, size_t sweeps,
    const int8_t *weights, size_t weight_count, const int8_t *fields, size_t field_count,
    void **program, char *error, size_t error_capacity) {
    @autoreleasepool {
        @try {
            if (program == NULL) return fail(error, error_capacity, @"Missing program output pointer");
            *program = NULL;
            if (error == NULL || error_capacity == 0) return 1;
            error[0] = '\0';
            size_t inputElements, allocationBytes;
            if (input_channels < 32 || input_channels > 16384 || input_channels % 32 != 0 || lanes < 32 || lanes > 128 || lanes % 32 != 0 ||
                tile_count == 0 || tile_count > 24 || lengths == NULL || sweeps == 0 || sweeps > 8 ||
                !multiply(input_channels, lanes, &inputElements) || !surfaceBytes(inputElements, &allocationBytes)) {
                return fail(error, error_capacity, @"Invalid ANE channel dimensions or block size");
            }
            size_t rows = 0, weightElements = 0;
            for (size_t tile = 0; tile < tile_count; ++tile) {
                if (lengths[tile] == 0 || lengths[tile] > 4096) return fail(error, error_capacity, @"Invalid ANE tile length");
                rows += lengths[tile];
                weightElements += ((lengths[tile] + 31) / 32 * 32) * input_channels;
            }
            if (rows > input_channels || weights == NULL || fields == NULL || weight_count != weightElements || field_count != input_channels) {
                return fail(error, error_capacity, @"Invalid weight, field, or tile buffer");
            }
            for (size_t i = 0; i < weight_count; ++i) {
                if (weights[i] < -1 || weights[i] > 1) return fail(error, error_capacity, @"Weights must be -1, 0, or 1");
            }
            for (size_t i = 0; i < field_count; ++i) {
                if (fields[i] < -1 || fields[i] > 1) return fail(error, error_capacity, @"Fields must be -1, 0, or 1");
            }
            // Keep the framework loaded for the process lifetime: its Objective-C
            // classes and service callbacks may outlive an individual program.
            static void *framework;
            static dispatch_once_t once;
            dispatch_once(&once, ^{
                framework = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
            });
            if (framework == NULL) return fail(error, error_capacity, @"AppleNeuralEngine framework unavailable");
            Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
            Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
            Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
            Class requestClass = NSClassFromString(@"_ANERequest");
            requireSelector(descriptorClass, @selector(alloc));
            requireSelector(modelClass, @selector(inMemoryModelWithDescriptor:));
            requireSelector(surfaceClass, @selector(objectWithIOSurface:));
            requireSelector(requestClass, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:));
            if (![descriptorClass instancesRespondToSelector:@selector(initWithNetworkText:weights:optionsPlist:isMILModel:)]) {
                return fail(error, error_capacity, @"ANE descriptor initializer unavailable");
            }
            NSError *nativeError = nil;
            NSData *plist = [NSPropertyListSerialization dataWithPropertyList:@{} format:NSPropertyListXMLFormat_v1_0 options:0 error:&nativeError];
            if (plist == nil) return fail(error, error_capacity, describe(@"Serialize ANE options", nativeError));
            NSData *mil = [makeMIL(input_channels, lanes, lengths, tile_count, sweeps, fields, weights) dataUsingEncoding:NSUTF8StringEncoding];
            id descriptor = [[descriptorClass alloc] initWithNetworkText:mil weights:@{} optionsPlist:plist isMILModel:YES];
            if (descriptor == nil) return fail(error, error_capacity, @"ANE descriptor creation failed");
            QuipAneProgram *result = [QuipAneProgram new];
            result->inputElements = inputElements;
            result->sweeps = sweeps;
            result->queue = dispatch_queue_create("org.quip.ane-program", DISPATCH_QUEUE_SERIAL);
            result->group = dispatch_group_create();
            result.model = [modelClass inMemoryModelWithDescriptor:descriptor];
            requireSelector(result.model, @selector(hexStringIdentifier));
            NSString *identifier = [result.model hexStringIdentifier];
            NSCharacterSet *invalidIdentifier = [[NSCharacterSet characterSetWithCharactersInString:@"0123456789abcdefABCDEF_"] invertedSet];
            if (![identifier isKindOfClass:NSString.class] || identifier.length == 0 ||
                [identifier rangeOfCharacterFromSet:invalidIdentifier].location != NSNotFound) {
                return fail(error, error_capacity, @"Invalid ANE model identifier");
            }
            NSString *directory = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
            // Refuse existing paths. Only a directory created by this program
            // becomes eligible for cleanup, even if an identity collision occurs.
            if (mkdir(directory.fileSystemRepresentation, 0700) != 0) {
                int savedErrno = errno;
                return fail(error, error_capacity, [NSString stringWithFormat:@"Create ANE staging directory: %s", strerror(savedErrno)]);
            }
            result.directory = directory;
            NSString *weightDirectory = [directory stringByAppendingPathComponent:@"weights"];
            if (![NSFileManager.defaultManager createDirectoryAtPath:weightDirectory withIntermediateDirectories:NO attributes:nil error:&nativeError] ||
                ![mil writeToFile:[directory stringByAppendingPathComponent:@"model.mil"] options:NSDataWritingAtomic error:&nativeError]) {
                return fail(error, error_capacity, describe(@"Stage ANE program", nativeError));
            }
            NSString *weightPath = [weightDirectory stringByAppendingPathComponent:@"weight_data.bin"];
            if (![makeWeightBlob(weights, input_channels, lengths, tile_count) writeToFile:weightPath options:NSDataWritingAtomic error:&nativeError]) return fail(error, error_capacity, describe(@"Stage ANE weights", nativeError));
            requireSelector(result.model, @selector(compileWithQoS:options:error:));
            if (![result.model compileWithQoS:21 options:@{} error:&nativeError]) return fail(error, error_capacity, describe(@"Compile ANE program", nativeError));
            requireSelector(result.model, @selector(loadWithQoS:options:error:));
            requireSelector(result.model, @selector(unloadWithQoS:error:));
            requireSelector(result.model, @selector(evaluateWithQoS:options:request:error:));
            if (![result.model loadWithQoS:21 options:@{} error:&nativeError]) return fail(error, error_capacity, describe(@"Load ANE program", nativeError));
            result.loaded = YES;
            if (![result removeDirectory:&nativeError]) return fail(error, error_capacity, describe(@"Remove ANE staging directory", nativeError));
            NSMutableArray *wrappers = [NSMutableArray new];
            for (size_t i = 0; i < 2 + 2 * sweeps; ++i) {
                IOSurfaceRef surface = makeSurface(inputElements);
                result->surfaces[i] = surface;
                if (surface == NULL) return fail(error, error_capacity, @"ANE IOSurface allocation failed");
                id wrapper = [surfaceClass objectWithIOSurface:surface];
                if (wrapper == nil) return fail(error, error_capacity, @"ANE IOSurface wrapping failed");
                [wrappers addObject:wrapper];
            }
            result.wrappers = wrappers;
            NSMutableArray *requests = [NSMutableArray new];
            for (size_t bank = 0; bank < 2; ++bank) {
                for (size_t state = 0; state < 2; ++state) {
                    NSMutableArray *inputs = [NSMutableArray arrayWithObject:wrappers[state]];
                    NSMutableArray *indices = [NSMutableArray arrayWithObject:@0];
                    for (size_t sweep = 0; sweep < sweeps; ++sweep) {
                        [inputs addObject:wrappers[2 + bank * sweeps + sweep]];
                        [indices addObject:@(sweep + 1)];
                    }
                    id request = [requestClass requestWithInputs:inputs inputIndices:indices
                        outputs:@[wrappers[1 - state]] outputIndices:@[@0]
                        weightsBuffer:nil perfStats:nil procedureIndex:@0];
                    if (request == nil) return fail(error, error_capacity, @"ANE request creation failed");
                    [requests addObject:request];
                }
            }
            result.requests = requests;
            *program = (__bridge_retained void *)result;
            return 0;
        } @catch (NSException *exception) {
            return fail(error, error_capacity, [NSString stringWithFormat:@"ANE create exception: %@", exception.reason]);
        }
    }
}

void quip_ane_convert_thresholds(const uint8_t *thresholds, uint16_t *output_bits, size_t count) {
    size_t index = 0;
    const uint16x8_t skippedValue = vreinterpretq_u16_s16(vdupq_n_s16(-128));
    for (; index + 8 <= count; index += 8) {
        uint16x8_t values = vmovl_u8(vld1_u8(thresholds + index));
        uint16x8_t skipped = vceqq_u16(values, vdupq_n_u16(255));
        int16x8_t normalized = vreinterpretq_s16_u16(vbslq_u16(skipped, skippedValue, values));
        vst1q_u16(output_bits + index, vreinterpretq_u16_f16(vcvtq_f16_s16(normalized)));
    }
    for (; index < count; ++index) {
        _Float16 value = (_Float16)(thresholds[index] == 255 ? -128 : thresholds[index]);
        memcpy(output_bits + index, &value, sizeof(value));
    }
}

static void convertSpins(const int8_t *spins, uint16_t *outputBits, size_t count) {
    size_t index = 0;
    for (; index + 8 <= count; index += 8) {
        int16x8_t values = vmovl_s8(vld1_s8(spins + index));
        vst1q_u16(outputBits + index, vreinterpretq_u16_f16(vcvtq_f16_s16(values)));
    }
    for (; index < count; ++index) {
        _Float16 value = (_Float16)spins[index];
        memcpy(outputBits + index, &value, sizeof(value));
    }
}

static BOOL stageSurface(IOSurfaceRef surface, const void *values, size_t count, BOOL thresholds, NSString **error) {
    IOReturn status = IOSurfaceLock(surface, 0, NULL);
    if (status != kIOReturnSuccess) { *error = @"ANE input lock failed"; return NO; }
    BOOL valid = YES;
    @try {
        uint16_t *destination = IOSurfaceGetBaseAddress(surface);
        if (destination == NULL) { *error = @"ANE input IOSurface has no base address"; valid = NO; }
        else if (thresholds) {
            // 255 represents a skipped sweep. -128 keeps every integer margin
            // negative even at the maximum supported signed local field, 21.
            quip_ane_convert_thresholds(values, destination, count);
        } else convertSpins(values, destination, count);
    } @finally {
        status = IOSurfaceUnlock(surface, 0, NULL);
        if (status != kIOReturnSuccess) { *error = @"ANE input unlock failed"; valid = NO; }
    }
    return valid;
}

static BOOL finishDispatch(QuipAneProgram *owned, QuipAneTimes *times, NSString **error) {
    if (!owned->pending) return YES;
    dispatch_group_wait(owned->group, DISPATCH_TIME_FOREVER);
    owned->pending = NO;
    times->dispatch_us += owned->pendingDispatchUS;
    if (owned.pendingError != nil) {
        *error = owned.pendingError;
        owned.pendingError = nil;
        return NO;
    }
    return YES;
}

int32_t quip_ane_reset(void *program, const int8_t *spins, size_t count, char *error, size_t error_capacity) {
    @autoreleasepool { @try {
        if (error == NULL || error_capacity == 0) return 1;
        error[0] = '\0';
        if (program == NULL || spins == NULL) return fail(error, error_capacity, @"Missing ANE reset pointer");
        QuipAneProgram *owned = (__bridge QuipAneProgram *)program;
        if (!owned.loaded || count != owned->inputElements) return fail(error, error_capacity, @"Invalid ANE reset dimensions");
        for (size_t i = 0; i < count; ++i) if (spins[i] != -1 && spins[i] != 1) return fail(error, error_capacity, @"ANE spin must be -1 or +1");
        QuipAneTimes ignored = {0, 0};
        NSString *pendingError = nil;
        if (!finishDispatch(owned, &ignored, &pendingError)) return fail(error, error_capacity, pendingError);
        owned->initialized = NO;
        NSString *stagingError = nil;
        if (!stageSurface(owned->surfaces[0], spins, count, NO, &stagingError)) return fail(error, error_capacity, stagingError);
        owned->current = 0;
        owned->initialized = YES;
        return 0;
    } @catch (NSException *exception) { return fail(error, error_capacity, [NSString stringWithFormat:@"ANE reset exception: %@", exception.reason]); } }
}

int32_t quip_ane_submit(void *program, const uint8_t *thresholds, size_t threshold_count,
    QuipAneTimes *times, char *error, size_t error_capacity) {
    @autoreleasepool { @try {
        if (error == NULL || error_capacity == 0) return 1;
        error[0] = '\0';
        if (program == NULL || thresholds == NULL || times == NULL) return fail(error, error_capacity, @"Missing ANE evaluation pointer");
        *times = (QuipAneTimes){0, 0};
        QuipAneProgram *owned = (__bridge QuipAneProgram *)program;
        if (!owned.loaded || (!owned->pending && !owned->initialized) || threshold_count != owned->inputElements * owned->sweeps) return fail(error, error_capacity, @"Invalid ANE evaluation dimensions or uninitialized state");
        uint64_t stagingStart = monotonicUS();
        for (size_t i = 0; i < threshold_count; ++i) if (thresholds[i] > 63 && thresholds[i] != 255) return fail(error, error_capacity, @"ANE threshold must be 0..63 or skipped (255)");
        NSString *stagingError = nil;
        size_t bank = owned->nextThresholdBank;
        for (size_t sweep = 0; sweep < owned->sweeps; ++sweep) {
            if (!stageSurface(owned->surfaces[2 + bank * owned->sweeps + sweep], thresholds + sweep * owned->inputElements, owned->inputElements, YES, &stagingError)) {
                NSString *pendingError = nil;
                (void)finishDispatch(owned, times, &pendingError);
                return fail(error, error_capacity, pendingError != nil ? pendingError : stagingError);
            }
        }
        times->staging_us = monotonicUS() - stagingStart;
        NSString *pendingError = nil;
        if (!finishDispatch(owned, times, &pendingError)) return fail(error, error_capacity, pendingError);
        id request = owned.requests[bank * 2 + owned->current];
        owned->nextThresholdBank = 1 - bank;
        owned->pendingDispatchUS = 0;
        owned.pendingError = nil;
        owned->pending = YES;
        dispatch_group_enter(owned->group);
        dispatch_async(owned->queue, ^{
            @autoreleasepool {
                uint64_t dispatchStart = monotonicUS();
                @try {
                    NSError *nativeError = nil;
                    BOOL completed = [owned.model evaluateWithQoS:21 options:@{} request:request error:&nativeError];
                    if (completed) {
                        owned->current = 1 - owned->current;
                        owned->initialized = YES;
                    } else {
                        owned->initialized = NO;
                        owned.pendingError = describe(@"Evaluate ANE program", nativeError);
                    }
                } @catch (NSException *exception) {
                    owned->initialized = NO;
                    owned.pendingError = [NSString stringWithFormat:@"ANE evaluate exception: %@", exception.reason];
                } @finally {
                    owned->pendingDispatchUS = monotonicUS() - dispatchStart;
                    dispatch_group_leave(owned->group);
                }
            }
        });
        return 0;
    } @catch (NSException *exception) { return fail(error, error_capacity, [NSString stringWithFormat:@"ANE evaluate exception: %@", exception.reason]); } }
}

int32_t quip_ane_finish(void *program, QuipAneTimes *times, char *error, size_t error_capacity) {
    @autoreleasepool { @try {
        if (error == NULL || error_capacity == 0) return 1;
        error[0] = '\0';
        if (program == NULL || times == NULL) return fail(error, error_capacity, @"Missing ANE finish pointer");
        *times = (QuipAneTimes){0, 0};
        QuipAneProgram *owned = (__bridge QuipAneProgram *)program;
        NSString *pendingError = nil;
        if (!finishDispatch(owned, times, &pendingError)) return fail(error, error_capacity, pendingError);
        return 0;
    } @catch (NSException *exception) { return fail(error, error_capacity, [NSString stringWithFormat:@"ANE finish exception: %@", exception.reason]); } }
}

int32_t quip_ane_evaluate(void *program, const uint8_t *thresholds, size_t threshold_count,
    QuipAneTimes *times, char *error, size_t error_capacity) {
    if (error == NULL || error_capacity == 0) return 1;
    error[0] = '\0';
    if (times == NULL) return fail(error, error_capacity, @"Missing ANE evaluation timing pointer");
    QuipAneTimes submitted = {0, 0};
    if (quip_ane_submit(program, thresholds, threshold_count, &submitted, error, error_capacity) != 0) return 1;
    QuipAneTimes finished = {0, 0};
    if (quip_ane_finish(program, &finished, error, error_capacity) != 0) return 1;
    *times = (QuipAneTimes){submitted.staging_us + finished.staging_us, submitted.dispatch_us + finished.dispatch_us};
    return 0;
}

int32_t quip_ane_read(void *program, int8_t *output, size_t count, char *error, size_t error_capacity) {
    @autoreleasepool { @try {
        if (error == NULL || error_capacity == 0) return 1;
        error[0] = '\0';
        if (program == NULL || output == NULL) return fail(error, error_capacity, @"Missing ANE read pointer");
        QuipAneProgram *owned = (__bridge QuipAneProgram *)program;
        if (!owned.loaded || count != owned->inputElements) return fail(error, error_capacity, @"Invalid ANE read dimensions or uninitialized state");
        QuipAneTimes ignored = {0, 0};
        NSString *pendingError = nil;
        if (!finishDispatch(owned, &ignored, &pendingError)) return fail(error, error_capacity, pendingError);
        if (!owned->initialized) return fail(error, error_capacity, @"Invalid ANE read dimensions or uninitialized state");
        IOSurfaceRef surface = owned->surfaces[owned->current];
        IOReturn status = IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL);
        if (status != kIOReturnSuccess) return fail(error, error_capacity, @"ANE output lock failed");
        NSString *outputError = nil;
        @try {
            const _Float16 *source = IOSurfaceGetBaseAddress(surface);
            if (source == NULL) outputError = @"ANE output IOSurface has no base address";
            else for (size_t i = 0; i < count; ++i) {
                if (source[i] != (_Float16)-1 && source[i] != (_Float16)1) { outputError = [NSString stringWithFormat:@"ANE non-spin output at %zu: %g", i, (double)source[i]]; break; }
                output[i] = (int8_t)source[i];
            }
        } @finally {
            status = IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
            if (status != kIOReturnSuccess) outputError = @"ANE output unlock failed";
        }
        if (outputError != nil) { owned->initialized = NO; return fail(error, error_capacity, outputError); }
        return 0;
    } @catch (NSException *exception) { return fail(error, error_capacity, [NSString stringWithFormat:@"ANE read exception: %@", exception.reason]); } }
}

int32_t quip_ane_destroy(void *program, char *error, size_t error_capacity) {
    @autoreleasepool {
        @try {
            if (program == NULL) return fail(error, error_capacity, @"Missing ANE program handle");
            // Consumes create's retain before any operation that can fail.
            QuipAneProgram *owned = CFBridgingRelease(program);
            QuipAneTimes ignored = {0, 0};
            NSString *pendingError = nil;
            BOOL completed = finishDispatch(owned, &ignored, &pendingError);
            NSError *nativeError = nil;
            if (![owned unload:&nativeError]) return fail(error, error_capacity, describe(@"Unload ANE program", nativeError));
            if (!completed) return fail(error, error_capacity, pendingError);
            if (error == NULL || error_capacity == 0) return 1;
            error[0] = '\0';
            return 0;
        } @catch (NSException *exception) {
            return fail(error, error_capacity, [NSString stringWithFormat:@"ANE destroy exception: %@", exception.reason]);
        }
    }
}

uint32_t quip_ane_parent_pid(void) {
    @autoreleasepool {
        @try {
            return (uint32_t)getppid();
        } @catch (NSException *exception) {
            (void)exception;
            return 0;
        }
    }
}
