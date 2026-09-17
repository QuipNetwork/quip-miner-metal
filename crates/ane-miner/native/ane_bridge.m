#import <Foundation/Foundation.h>
#import <IOSurface/IOSurface.h>
#import <CoreFoundation/CFByteOrder.h>
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
    BOOL initialized;
    IOSurfaceRef surfaces[10];
}
@property(nonatomic, strong) id model;
@property(nonatomic, strong) NSArray *requests;
@property(nonatomic, strong) NSArray *wrappers;
@property(nonatomic, strong) NSString *directory;
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
    for (size_t i = 0; i < 10; ++i) {
        if (surfaces[i] != NULL) CFRelease(surfaces[i]);
    }
}
@end

static NSString *shape(size_t channels) {
    return [NSString stringWithFormat:@"tensor<fp16, [1, %zu, 1, 128]>", channels];
}

static void slice(NSMutableString *mil, NSString *name, NSString *source, size_t begin, size_t count) {
    [mil appendFormat:@"    tensor<int32, [4]> %@begin = const()[name=string(\"%@begin\"), val=tensor<int32, [4]>([0,%zu,0,0])];\n", name, name, begin];
    [mil appendFormat:@"    tensor<int32, [4]> %@size = const()[name=string(\"%@size\"), val=tensor<int32, [4]>([1,%zu,1,128])];\n", name, name, count];
    [mil appendFormat:@"    %@ %@ = slice_by_size(x=%@, begin=%@begin, size=%@size)[name=string(\"%@\")];\n", shape(count), name, source, name, name, name];
}

static NSString *makeMIL(size_t channels, const size_t *lengths, size_t tiles, size_t sweeps, const int8_t *fields) {
    NSMutableString *mil = [NSMutableString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}, {\"quip-ane-msa\", \"%@\"}})]\n{\n  func main<ios18>(%@ a_state", NSUUID.UUID.UUIDString, shape(channels)];
    for (size_t sweep = 0; sweep < sweeps; ++sweep) [mil appendFormat:@", %@ t%zu", shape(channels), sweep];
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
    size_t begin = 0, blobOffset = 64;
    for (size_t tile = 0; tile < tiles; ++tile) {
        size_t count = lengths[tile], padded = (count + 31) / 32 * 32;
        [mil appendFormat:@"    tensor<fp16, [%zu, %zu, 1, 1]> w%zu = const()[name=string(\"w%zu\"), val=tensor<fp16, [%zu, %zu, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/weight_data.bin\"), offset=uint64(%zu)))];\n", padded, channels, tile, tile, padded, channels, blobOffset];
        blobOffset += 64 + padded * channels * sizeof(_Float16);
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
            slice(mil, own, state, begin, count);
            slice(mil, threshold, [NSString stringWithFormat:@"t%zu", sweep], begin, count);
            [mil appendFormat:@"    %@ %@ = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w%zu, x=%@)[name=string(\"%@\")];\n", shape(padded), raw, tile, state, raw];
            slice(mil, js, raw, 0, count);
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
            for (NSArray *op in ops) [mil appendFormat:@"    %@ %@%@ = %@[name=string(\"%@%@\")];\n", shape(count), prefix, op[0], op[1], prefix, op[0]];
            NSMutableArray *parts = [NSMutableArray new];
            if (begin > 0) {
                NSString *head = [prefix stringByAppendingString:@"head"];
                slice(mil, head, state, 0, begin);
                [parts addObject:head];
            }
            [parts addObject:[prefix stringByAppendingString:@"updated"]];
            if (begin + count < channels) {
                NSString *tail = [prefix stringByAppendingString:@"tail"];
                slice(mil, tail, state, begin + count, channels - begin - count);
                [parts addObject:tail];
            }
            if (parts.count == 1) {
                state = parts[0];
            } else {
                state = [prefix stringByAppendingString:@"state"];
                [mil appendFormat:@"    %@ %@ = concat(values=(%@), axis=axis, interleave=interleave)[name=string(\"%@\")];\n", shape(channels), state, [parts componentsJoinedByString:@", "], state];
            }
            begin += count;
        }
    }
    [mil appendFormat:@"  } -> (%@);\n}\n", state];
    return mil;
}

static NSData *makeWeightBlob(const int8_t *weights, size_t channels, const size_t *lengths, size_t tiles, size_t count) {
    NSMutableData *blob = [NSMutableData dataWithLength:64 + 64 * tiles + count * sizeof(_Float16)];
    uint8_t *bytes = blob.mutableBytes;
    uint32_t chunkCount = CFSwapInt32HostToLittle((uint32_t)tiles);
    memcpy(bytes, &chunkCount, sizeof(chunkCount));
    bytes[4] = 2;
    size_t offset = 64, weightOffset = 0;
    for (size_t tile = 0; tile < tiles; ++tile) {
        size_t elements = ((lengths[tile] + 31) / 32 * 32) * channels;
        const size_t positions[] = {0, 4};
        const uint32_t values[] = {0xDEADBEEF, 1};
        for (size_t i = 0; i < 2; ++i) {
            uint32_t little = CFSwapInt32HostToLittle(values[i]);
            memcpy(bytes + offset + positions[i], &little, sizeof(little));
        }
        uint64_t size = CFSwapInt64HostToLittle(elements * sizeof(_Float16));
        uint64_t payload = CFSwapInt64HostToLittle(offset + 64);
        memcpy(bytes + offset + 8, &size, sizeof(size));
        memcpy(bytes + offset + 16, &payload, sizeof(payload));
        for (size_t i = 0; i < elements; ++i) {
            _Float16 value = (_Float16)weights[weightOffset + i];
            uint16_t bits;
            memcpy(&bits, &value, sizeof(bits));
            bits = CFSwapInt16HostToLittle(bits);
            memcpy(bytes + offset + 64 + i * sizeof(bits), &bits, sizeof(bits));
        }
        weightOffset += elements;
        offset += 64 + elements * sizeof(_Float16);
    }
    return blob;
}

int32_t quip_ane_create(size_t input_channels, const size_t *lengths, size_t tile_count, size_t sweeps,
    const int8_t *weights, size_t weight_count, const int8_t *fields, size_t field_count,
    void **program, char *error, size_t error_capacity) {
    @autoreleasepool {
        @try {
            if (program == NULL) return fail(error, error_capacity, @"Missing program output pointer");
            *program = NULL;
            if (error == NULL || error_capacity == 0) return 1;
            error[0] = '\0';
            size_t inputElements, allocationBytes;
            if (input_channels < 32 || input_channels > 16384 || input_channels % 32 != 0 ||
                tile_count == 0 || tile_count > 24 || lengths == NULL || sweeps == 0 || sweeps > 8 ||
                !multiply(input_channels, 128, &inputElements) || !surfaceBytes(inputElements, &allocationBytes)) {
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
            NSData *mil = [makeMIL(input_channels, lengths, tile_count, sweeps, fields) dataUsingEncoding:NSUTF8StringEncoding];
            id descriptor = [[descriptorClass alloc] initWithNetworkText:mil weights:@{} optionsPlist:plist isMILModel:YES];
            if (descriptor == nil) return fail(error, error_capacity, @"ANE descriptor creation failed");
            QuipAneProgram *result = [QuipAneProgram new];
            result->inputElements = inputElements;
            result->sweeps = sweeps;
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
            if (![makeWeightBlob(weights, input_channels, lengths, tile_count, weight_count) writeToFile:weightPath options:NSDataWritingAtomic error:&nativeError]) return fail(error, error_capacity, describe(@"Stage ANE weights", nativeError));
            requireSelector(result.model, @selector(compileWithQoS:options:error:));
            if (![result.model compileWithQoS:21 options:@{} error:&nativeError]) return fail(error, error_capacity, describe(@"Compile ANE program", nativeError));
            requireSelector(result.model, @selector(loadWithQoS:options:error:));
            requireSelector(result.model, @selector(unloadWithQoS:error:));
            requireSelector(result.model, @selector(evaluateWithQoS:options:request:error:));
            if (![result.model loadWithQoS:21 options:@{} error:&nativeError]) return fail(error, error_capacity, describe(@"Load ANE program", nativeError));
            result.loaded = YES;
            if (![result removeDirectory:&nativeError]) return fail(error, error_capacity, describe(@"Remove ANE staging directory", nativeError));
            NSMutableArray *wrappers = [NSMutableArray new];
            for (size_t i = 0; i < sweeps + 2; ++i) {
                IOSurfaceRef surface = makeSurface(inputElements);
                result->surfaces[i] = surface;
                if (surface == NULL) return fail(error, error_capacity, @"ANE IOSurface allocation failed");
                id wrapper = [surfaceClass objectWithIOSurface:surface];
                if (wrapper == nil) return fail(error, error_capacity, @"ANE IOSurface wrapping failed");
                [wrappers addObject:wrapper];
            }
            result.wrappers = wrappers;
            NSMutableArray *requests = [NSMutableArray new];
            for (size_t state = 0; state < 2; ++state) {
                NSMutableArray *inputs = [NSMutableArray arrayWithObject:wrappers[state]];
                NSMutableArray *indices = [NSMutableArray arrayWithObject:@0];
                for (size_t sweep = 0; sweep < sweeps; ++sweep) {
                    [inputs addObject:wrappers[sweep + 2]];
                    [indices addObject:@(sweep + 1)];
                }
                id request = [requestClass requestWithInputs:inputs inputIndices:indices
                    outputs:@[wrappers[1 - state]] outputIndices:@[@0]
                    weightsBuffer:nil perfStats:nil procedureIndex:@0];
                if (request == nil) return fail(error, error_capacity, @"ANE request creation failed");
                [requests addObject:request];
            }
            result.requests = requests;
            *program = (__bridge_retained void *)result;
            return 0;
        } @catch (NSException *exception) {
            return fail(error, error_capacity, [NSString stringWithFormat:@"ANE create exception: %@", exception.reason]);
        }
    }
}

static BOOL stageSurface(IOSurfaceRef surface, const void *values, size_t count, BOOL thresholds, NSString **error) {
    IOReturn status = IOSurfaceLock(surface, 0, NULL);
    if (status != kIOReturnSuccess) { *error = @"ANE input lock failed"; return NO; }
    BOOL valid = YES;
    @try {
        _Float16 *destination = IOSurfaceGetBaseAddress(surface);
        if (destination == NULL) { *error = @"ANE input IOSurface has no base address"; valid = NO; }
        else for (size_t i = 0; i < count; ++i) {
            // 255 represents a skipped sweep. -128 keeps every integer margin
            // negative even at the maximum supported signed local field, 21.
            int value = thresholds ? ((const uint8_t *)values)[i] : ((const int8_t *)values)[i];
            destination[i] = (_Float16)(thresholds && value == 255 ? -128 : value);
        }
    } @finally {
        status = IOSurfaceUnlock(surface, 0, NULL);
        if (status != kIOReturnSuccess) { *error = @"ANE input unlock failed"; valid = NO; }
    }
    return valid;
}

int32_t quip_ane_reset(void *program, const int8_t *spins, size_t count, char *error, size_t error_capacity) {
    @autoreleasepool { @try {
        if (error == NULL || error_capacity == 0) return 1;
        error[0] = '\0';
        if (program == NULL || spins == NULL) return fail(error, error_capacity, @"Missing ANE reset pointer");
        QuipAneProgram *owned = (__bridge QuipAneProgram *)program;
        if (!owned.loaded || count != owned->inputElements) return fail(error, error_capacity, @"Invalid ANE reset dimensions");
        for (size_t i = 0; i < count; ++i) if (spins[i] != -1 && spins[i] != 1) return fail(error, error_capacity, @"ANE spin must be -1 or +1");
        owned->initialized = NO;
        NSString *stagingError = nil;
        if (!stageSurface(owned->surfaces[0], spins, count, NO, &stagingError)) return fail(error, error_capacity, stagingError);
        owned->current = 0;
        owned->initialized = YES;
        return 0;
    } @catch (NSException *exception) { return fail(error, error_capacity, [NSString stringWithFormat:@"ANE reset exception: %@", exception.reason]); } }
}

int32_t quip_ane_evaluate(void *program, const uint8_t *thresholds, size_t threshold_count,
    QuipAneTimes *times, char *error, size_t error_capacity) {
    @autoreleasepool { @try {
        if (error == NULL || error_capacity == 0) return 1;
        error[0] = '\0';
        if (program == NULL || thresholds == NULL || times == NULL) return fail(error, error_capacity, @"Missing ANE evaluation pointer");
        *times = (QuipAneTimes){0, 0};
        QuipAneProgram *owned = (__bridge QuipAneProgram *)program;
        if (!owned.loaded || !owned->initialized || threshold_count != owned->inputElements * owned->sweeps) return fail(error, error_capacity, @"Invalid ANE evaluation dimensions or uninitialized state");
        uint64_t stagingStart = monotonicUS();
        for (size_t i = 0; i < threshold_count; ++i) if (thresholds[i] > 63 && thresholds[i] != 255) return fail(error, error_capacity, @"ANE threshold must be 0..63 or skipped (255)");
        NSString *stagingError = nil;
        for (size_t sweep = 0; sweep < owned->sweeps; ++sweep) {
            if (!stageSurface(owned->surfaces[sweep + 2], thresholds + sweep * owned->inputElements, owned->inputElements, YES, &stagingError)) return fail(error, error_capacity, stagingError);
        }
        times->staging_us = monotonicUS() - stagingStart;
        NSError *nativeError = nil;
        uint64_t dispatchStart = monotonicUS();
        owned->initialized = NO;
        BOOL completed = [owned.model evaluateWithQoS:21 options:@{} request:owned.requests[owned->current] error:&nativeError];
        times->dispatch_us = monotonicUS() - dispatchStart;
        if (!completed) { owned->initialized = NO; return fail(error, error_capacity, describe(@"Evaluate ANE program", nativeError)); }
        owned->current = 1 - owned->current;
        owned->initialized = YES;
        return 0;
    } @catch (NSException *exception) { return fail(error, error_capacity, [NSString stringWithFormat:@"ANE evaluate exception: %@", exception.reason]); } }
}

int32_t quip_ane_read(void *program, int8_t *output, size_t count, char *error, size_t error_capacity) {
    @autoreleasepool { @try {
        if (error == NULL || error_capacity == 0) return 1;
        error[0] = '\0';
        if (program == NULL || output == NULL) return fail(error, error_capacity, @"Missing ANE read pointer");
        QuipAneProgram *owned = (__bridge QuipAneProgram *)program;
        if (!owned.loaded || !owned->initialized || count != owned->inputElements) return fail(error, error_capacity, @"Invalid ANE read dimensions or uninitialized state");
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
            NSError *nativeError = nil;
            if (![owned unload:&nativeError]) return fail(error, error_capacity, describe(@"Unload ANE program", nativeError));
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
