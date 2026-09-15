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
    size_t outputElements;
    IOSurfaceRef surfaces[4];
}
@property(nonatomic, strong) id model;
@property(nonatomic, strong) id request;
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
    _request = nil;
    _wrappers = nil;
    _model = nil;
    for (size_t i = 0; i < 4; ++i) {
        if (surfaces[i] != NULL) CFRelease(surfaces[i]);
    }
}
@end

static NSString *makeMIL(size_t inputChannels, size_t outputChannels,
                         const int8_t *fields) {
    NSMutableArray<NSString *> *values = [NSMutableArray arrayWithCapacity:outputChannels];
    for (size_t i = 0; i < outputChannels; ++i) {
        [values addObject:[NSString stringWithFormat:@"%d.0", (int)fields[i]]];
    }
    NSString *shape = [NSString stringWithFormat:
        @"tensor<fp16, [1, %zu, 1, 128]>", outputChannels];
    NSMutableString *mil = [NSMutableString stringWithFormat:
        @"program(1.3)\n"
         "[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, "
         "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, "
         "{\"coremltools-version\", \"9.0\"}, {\"quip-ane-msa\", \"%@\"}})]\n"
         "{\n  func main<ios18>(tensor<fp16, [1, %zu, 1, 128]> a_neighbors, "
         "%@ b_spins, %@ c_thresholds) {\n",
        NSUUID.UUID.UUIDString, inputChannels, shape, shape];
    [mil appendFormat:
        @"    tensor<fp16, [%zu, %zu, 1, 1]> w = const()[name=string(\"w\"), "
         "val=tensor<fp16, [%zu, %zu, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/weight_data.bin\"), offset=uint64(64)))];\n",
        outputChannels, inputChannels, outputChannels, inputChannels];
    [mil appendFormat:
        @"    tensor<fp16, [%zu]> hFlat = const()[name=string(\"hFlat\"), val=tensor<fp16, [%zu]>([%@])];\n",
        outputChannels, outputChannels, [values componentsJoinedByString:@","]];
    [mil appendFormat:
        @"    tensor<int32, [4]> hShape = const()[name=string(\"hShape\"), val=tensor<int32, [4]>([1,%zu,1,1])];\n"
         "    tensor<fp16, [1,%zu,1,1]> h = reshape(x=hFlat, shape=hShape)[name=string(\"h\")];\n",
        outputChannels, outputChannels];
    [mil appendString:
        @"    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
         "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
         "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
         "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
         "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
         "    fp16 zero = const()[name=string(\"zero\"), val=fp16(0.0)];\n"
         "    fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];\n"
         "    fp16 minusTwo = const()[name=string(\"minusTwo\"), val=fp16(-2.0)];\n"];
    NSArray<NSArray<NSString *> *> *operations = @[
        @[@"js", @"conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w, x=a_neighbors)"],
        @[@"field", @"add(x=js, y=h)"],
        @[@"signedField", @"mul(x=b_spins, y=field)"],
        @[@"margin", @"add(x=c_thresholds, y=signedField)"],
        @[@"shifted", @"add(x=margin, y=one)"],
        @[@"accept", @"clip(x=shifted, alpha=zero, beta=one)"],
        @[@"negativeMask", @"mul(x=accept, y=minusTwo)"],
        @[@"factor", @"add(x=one, y=negativeMask)"],
        @[@"y", @"mul(x=b_spins, y=factor)"]
    ];
    for (NSArray<NSString *> *operation in operations) {
        [mil appendFormat:@"    %@ %@ = %@[name=string(\"%@\")];\n",
            shape, operation[0], operation[1], operation[0]];
    }
    [mil appendString:@"  } -> (y);\n}\n"];
    return mil;
}

static NSData *makeWeightBlob(const int8_t *weights, size_t count, size_t payloadBytes) {
    NSMutableData *blob = [NSMutableData dataWithLength:128 + payloadBytes];
    uint8_t *bytes = blob.mutableBytes;
    bytes[0] = 1;
    bytes[4] = 2;
    const size_t offsets[] = {64, 68, 72, 80};
    const uint32_t values[] = {0xDEADBEEF, 1, (uint32_t)payloadBytes, 128};
    for (size_t i = 0; i < 4; ++i) {
        uint32_t little = CFSwapInt32HostToLittle(values[i]);
        memcpy(bytes + offsets[i], &little, sizeof(little));
    }
    for (size_t i = 0; i < count; ++i) {
        _Float16 value = (_Float16)weights[i];
        uint16_t bits;
        memcpy(&bits, &value, sizeof(bits));
        bits = CFSwapInt16HostToLittle(bits);
        memcpy(bytes + 128 + i * sizeof(bits), &bits, sizeof(bits));
    }
    return blob;
}

int32_t quip_ane_create(size_t input_channels, size_t output_channels,
    const int8_t *weights, size_t weight_count, const int8_t *fields, size_t field_count,
    void **program, char *error, size_t error_capacity) {
    @autoreleasepool {
        @try {
            if (program == NULL) return fail(error, error_capacity, @"Missing program output pointer");
            *program = NULL;
            if (error == NULL || error_capacity == 0) return 1;
            error[0] = '\0';
            size_t weightElements, payloadBytes, inputElements, outputElements, allocationBytes;
            if (input_channels < 32 || input_channels > 16384 || input_channels % 32 != 0 ||
                output_channels < 32 || output_channels > 4096 || output_channels % 32 != 0 ||
                !multiply(input_channels, output_channels, &weightElements) ||
                !multiply(weightElements, sizeof(_Float16), &payloadBytes) ||
                payloadBytes > UINT32_MAX || payloadBytes > SIZE_MAX - 128 ||
                !multiply(input_channels, 128, &inputElements) ||
                !multiply(output_channels, 128, &outputElements) ||
                !surfaceBytes(inputElements, &allocationBytes) ||
                !surfaceBytes(outputElements, &allocationBytes)) {
                return fail(error, error_capacity, @"Invalid ANE channel dimensions or byte size");
            }
            if (weights == NULL || fields == NULL || weight_count != weightElements || field_count != output_channels) {
                return fail(error, error_capacity, @"Invalid weight or field buffer");
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
            NSData *mil = [makeMIL(input_channels, output_channels, fields) dataUsingEncoding:NSUTF8StringEncoding];
            id descriptor = [[descriptorClass alloc] initWithNetworkText:mil weights:@{} optionsPlist:plist isMILModel:YES];
            if (descriptor == nil) return fail(error, error_capacity, @"ANE descriptor creation failed");
            QuipAneProgram *result = [QuipAneProgram new];
            result->inputElements = inputElements;
            result->outputElements = outputElements;
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
                ![mil writeToFile:[directory stringByAppendingPathComponent:@"model.mil"] options:NSDataWritingAtomic error:&nativeError] ||
                ![makeWeightBlob(weights, weight_count, payloadBytes) writeToFile:[weightDirectory stringByAppendingPathComponent:@"weight_data.bin"] options:NSDataWritingAtomic error:&nativeError]) {
                return fail(error, error_capacity, describe(@"Stage ANE program", nativeError));
            }
            requireSelector(result.model, @selector(compileWithQoS:options:error:));
            if (![result.model compileWithQoS:21 options:@{} error:&nativeError]) return fail(error, error_capacity, describe(@"Compile ANE program", nativeError));
            requireSelector(result.model, @selector(loadWithQoS:options:error:));
            requireSelector(result.model, @selector(unloadWithQoS:error:));
            requireSelector(result.model, @selector(evaluateWithQoS:options:request:error:));
            if (![result.model loadWithQoS:21 options:@{} error:&nativeError]) return fail(error, error_capacity, describe(@"Load ANE program", nativeError));
            result.loaded = YES;
            if (![result removeDirectory:&nativeError]) return fail(error, error_capacity, describe(@"Remove ANE staging directory", nativeError));
            NSMutableArray *wrappers = [NSMutableArray arrayWithCapacity:4];
            for (size_t i = 0; i < 4; ++i) {
                result->surfaces[i] = makeSurface(i == 0 ? inputElements : outputElements);
                if (result->surfaces[i] == NULL) return fail(error, error_capacity, @"ANE IOSurface allocation failed");
                requireSelector(surfaceClass, @selector(objectWithIOSurface:));
                id wrapper = [surfaceClass objectWithIOSurface:result->surfaces[i]];
                if (wrapper == nil) return fail(error, error_capacity, @"ANE IOSurface wrapping failed");
                [wrappers addObject:wrapper];
            }
            result.wrappers = wrappers;
            requireSelector(requestClass, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:));
            result.request = [requestClass requestWithInputs:[wrappers subarrayWithRange:NSMakeRange(0, 3)]
                inputIndices:@[@0, @1, @2] outputs:@[wrappers[3]] outputIndices:@[@0]
                weightsBuffer:nil perfStats:nil procedureIndex:@0];
            if (result.request == nil) return fail(error, error_capacity, @"ANE request creation failed");
            *program = (__bridge_retained void *)result;
            return 0;
        } @catch (NSException *exception) {
            return fail(error, error_capacity, [NSString stringWithFormat:@"ANE create exception: %@", exception.reason]);
        }
    }
}

static BOOL stageSurface(IOSurfaceRef surface, const void *values, size_t count, BOOL unsignedValues, NSString **error) {
    IOReturn status = IOSurfaceLock(surface, 0, NULL);
    if (status != kIOReturnSuccess) {
        *error = [NSString stringWithFormat:@"ANE input lock failed: %d", status];
        return NO;
    }
    BOOL valid = YES;
    @try {
        _Float16 *destination = IOSurfaceGetBaseAddress(surface);
        if (destination == NULL) {
            *error = @"ANE input IOSurface has no base address";
            valid = NO;
        } else {
            for (size_t i = 0; i < count; ++i) {
                destination[i] = unsignedValues ? (_Float16)((const uint8_t *)values)[i] : (_Float16)((const int8_t *)values)[i];
            }
        }
    } @finally {
        status = IOSurfaceUnlock(surface, 0, NULL);
        if (status != kIOReturnSuccess) {
            *error = [NSString stringWithFormat:@"ANE input unlock failed: %d", status];
            valid = NO;
        }
    }
    return valid;
}

int32_t quip_ane_evaluate(void *program,
    const int8_t *neighbors, size_t neighbor_count, const int8_t *spins, size_t spin_count,
    const uint8_t *thresholds, size_t threshold_count, int8_t *output, size_t output_count,
    QuipAneTimes *times, char *error, size_t error_capacity) {
    @autoreleasepool {
        @try {
            if (error == NULL || error_capacity == 0) return 1;
            error[0] = '\0';
            if (program == NULL || neighbors == NULL || spins == NULL || thresholds == NULL || output == NULL || times == NULL) {
                return fail(error, error_capacity, @"Missing ANE evaluation pointer");
            }
            *times = (QuipAneTimes){0, 0};
            QuipAneProgram *owned = (__bridge QuipAneProgram *)program;
            if (!owned.loaded || neighbor_count != owned->inputElements || spin_count != owned->outputElements ||
                threshold_count != owned->outputElements || output_count != owned->outputElements) {
                return fail(error, error_capacity, @"Invalid ANE evaluation dimensions or unloaded program");
            }
            uint64_t stagingStart = monotonicUS();
            for (size_t i = 0; i < neighbor_count; ++i) {
                if (neighbors[i] < -21 || neighbors[i] > 21) return fail(error, error_capacity, @"ANE neighbor outside [-21, 21]");
            }
            for (size_t i = 0; i < spin_count; ++i) {
                if (spins[i] != -1 && spins[i] != 1) return fail(error, error_capacity, @"ANE own spin must be -1 or +1");
                if (thresholds[i] > 63) return fail(error, error_capacity, @"ANE threshold exceeds 63");
            }
            NSString *stagingError = nil;
            if (!stageSurface(owned->surfaces[0], neighbors, neighbor_count, NO, &stagingError) ||
                !stageSurface(owned->surfaces[1], spins, spin_count, NO, &stagingError) ||
                !stageSurface(owned->surfaces[2], thresholds, threshold_count, YES, &stagingError)) {
                return fail(error, error_capacity, stagingError);
            }
            times->staging_us = monotonicUS() - stagingStart;
            requireSelector(owned.model, @selector(evaluateWithQoS:options:request:error:));
            NSError *nativeError = nil;
            uint64_t dispatchStart = monotonicUS();
            BOOL completed = [owned.model evaluateWithQoS:21 options:@{} request:owned.request error:&nativeError];
            times->dispatch_us = monotonicUS() - dispatchStart;
            if (!completed) return fail(error, error_capacity, describe(@"Evaluate ANE program", nativeError));
            IOReturn status = IOSurfaceLock(owned->surfaces[3], kIOSurfaceLockReadOnly, NULL);
            if (status != kIOReturnSuccess) return fail(error, error_capacity, [NSString stringWithFormat:@"ANE output lock failed: %d", status]);
            NSString *outputError = nil;
            @try {
                const _Float16 *source = IOSurfaceGetBaseAddress(owned->surfaces[3]);
                if (source == NULL) {
                    outputError = @"ANE output IOSurface has no base address";
                } else {
                    for (size_t i = 0; i < output_count; ++i) {
                        if (source[i] != (_Float16)-1.0 && source[i] != (_Float16)1.0) {
                            outputError = [NSString stringWithFormat:@"ANE non-spin output at %zu: %g", i, (double)source[i]];
                            break;
                        }
                        output[i] = (int8_t)source[i];
                    }
                }
            } @finally {
                status = IOSurfaceUnlock(owned->surfaces[3], kIOSurfaceLockReadOnly, NULL);
                if (status != kIOReturnSuccess) outputError = [NSString stringWithFormat:@"ANE output unlock failed: %d", status];
            }
            if (outputError != nil) return fail(error, error_capacity, outputError);
            return 0;
        } @catch (NSException *exception) {
            return fail(error, error_capacity, [NSString stringWithFormat:@"ANE evaluate exception: %@", exception.reason]);
        }
    }
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
