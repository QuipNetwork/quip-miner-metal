// Isolated profiling runner. Production solver entry points are unchanged.
#include "../native/ane_bridge.m"
#include <sys/resource.h>
#include <objc/runtime.h>

@interface NSObject (ProbePerformance)
- (id)perfStats;
- (id)perfStatsArray;
- (uint64_t)hwExecutionTime;
- (id)performanceCounters;
- (void)setPerfStatsMask:(uint32_t)mask;
- (uint32_t)perfStatsMask;
- (id)model;
- (id)modelAttributes;
- (id)procedureInfoForProcedureIndex:(uint32_t)index;
- (id)sharedConnection;
- (id)shallowCopy;
+ (id)adapterWeightsAccessEntitlement;
+ (id)weightWithSymbolAndURL:(NSString *)symbol weightURL:(NSURL *)url;
+ (id)procedureDataWithSymbol:(NSString *)symbol weightArray:(NSArray *)weights;
+ (id)withProcedureData:(NSString *)name procedureArray:(NSArray *)procedures;
- (BOOL)loadModelNewInstance:(id)model options:(NSDictionary *)options modelInstParams:(id)parameters qos:(uint32_t)qos error:(NSError **)error;
- (BOOL)unloadModel:(id)model options:(NSDictionary *)options qos:(uint32_t)qos error:(NSError **)error;
+ (id)statsWithHardwareExecutionNS:(uint64_t)time;
- (void)setPerfStats:(id)stats;
@end

static void setStatsMask(id model, uint32_t mask) {
    if ([model respondsToSelector:@selector(setPerfStatsMask:)]) {
        [model setPerfStatsMask:mask];
        printf("stage=stats_mask class=%s requested=%u retained=%u\n", class_getName([model class]), mask, [model perfStatsMask]);
    } else printf("stage=stats_mask class=%s unavailable=1\n", class_getName([model class]));
}

static uint64_t processCPUUS(struct rusage usage) {
    return (uint64_t)usage.ru_utime.tv_sec * 1000000 + usage.ru_utime.tv_usec +
        (uint64_t)usage.ru_stime.tv_sec * 1000000 + usage.ru_stime.tv_usec;
}

static void require(BOOL condition, NSString *message) {
    if (!condition) @throw [NSException exceptionWithName:@"ProbeFailure" reason:message userInfo:nil];
}

static NSData *readData(NSString *path) {
    NSError *error = nil;
    NSData *data = [NSData dataWithContentsOfFile:path options:NSDataReadingMappedIfSafe error:&error];
    require(data != nil, describe(path, error));
    return data;
}

static NSString *resolvePath(NSString *root, NSString *path) {
    return path.isAbsolutePath ? path : [root stringByAppendingPathComponent:path];
}

int main(int argc, const char **argv) {
    setbuf(stdout, NULL);
    if (argc != 3) { fprintf(stderr, "usage: profile-runner manifest.json timeout-seconds\n"); return 2; }
    alarm((unsigned)strtoul(argv[2], NULL, 10));
    @autoreleasepool { @try {
        NSError *error = nil;
        NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:readData(@(argv[1])) options:0 error:&error];
        require(manifest != nil, describe(@"manifest", error));
        NSString *root = [@(argv[1]) stringByDeletingLastPathComponent];
        NSData *mil = readData([root stringByAppendingPathComponent:manifest[@"mil"]]);
        require(dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW) != NULL, @"ANE framework unavailable");
        NSDictionary *compilerOptions = manifest[@"compiler_options"] ?: @{};
        NSData *plist = [NSPropertyListSerialization dataWithPropertyList:compilerOptions format:NSPropertyListXMLFormat_v1_0 options:0 error:&error];
        BOOL freshReload = [manifest[@"fresh_descriptor_reload"] boolValue];
        NSDictionary *descriptorWeights = freshReload ? @{@"@model_path/weights/weight_data.bin": @{@"offset": @0, @"data": readData(resolvePath(root, manifest[@"weights"]))}} : @{};
        id descriptor = [[NSClassFromString(@"_ANEInMemoryModelDescriptor") alloc] initWithNetworkText:mil weights:descriptorWeights optionsPlist:plist isMILModel:YES];
        QuipAneProgram *owner = [QuipAneProgram new];
        owner.model = [NSClassFromString(@"_ANEInMemoryModel") inMemoryModelWithDescriptor:descriptor];
        require(owner.model != nil, @"model creation failed");
        uint32_t statsMask = [manifest[@"stats_mask"] unsignedIntValue];
        if (statsMask != 0) setStatsMask(owner.model, statsMask);
        NSString *directory = [NSTemporaryDirectory() stringByAppendingPathComponent:[owner.model hexStringIdentifier]];
        if (mkdir(directory.fileSystemRepresentation, 0700) != 0) {
            require(errno == EEXIST, @"staging directory creation failed");
            require([NSFileManager.defaultManager removeItemAtPath:directory error:&error], describe(@"remove leftover staging", error));
            require(mkdir(directory.fileSystemRepresentation, 0700) == 0, @"staging directory recreation failed");
        }
        owner.directory = directory;
        require([mil writeToFile:[owner.directory stringByAppendingPathComponent:@"model.mil"] options:0 error:&error], describe(@"stage MIL", error));
        if (manifest[@"weights"] != nil) {
            NSString *weights = [owner.directory stringByAppendingPathComponent:@"weights"];
            require([NSFileManager.defaultManager createDirectoryAtPath:weights withIntermediateDirectories:NO attributes:nil error:&error], describe(@"weights directory", error));
            require([NSFileManager.defaultManager copyItemAtPath:[root stringByAppendingPathComponent:manifest[@"weights"]] toPath:[weights stringByAppendingPathComponent:@"weight_data.bin"] error:&error], describe(@"stage weights", error));
        }
        printf("stage=compile mil_bytes=%zu directory=%s\n", mil.length, owner.directory.UTF8String);
        uint64_t started = monotonicUS();
        require([owner.model compileWithQoS:21 options:compilerOptions error:&error], describe(@"compile", error));
        printf("stage=compiled compile_us=%llu compile_count=1\n", (unsigned long long)(monotonicUS()-started));
        if ([manifest[@"compile_only"] boolValue]) {
            struct rusage usage;
            require(getrusage(RUSAGE_SELF, &usage) == 0, @"getrusage failed");
            printf("stage=compile_only_complete dispatch_count=0 peak_rss_bytes=%ld\n", usage.ru_maxrss);
            require([owner removeDirectory:&error], describe(@"remove staging", error));
            return 0;
        }
        if (statsMask != 0) setStatsMask(owner.model, statsMask);
        uint64_t loadStarted = monotonicUS();
        require([owner.model loadWithQoS:21 options:@{} error:&error], describe(@"load", error));
        owner.loaded = YES;
        printf("stage=loaded load_us=%llu\n", (unsigned long long)(monotonicUS()-loadStarted));
        if (manifest[@"inspect_weight_instance"] != nil) {
            printf("stage=adapter_entitlement name=%s\n", [[NSClassFromString(@"_ANEStrings") adapterWeightsAccessEntitlement] UTF8String]);
            NSMutableArray *weights = [NSMutableArray new];
            for (NSDictionary *spec in manifest[@"inspect_weight_instance"]) {
                NSURL *url = [NSURL fileURLWithPath:resolvePath(root, spec[@"file"])];
                require([NSFileManager.defaultManager fileExistsAtPath:url.path], @"instance weight file missing");
                id weight = [NSClassFromString(@"_ANEWeight") weightWithSymbolAndURL:spec[@"symbol"] weightURL:url];
                require(weight != nil, @"instance weight object failed");
                [weights addObject:weight];
            }
            id procedure = [NSClassFromString(@"_ANEProcedureData") procedureDataWithSymbol:@"main" weightArray:weights];
            id parameters = [NSClassFromString(@"_ANEModelInstanceParameters") withProcedureData:@"weight-probe" procedureArray:@[procedure]];
            id instance = [[owner.model model] shallowCopy];
            id client = [owner.model sharedConnection];
            require(instance != nil && parameters != nil && client != nil, @"instance diagnostic setup failed");
            NSError *instanceError = nil;
            uint64_t instanceStart = monotonicUS();
            BOOL loaded = [client loadModelNewInstance:instance options:@{} modelInstParams:parameters qos:21 error:&instanceError];
            printf("stage=weight_instance loaded=%d load_us=%llu parameters=%s error=%s\n", loaded, (unsigned long long)(monotonicUS()-instanceStart), [[parameters description] UTF8String], [[instanceError description] UTF8String] ?: "nil");
            if (loaded) require([client unloadModel:instance options:@{} qos:21 error:&instanceError], describe(@"unload weight instance", instanceError));
        }
        if ([manifest[@"inspect_model"] boolValue]) {
            id inner = [owner.model model];
            printf("stage=model_metadata wrapper=%s inner=%s procedure=%s\n", [[owner.model modelAttributes] description].UTF8String, [[inner modelAttributes] description].UTF8String, [[inner procedureInfoForProcedureIndex:0] description].UTF8String);
            require([NSFileManager.defaultManager copyItemAtPath:owner.directory toPath:[root stringByAppendingPathComponent:@"compiled-template"] error:&error], describe(@"copy owned compiled template", error));
        }
        if (statsMask != 0) {
            setStatsMask(owner.model, statsMask);
            if ([owner.model respondsToSelector:@selector(model)]) setStatsMask([owner.model model], statsMask);
        }
        if (manifest[@"reload_weights"] == nil) require([owner removeDirectory:&error], describe(@"remove staging", error));
        NSArray *inputSpecs = manifest[@"inputs"];
        uint64_t setupStarted = monotonicUS();
        require(inputSpecs.count < 64, @"too many probe inputs");
        NSUInteger thresholdIndex = [inputSpecs indexOfObjectPassingTest:^BOOL(NSDictionary *spec, NSUInteger index, BOOL *stop) {
            (void)index;
            (void)stop;
            return [spec[@"name"] isEqualToString:@"c_threshold"];
        }];
        if (manifest[@"threshold_input_index"] != nil) require(thresholdIndex == [manifest[@"threshold_input_index"] unsignedIntegerValue], @"threshold index does not match named input");
        IOSurfaceRef surfaces[64] = {0};
        NSMutableArray *wrappers = [NSMutableArray new], *indices = [NSMutableArray new];
        for (NSUInteger i = 0; i < inputSpecs.count; ++i) {
            size_t elements = [inputSpecs[i][@"elements"] unsignedLongLongValue];
            surfaces[i] = makeSurface(elements);
            require(surfaces[i] != NULL, @"input surface failed");
            [wrappers addObject:[NSClassFromString(@"_ANEIOSurfaceObject") objectWithIOSurface:surfaces[i]]];
            [indices addObject:@(i)];
        }
        size_t outputElements = [manifest[@"output_elements"] unsignedLongLongValue];
        IOSurfaceRef output = makeSurface(outputElements);
        require(output != NULL, @"output surface failed");
        id outputWrapper = [NSClassFromString(@"_ANEIOSurfaceObject") objectWithIOSurface:output];
        size_t iterations = manifest[@"iterations"] == nil ? 1 : [manifest[@"iterations"] unsignedLongLongValue];
        size_t evalRepeats = manifest[@"eval_repeats"] == nil ? 1 : [manifest[@"eval_repeats"] unsignedLongLongValue];
        require(iterations > 0 && iterations <= 65536, @"invalid iterator count");
        require(evalRepeats > 0 && evalRepeats <= 16, @"invalid eval repeat count");
        require(iterations == 1 || thresholdIndex != NSNotFound, @"iterator requires a threshold input");
        BOOL pingPong = ![manifest[@"fixed_io"] boolValue];
        IOSurfaceRef *thresholdSurfaces = calloc(iterations, sizeof(IOSurfaceRef));
        require(thresholdSurfaces != NULL, @"threshold surface list allocation failed");
        if (thresholdIndex != NSNotFound) thresholdSurfaces[0] = surfaces[thresholdIndex];
        NSMutableArray *requests = [NSMutableArray new];
        for (size_t block = 0; block < iterations; ++block) {
            NSMutableArray *inputs = [wrappers mutableCopy];
            BOOL swap = pingPong && (block % 2 == 1);
            inputs[0] = swap ? outputWrapper : wrappers[0];
            if (block > 0) {
                thresholdSurfaces[block] = makeSurface([inputSpecs[thresholdIndex][@"elements"] unsignedLongLongValue]);
                require(thresholdSurfaces[block] != NULL, @"threshold surface failed");
                inputs[thresholdIndex] = [NSClassFromString(@"_ANEIOSurfaceObject") objectWithIOSurface:thresholdSurfaces[block]];
            }
            id target = swap ? wrappers[0] : outputWrapper;
            id request = [NSClassFromString(@"_ANERequest") requestWithInputs:inputs inputIndices:indices outputs:@[target] outputIndices:@[@0] weightsBuffer:nil perfStats:nil procedureIndex:@0];
            require(request != nil, @"request failed");
            if (statsMask != 0) [request setPerfStats:[NSClassFromString(@"_ANEPerformanceStats") statsWithHardwareExecutionNS:0]];
            [requests addObject:request];
        }
        NSArray *expectedFiles = manifest[@"expected"];
        size_t residentBytes = IOSurfaceGetAllocSize(output);
        for (NSUInteger i = 0; i < inputSpecs.count; ++i) residentBytes += IOSurfaceGetAllocSize(surfaces[i]);
        for (size_t block = 1; block < iterations; ++block) residentBytes += IOSurfaceGetAllocSize(thresholdSurfaces[block]);
        printf("stage=setup setup_us=%llu surface_bytes=%zu\n", (unsigned long long)(monotonicUS()-setupStarted), residentBytes);
        NSMutableArray *reloadDonors = [NSMutableArray new];
        for (NSUInteger job = 0; job < expectedFiles.count; ++job) {
            if (job > 0 && manifest[@"reload_weights"] != nil) {
                uint64_t reloadStart = monotonicUS();
                NSMutableDictionary *savedFiles = [NSMutableDictionary new];
                NSArray *paths = [NSFileManager.defaultManager subpathsOfDirectoryAtPath:owner.directory error:&error];
                require(paths != nil, describe(@"list compiled template", error));
                for (NSString *relative in paths) {
                    NSString *path = [owner.directory stringByAppendingPathComponent:relative];
                    BOOL isDirectory = NO;
                    require([NSFileManager.defaultManager fileExistsAtPath:path isDirectory:&isDirectory], @"template file missing");
                    if (!isDirectory) {
                        NSData *data = readData(path);
                        savedFiles[relative] = [NSData dataWithBytes:data.bytes length:data.length];
                    }
                }
                require([owner unload:&error], describe(@"reload unload", error));
                for (NSString *relative in savedFiles) {
                    NSString *path = [owner.directory stringByAppendingPathComponent:relative];
                    require([NSFileManager.defaultManager createDirectoryAtPath:path.stringByDeletingLastPathComponent withIntermediateDirectories:YES attributes:nil error:&error], describe(@"restore template directory", error));
                    require([savedFiles[relative] writeToFile:path options:0 error:&error], describe(@"restore compiled template", error));
                }
                NSString *weightPath = [owner.directory stringByAppendingPathComponent:@"weights/weight_data.bin"];
                NSData *replacement = readData(resolvePath(root, manifest[@"reload_weights"][job]));
                require([replacement writeToFile:weightPath options:0 error:&error], describe(@"replace weights", error));
                require([readData(weightPath) isEqualToData:replacement], @"replacement bytes differ on disk");
                if (freshReload) {
                    require([replacement writeToFile:[owner.directory stringByAppendingPathComponent:@"data"] options:0 error:&error], describe(@"replace root data", error));
                    NSDictionary *weights = @{@"@model_path/weights/weight_data.bin": @{@"offset": @0, @"data": replacement}};
                    id nextDescriptor = [[NSClassFromString(@"_ANEInMemoryModelDescriptor") alloc] initWithNetworkText:mil weights:weights optionsPlist:plist isMILModel:YES];
                    id nextModel = [NSClassFromString(@"_ANEInMemoryModel") inMemoryModelWithDescriptor:nextDescriptor];
                    if (![[nextModel hexStringIdentifier] isEqualToString:[owner.model hexStringIdentifier]]) {
                        NSString *nextDirectory = [NSTemporaryDirectory() stringByAppendingPathComponent:[nextModel hexStringIdentifier]];
                        require(mkdir(nextDirectory.fileSystemRepresentation, 0700) == 0, @"new reload directory already exists");
                        for (NSString *relative in savedFiles) {
                            NSString *path = [nextDirectory stringByAppendingPathComponent:relative];
                            require([NSFileManager.defaultManager createDirectoryAtPath:path.stringByDeletingLastPathComponent withIntermediateDirectories:YES attributes:nil error:&error], describe(@"copy A template directory", error));
                            require([savedFiles[relative] writeToFile:path options:0 error:&error], describe(@"copy A compiled template", error));
                        }
                        require([replacement writeToFile:[nextDirectory stringByAppendingPathComponent:@"weights/weight_data.bin"] options:0 error:&error], describe(@"new descriptor weights", error));
                        require([replacement writeToFile:[nextDirectory stringByAppendingPathComponent:@"data"] options:0 error:&error], describe(@"new descriptor data", error));
                        require([NSFileManager.defaultManager removeItemAtPath:owner.directory error:&error], describe(@"remove old reload template", error));
                        owner.directory = nextDirectory;
                    }
                    [reloadDonors addObject:owner.model];
                    owner.model = nextModel;
                }
                require([owner.model loadWithQoS:21 options:@{} error:&error], describe(@"reload load", error));
                owner.loaded = YES;
                printf("stage=reloaded job=%zu reload_us=%llu explicit_compile_count=1\n", job, (unsigned long long)(monotonicUS()-reloadStart));
            }
            size_t hostWriteCount = 0, hostWriteBytes = 0;
            uint64_t uploadStart = monotonicUS();
            for (NSUInteger i = 0; i < inputSpecs.count; ++i) {
                NSData *data = readData(resolvePath(root, inputSpecs[i][@"files"][job]));
                size_t bytes = [inputSpecs[i][@"elements"] unsignedLongLongValue] * 2;
                size_t blocks = i == thresholdIndex ? iterations : 1;
                require(data.length == bytes * blocks, @"input byte count mismatch");
                for (size_t block = 0; block < blocks; ++block) {
                    IOSurfaceRef surface = i == thresholdIndex ? thresholdSurfaces[block] : surfaces[i];
                    require(IOSurfaceLock(surface, 0, NULL) == kIOReturnSuccess, @"input lock failed");
                    memcpy(IOSurfaceGetBaseAddress(surface), (const uint8_t *)data.bytes + block * bytes, bytes);
                    ++hostWriteCount;
                    hostWriteBytes += bytes;
                    require(IOSurfaceUnlock(surface, 0, NULL) == kIOReturnSuccess, @"input unlock failed");
                }
            }
            printf("stage=uploaded job=%zu upload_us=%llu input_count=%zu iterations=%zu eval_repeats=%zu ping_pong=%d\n", job, (unsigned long long)(monotonicUS()-uploadStart), inputSpecs.count, iterations, evalRepeats, pingPong);
            NSData *expected = readData(resolvePath(root, expectedFiles[job]));
            require(expected.length == outputElements * 2, @"expected byte count mismatch");
            require(![manifest[@"validate_first"] boolValue] || iterations == 1, @"first validation requires one block");
            printf("stage=dispatch_boundary job=%zu boundary=before host_write_count=%zu host_write_bytes=%zu request_count=%zu\n", job, hostWriteCount, hostWriteBytes, iterations);
            started = monotonicUS();
            uint64_t *blockTimes = calloc(iterations, sizeof(uint64_t));
            require(blockTimes != NULL, @"block timing allocation failed");
            for (size_t block = 0; block < iterations; ++block) {
                for (size_t repeat = 0; repeat < evalRepeats; ++repeat) {
                    struct rusage cpuBefore = {0}, cpuAfter = {0};
                    require(getrusage(RUSAGE_SELF, &cpuBefore) == 0, @"CPU stats failed");
                    uint64_t blockStart = monotonicUS();
                    require([owner.model evaluateWithQoS:21 options:@{} request:requests[block] error:&error], describe(@"evaluate", error));
                    uint64_t evaluateUS = monotonicUS()-blockStart;
                    require(getrusage(RUSAGE_SELF, &cpuAfter) == 0, @"CPU stats failed");
                    blockTimes[block] = evaluateUS;
                    printf("stage=evaluate job=%zu block=%zu repeat=%zu evaluate_us=%llu process_cpu_us=%llu\n", job, block, repeat, (unsigned long long)evaluateUS, (unsigned long long)(processCPUUS(cpuAfter)-processCPUUS(cpuBefore)));
                    if (repeat == 0 && [manifest[@"validate_first"] boolValue]) {
                        uint64_t validationStarted = monotonicUS();
                        require(IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL) == kIOReturnSuccess, @"first output lock failed");
                        const _Float16 *first = IOSurfaceGetBaseAddress(output), *wanted = expected.bytes;
                        size_t firstMismatches = 0;
                        for (size_t i = 0; i < outputElements; ++i) firstMismatches += first[i] != wanted[i];
                        require(IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL) == kIOReturnSuccess, @"first output unlock failed");
                        printf("stage=first_validation job=%zu validation_us=%llu mismatches=%zu\n", job, (unsigned long long)(monotonicUS()-validationStarted), firstMismatches);
                        require(firstMismatches == 0, @"first output oracle mismatch");
                    }
                    if ([manifest[@"diagnostics"] boolValue]) {
                        id stats = [requests[block] perfStats];
                        printf("stage=device_stats job=%zu block=%zu repeat=%zu stats=%s stats_array=%s\n", job, block, repeat, [[stats description] UTF8String] ?: "nil", [[[requests[block] perfStatsArray] description] UTF8String] ?: "nil");
                        if ([stats respondsToSelector:@selector(hwExecutionTime)]) printf("stage=hardware_time value=%llu counters=%s\n", (unsigned long long)[stats hwExecutionTime], [[[stats performanceCounters] description] UTF8String] ?: "nil");
                    }
                }
                if (iterations > 1 && (block + 1) % 32 == 0) printf("stage=progress job=%zu completed_blocks=%zu\n", job, block + 1);
            }
            uint64_t elapsed = monotonicUS()-started;
            printf("stage=dispatch_boundary job=%zu boundary=after host_write_count=%zu host_write_bytes=%zu request_count=%zu\n", job, hostWriteCount, hostWriteBytes, iterations);
            for (size_t block = 0; block < iterations; ++block) printf("stage=block job=%zu block=%zu evaluate_us=%llu\n", job, block, (unsigned long long)blockTimes[block]);
            free(blockTimes);
            uint64_t readbackStart = monotonicUS();
            IOSurfaceRef finalState = (pingPong && iterations % 2 == 0) ? surfaces[0] : output;
            require(IOSurfaceLock(finalState, kIOSurfaceLockReadOnly, NULL) == kIOReturnSuccess, @"output lock failed");
            NSData *readback = [NSData dataWithBytes:IOSurfaceGetBaseAddress(finalState) length:outputElements*2];
            require(IOSurfaceUnlock(finalState, kIOSurfaceLockReadOnly, NULL) == kIOReturnSuccess, @"output unlock failed");
            printf("stage=readback job=%zu readback_us=%llu includes=lock,copy,unlock\n", job, (unsigned long long)(monotonicUS()-readbackStart));
            uint64_t validationStarted = monotonicUS();
            const _Float16 *actual = readback.bytes, *wanted = expected.bytes;
            size_t mismatches = 0;
            for (size_t i = 0; i < outputElements; ++i) {
                if (actual[i] != wanted[i]) {
                    if (mismatches < 5) printf("mismatch index=%zu actual=%g expected=%g\n", i, (double)actual[i], (double)wanted[i]);
                    ++mismatches;
                }
            }
            NSString *actualPath = [root stringByAppendingPathComponent:[NSString stringWithFormat:@"actual%zu.bin", job]];
            printf("stage=validation job=%zu validation_us=%llu mismatches=%zu\n", job, (unsigned long long)(monotonicUS()-validationStarted), mismatches);
            require([readback writeToFile:actualPath options:0 error:&error], describe(@"write actual", error));
            struct rusage usage;
            require(getrusage(RUSAGE_SELF, &usage) == 0, @"getrusage failed");
            printf("stage=completed job=%zu dispatch_count=%zu job_dispatch_count=%zu evaluate_us=%llu mismatches=%zu elements=%zu peak_rss_bytes=%ld\n", job, (job+1)*iterations*evalRepeats, iterations*evalRepeats, (unsigned long long)elapsed, mismatches, outputElements, usage.ru_maxrss);
            if (mismatches != 0 && manifest[@"reload_weights"] != nil) {
                require([owner unload:&error], describe(@"failed reload cleanup", error));
                if (![NSFileManager.defaultManager fileExistsAtPath:owner.directory]) owner.directory = nil;
            }
            require(mismatches == 0, @"oracle mismatch");
        }
        require([owner unload:&error], describe(@"unload", error));
        if (owner.directory != nil && ![NSFileManager.defaultManager fileExistsAtPath:owner.directory]) owner.directory = nil;
        CFRelease(output);
        for (size_t block = 1; block < iterations; ++block) CFRelease(thresholdSurfaces[block]);
        free(thresholdSurfaces);
        for (NSUInteger i = 0; i < inputSpecs.count; ++i) CFRelease(surfaces[i]);
        return 0;
    } @catch (NSException *exception) {
        fprintf(stderr, "probe_failed=%s\n", exception.reason.UTF8String);
        return 1;
    } }
}
