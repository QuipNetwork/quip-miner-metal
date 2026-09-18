// Standalone probe that times each stage of `quip_ane_create` on the real
// M4 Max fixture topology. Reuses the bridge source verbatim (the `#include`
// below is the same pattern `single_call.m` already uses in this directory)
// so the probe measures the code that ships, not a re-typed copy of it.
// Production bridge entry points are unchanged.
#include "../native/ane_bridge.m"
#include <mach/mach_time.h>
#include <stdlib.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static uint64_t now(void) { return mach_absolute_time(); }

// Eight colors from docs/perf/2026-09-16-ane-local-routing.md:13, descending
// degree then ascending index over tests/fixtures/advantage2-system1.edges,
// exactly as graph.rs colors it. No color exceeds the 4,096-output tile limit.
static const size_t kLengths[8] = {857, 849, 817, 740, 685, 480, 136, 13};
static const size_t kTiles = 8;
// Next multiple of 32 at or above 4,577.
static const size_t kChannels = 4608;
// BLOCK_SWEEPS in crates/ane-miner/src/native.rs: every production compile
// call, `--solve` included, builds the program with exactly this many sweeps.
static const size_t kSweeps = 2;

int main(void) {
    setbuf(stdout, NULL);
    mach_timebase_info(&g_tb);

    size_t sum = 0;
    for (size_t i = 0; i < kTiles; ++i) sum += kLengths[i];
    if (sum != 4577) { fprintf(stderr, "tile length sum is %zu, expected 4577\n", sum); return 2; }
    if (kChannels != 4608) { fprintf(stderr, "channel count is %zu, expected 4608\n", kChannels); return 2; }
    if (kSweeps + 2 != 4) { fprintf(stderr, "kSweeps changed; update the surfaces[4] array below\n"); return 2; }

    @autoreleasepool {
    @try {
        NSError *error = nil;

        // graph_prep_ms: stands in for the Rust-side weights/fields vector
        // build in native.rs AneProgram::compile. No Rust runs in this probe,
        // so this fills the same-shaped buffers with seeded synthetic data
        // instead of walking real neighbor lists: couplings in {-1, 1} from
        // seed 123, zero fields, matching the brief exactly.
        uint64_t t0 = now();
        size_t weightCount = 0;
        for (size_t i = 0; i < kTiles; ++i) weightCount += ((kLengths[i] + 31) / 32 * 32) * kChannels;
        int8_t *weights = malloc(weightCount);
        int8_t *fields = calloc(kChannels, 1);
        if (weights == NULL || fields == NULL) { fprintf(stderr, "weight or field allocation failed\n"); return 2; }
        srandom(123);
        for (size_t i = 0; i < weightCount; ++i) weights[i] = (random() & 1) ? (int8_t)1 : (int8_t)-1;
        double graph_prep_ms = tb_ms(now() - t0);

        // The range-validation loop quip_ane_create runs over the same
        // buffers before touching the ANE runtime (ane_bridge.m:275-280).
        // Not one of the required stage keys; kept as its own bucket so the
        // gap analysis in the report can name it instead of guessing.
        t0 = now();
        for (size_t i = 0; i < weightCount; ++i) if (weights[i] < -1 || weights[i] > 1) { fprintf(stderr, "weight out of range\n"); return 2; }
        for (size_t i = 0; i < kChannels; ++i) if (fields[i] < -1 || fields[i] > 1) { fprintf(stderr, "field out of range\n"); return 2; }
        double range_validate_ms = tb_ms(now() - t0);

        // mil_build_ms: brief step 2, bracketing only the makeMIL call.
        t0 = now();
        NSData *mil = [makeMIL(kChannels, kLengths, kTiles, kSweeps, fields) dataUsingEncoding:NSUTF8StringEncoding];
        double mil_build_ms = tb_ms(now() - t0);

        // blob_build_ms: brief step 2, bracketing only makeWeightBlob.
        t0 = now();
        NSData *blob = makeWeightBlob(weights, kChannels, kLengths, kTiles, weightCount);
        double blob_build_ms = tb_ms(now() - t0);

        // Framework dlopen, private class lookups, plist/descriptor/model
        // setup, and staging directory creation: real quip_ane_create work,
        // also not one of the required keys. One bucket, same reasoning as
        // range_validate_ms above.
        t0 = now();
        if (dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW) == NULL) {
            fprintf(stderr, "ANE framework dlopen failed\n");
            return 2;
        }
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        requireSelector(descriptorClass, @selector(alloc));
        requireSelector(modelClass, @selector(inMemoryModelWithDescriptor:));
        requireSelector(surfaceClass, @selector(objectWithIOSurface:));
        NSData *plist = [NSPropertyListSerialization dataWithPropertyList:@{} format:NSPropertyListXMLFormat_v1_0 options:0 error:&error];
        if (plist == nil) { fprintf(stderr, "plist serialization failed: %s\n", error.description.UTF8String); return 2; }
        id descriptor = [[descriptorClass alloc] initWithNetworkText:mil weights:@{} optionsPlist:plist isMILModel:YES];
        if (descriptor == nil) { fprintf(stderr, "descriptor creation failed\n"); return 2; }
        QuipAneProgram *owner = [QuipAneProgram new];
        owner.model = [modelClass inMemoryModelWithDescriptor:descriptor];
        if (owner.model == nil) { fprintf(stderr, "model creation failed\n"); return 2; }
        requireSelector(owner.model, @selector(hexStringIdentifier));
        NSString *identifier = [owner.model hexStringIdentifier];
        NSString *directory = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        if (mkdir(directory.fileSystemRepresentation, 0700) != 0) {
            fprintf(stderr, "staging directory creation failed: %s\n", strerror(errno));
            return 2;
        }
        owner.directory = directory;
        NSString *weightDirectory = [directory stringByAppendingPathComponent:@"weights"];
        if (![NSFileManager.defaultManager createDirectoryAtPath:weightDirectory withIntermediateDirectories:NO attributes:nil error:&error]) {
            fprintf(stderr, "weights directory creation failed: %s\n", error.description.UTF8String);
            return 2;
        }
        if (![mil writeToFile:[directory stringByAppendingPathComponent:@"model.mil"] options:NSDataWritingAtomic error:&error]) {
            fprintf(stderr, "MIL file write failed: %s\n", error.description.UTF8String);
            return 2;
        }
        double dlopen_class_setup_ms = tb_ms(now() - t0);

        // blob_write_ms: brief step 2, bracketing only the blob file write.
        NSString *weightPath = [weightDirectory stringByAppendingPathComponent:@"weight_data.bin"];
        t0 = now();
        BOOL wroteBlob = [blob writeToFile:weightPath options:NSDataWritingAtomic error:&error];
        double blob_write_ms = tb_ms(now() - t0);
        if (!wroteBlob) { fprintf(stderr, "weight blob write failed: %s\n", error.description.UTF8String); return 2; }

        // compile_ms
        requireSelector(owner.model, @selector(compileWithQoS:options:error:));
        t0 = now();
        BOOL compiled = [owner.model compileWithQoS:21 options:@{} error:&error];
        double compile_ms = tb_ms(now() - t0);
        if (!compiled) { fprintf(stderr, "compile failed: %s\n", error.description.UTF8String); return 2; }

        // load_ms
        requireSelector(owner.model, @selector(loadWithQoS:options:error:));
        requireSelector(owner.model, @selector(unloadWithQoS:error:));
        t0 = now();
        BOOL loaded = [owner.model loadWithQoS:21 options:@{} error:&error];
        double load_ms = tb_ms(now() - t0);
        if (!loaded) { fprintf(stderr, "load failed: %s\n", error.description.UTF8String); return 2; }
        owner.loaded = YES;

        if (![owner removeDirectory:&error]) { fprintf(stderr, "staging directory removal failed: %s\n", error.description.UTF8String); return 2; }

        // surface_setup_ms: the sweeps+2 IOSurfaces quip_ane_create allocates
        // and wraps, same count as the real create path (kSweeps == 2).
        size_t inputElements;
        if (!multiply(kChannels, 128, &inputElements)) { fprintf(stderr, "input element count overflow\n"); return 2; }
        IOSurfaceRef surfaces[4] = {0}; // kSweeps + 2
        NSMutableArray *wrappers = [NSMutableArray new];
        t0 = now();
        for (size_t i = 0; i < kSweeps + 2; ++i) {
            surfaces[i] = makeSurface(inputElements);
            if (surfaces[i] == NULL) { fprintf(stderr, "IOSurface allocation failed\n"); return 2; }
            id wrapper = [surfaceClass objectWithIOSurface:surfaces[i]];
            if (wrapper == nil) { fprintf(stderr, "IOSurface wrapping failed\n"); return 2; }
            [wrappers addObject:wrapper];
        }
        double surface_setup_ms = tb_ms(now() - t0);

        if (![owner unload:&error]) fprintf(stderr, "warning: cleanup unload failed: %s\n", error.description.UTF8String);
        for (size_t i = 0; i < kSweeps + 2; ++i) if (surfaces[i] != NULL) CFRelease(surfaces[i]);
        free(weights);
        free(fields);

        double requiredSum = graph_prep_ms + mil_build_ms + blob_build_ms + blob_write_ms + compile_ms + load_ms + surface_setup_ms;
        double probeInternalTotal = requiredSum + range_validate_ms + dlopen_class_setup_ms;

        printf("{"
            "\"graph_prep_ms\":%.3f,"
            "\"mil_build_ms\":%.3f,"
            "\"blob_build_ms\":%.3f,"
            "\"blob_write_ms\":%.3f,"
            "\"compile_ms\":%.3f,"
            "\"load_ms\":%.3f,"
            "\"surface_setup_ms\":%.3f,"
            "\"blob_bytes\":%llu,"
            "\"mil_bytes\":%llu,"
            "\"range_validate_ms\":%.3f,"
            "\"dlopen_class_setup_ms\":%.3f,"
            "\"required_stage_sum_ms\":%.3f,"
            "\"probe_internal_total_ms\":%.3f"
            "}\n",
            graph_prep_ms, mil_build_ms, blob_build_ms, blob_write_ms, compile_ms, load_ms, surface_setup_ms,
            (unsigned long long)blob.length, (unsigned long long)mil.length,
            range_validate_ms, dlopen_class_setup_ms, requiredSum, probeInternalTotal);
        return 0;
    } @catch (NSException *exception) {
        fprintf(stderr, "probe_failed=%s\n", exception.reason.UTF8String);
        return 1;
    }
    }
}
