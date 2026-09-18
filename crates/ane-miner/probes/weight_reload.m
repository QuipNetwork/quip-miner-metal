// Task 4 probe: does overwriting the weight blob on disk and calling
// unloadWithQoS: then loadWithQoS: change a compiled program's couplings,
// without a new compileWithQoS: pass?
//
// This mirrors the create path in quip_ane_create, ane_bridge.m:251, using
// the same reuse pattern setup_profile.m already establishes in this
// directory (`#include "../native/ane_bridge.m"`, calling its static
// makeMIL, makeWeightBlob, and QuipAneProgram helpers unchanged). The one
// deliberate difference from that create path: this probe skips the
// removeDirectory call at ane_bridge.m:339, so weights/weight_data.bin is
// still on disk to overwrite after the first evaluation. Everything else,
// including quip_ane_reset, quip_ane_evaluate, and quip_ane_read, calls the
// real exported bridge functions unchanged. Production bridge entry points
// are unchanged.
#include "../native/ane_bridge.m"
#include <mach/mach_time.h>
#include <stdlib.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static uint64_t now(void) { return mach_absolute_time(); }

// Small shape per the task brief: one tile covering every channel, so the
// weight matrix is square and every row's conv sum reduces to a plain
// matrix-vector product with no padding rows to trim.
static const size_t kChannels = 512;
static const size_t kLengths[1] = {512};
static const size_t kTiles = 1;
static const size_t kSweeps = 2;
// compile_ms median this probe's positive result would replace, from
// .superpowers/sdd/2026-09-17-ane-throughput/task-1-report.md.
static const double kCompileMsBaseline = 283.887;

// Ring topology: node i couples to node i+1 (mod channels) with weight +1.
// Row-major [output_channel, input_channel] matches the conv weight layout
// makeMIL emits (ane_bridge.m:160): raw[row] = sum_c weight[row,c] * x[c].
static void buildRingWeights(int8_t *weights, size_t channels) {
    memset(weights, 0, channels * channels);
    for (size_t row = 0; row < channels; ++row) weights[row * channels + (row + 1) % channels] = 1;
}

// Host emulation of one MIL sweep tile: conv sum against the same weight
// matrix the ANE program uses, then the same threshold==0 step function
// makeMIL builds at ane_bridge.m:183-193 (margin = threshold + own*(raw+h);
// flip when margin >= 0). h is zero throughout this probe, so it is omitted
// here. This is plain integer arithmetic, matching the task brief's "do not
// accept 'the output changed' as a pass" requirement.
static void hostStep(const int8_t *weights, size_t channels, const int8_t *x, int8_t *out) {
    for (size_t row = 0; row < channels; ++row) {
        int raw = 0;
        for (size_t col = 0; col < channels; ++col) raw += (int)weights[row * channels + col] * (int)x[col];
        int own = x[row];
        int margin = own * raw;
        out[row] = (int8_t)(margin >= 0 ? -own : own);
    }
}

// Compares a lane-replicated device readout (inputElements = channels * 128
// entries) against a per-node host prediction (channels entries). Every
// lane must agree, since every lane in this probe evaluates the same
// constant threshold and should therefore land on the same flip decision.
static BOOL matchesReplicated(const int8_t *actual, size_t channels, const int8_t *expectedPerNode) {
    for (size_t c = 0; c < channels; ++c)
        for (size_t l = 0; l < 128; ++l)
            if (actual[c * 128 + l] != expectedPerNode[c]) return NO;
    return YES;
}

// Simple 64-bit FNV-1a checksum, used only to prove the two weight blobs
// this probe writes to disk are not the same bytes, and that each write
// landed intact. Not a security hash; collisions are not a concern for
// artifacts this small and this distinct.
static uint64_t fnv1a64(const void *data, size_t length) {
    const uint8_t *bytes = data;
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < length; ++i) {
        hash ^= bytes[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

// Deterministic pseudo-random +/-1 pattern, not a fixed alternation, so the
// host check exercises real per-node flip/no-flip variation instead of one
// uniform outcome across every node.
static void buildSpins(int8_t *x, size_t channels) {
    uint32_t state = 0x2545F491u;
    for (size_t i = 0; i < channels; ++i) {
        state = state * 1664525u + 1013904223u;
        x[i] = (state & 0x80000000u) ? (int8_t)1 : (int8_t)-1;
    }
}

int main(void) {
    setbuf(stdout, NULL);
    mach_timebase_info(&g_tb);

    @autoreleasepool {
    @try {
        NSError *error = nil;
        size_t weightCount = kChannels * kChannels;

        int8_t *weightsA = malloc(weightCount);
        int8_t *fields = calloc(kChannels, 1); // zero fields: this probe tests the coupling swap alone
        int8_t *x = malloc(kChannels);
        if (weightsA == NULL || fields == NULL || x == NULL) { fprintf(stderr, "allocation failed\n"); return 2; }
        buildRingWeights(weightsA, kChannels);
        buildSpins(x, kChannels);

        // === Step 1: compile a program and keep the staging directory ===
        NSData *mil = [makeMIL(kChannels, kLengths, kTiles, kSweeps, fields) dataUsingEncoding:NSUTF8StringEncoding];
        NSData *blobA = makeWeightBlob(weightsA, kChannels, kLengths, kTiles, weightCount);

        if (dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW) == NULL) {
            fprintf(stderr, "ANE framework dlopen failed\n");
            return 2;
        }
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        Class requestClass = NSClassFromString(@"_ANERequest");
        if (![descriptorClass instancesRespondToSelector:@selector(initWithNetworkText:weights:optionsPlist:isMILModel:)]) {
            fprintf(stderr, "ANE descriptor initializer unavailable\n");
            return 2;
        }
        requireSelector(modelClass, @selector(inMemoryModelWithDescriptor:));
        requireSelector(surfaceClass, @selector(objectWithIOSurface:));
        requireSelector(requestClass, @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:));

        NSData *plist = [NSPropertyListSerialization dataWithPropertyList:@{} format:NSPropertyListXMLFormat_v1_0 options:0 error:&error];
        if (plist == nil) { fprintf(stderr, "plist serialization failed: %s\n", error.description.UTF8String); return 2; }
        id descriptor = [[descriptorClass alloc] initWithNetworkText:mil weights:@{} optionsPlist:plist isMILModel:YES];
        if (descriptor == nil) { fprintf(stderr, "descriptor creation failed\n"); return 2; }

        QuipAneProgram *owner = [QuipAneProgram new];
        owner->inputElements = kChannels * 128;
        owner->sweeps = kSweeps;
        owner.model = [modelClass inMemoryModelWithDescriptor:descriptor];
        if (owner.model == nil) { fprintf(stderr, "model creation failed\n"); return 2; }
        requireSelector(owner.model, @selector(hexStringIdentifier));
        NSString *identifier = [owner.model hexStringIdentifier];
        NSString *directory = [NSTemporaryDirectory() stringByAppendingPathComponent:identifier];
        if (mkdir(directory.fileSystemRepresentation, 0700) != 0) {
            fprintf(stderr, "staging directory creation failed: %s\n", strerror(errno));
            return 2;
        }
        owner.directory = directory; // kept: this probe does not call removeDirectory until the end
        NSString *weightDirectory = [directory stringByAppendingPathComponent:@"weights"];
        if (![NSFileManager.defaultManager createDirectoryAtPath:weightDirectory withIntermediateDirectories:NO attributes:nil error:&error]) {
            fprintf(stderr, "weights directory creation failed: %s\n", error.description.UTF8String);
            return 2;
        }
        if (![mil writeToFile:[directory stringByAppendingPathComponent:@"model.mil"] options:NSDataWritingAtomic error:&error]) {
            fprintf(stderr, "MIL file write failed: %s\n", error.description.UTF8String);
            return 2;
        }
        NSString *weightPath = [weightDirectory stringByAppendingPathComponent:@"weight_data.bin"];
        if (![blobA writeToFile:weightPath options:NSDataWritingAtomic error:&error]) {
            fprintf(stderr, "weight blob write failed: %s\n", error.description.UTF8String);
            return 2;
        }
        // Prove the write landed where and as intended, not just that
        // writeToFile: returned YES. Read the file back and hash it rather
        // than trusting the in-memory NSData this probe already holds.
        uint64_t hashA = fnv1a64(blobA.bytes, blobA.length);
        NSData *onDiskA = [NSData dataWithContentsOfFile:weightPath];
        if (onDiskA == nil || fnv1a64(onDiskA.bytes, onDiskA.length) != hashA) {
            fprintf(stderr, "weight blob A readback does not match what this probe wrote\n");
            return 2;
        }
        printf("blobA bytes=%llu fnv1a64=%016llx (readback confirmed)\n",
            (unsigned long long)blobA.length, (unsigned long long)hashA);

        requireSelector(owner.model, @selector(compileWithQoS:options:error:));
        uint64_t t0 = now();
        BOOL compiled = [owner.model compileWithQoS:21 options:@{} error:&error];
        double compile_ms = tb_ms(now() - t0);
        if (!compiled) { fprintf(stderr, "compile failed: %s\n", error.description.UTF8String); return 2; }

        requireSelector(owner.model, @selector(loadWithQoS:options:error:));
        requireSelector(owner.model, @selector(unloadWithQoS:error:));
        requireSelector(owner.model, @selector(evaluateWithQoS:options:request:error:));
        BOOL loaded = [owner.model loadWithQoS:21 options:@{} error:&error];
        if (!loaded) { fprintf(stderr, "load failed: %s\n", error.description.UTF8String); return 2; }
        owner.loaded = YES;
        printf("compile_ms=%.3f (staging directory kept, unlike production)\n", compile_ms);

        size_t inputElements = kChannels * 128;
        IOSurfaceRef surfaces[4] = {0}; // kSweeps + 2
        NSMutableArray *wrappers = [NSMutableArray new];
        for (size_t i = 0; i < kSweeps + 2; ++i) {
            surfaces[i] = makeSurface(inputElements);
            owner->surfaces[i] = surfaces[i];
            if (surfaces[i] == NULL) { fprintf(stderr, "IOSurface allocation failed\n"); return 2; }
            id wrapper = [surfaceClass objectWithIOSurface:surfaces[i]];
            if (wrapper == nil) { fprintf(stderr, "IOSurface wrapping failed\n"); return 2; }
            [wrappers addObject:wrapper];
        }
        owner.wrappers = wrappers;
        NSMutableArray *requests = [NSMutableArray new];
        for (size_t state = 0; state < 2; ++state) {
            NSMutableArray *inputs = [NSMutableArray arrayWithObject:wrappers[state]];
            NSMutableArray *indices = [NSMutableArray arrayWithObject:@0];
            for (size_t sweep = 0; sweep < kSweeps; ++sweep) {
                [inputs addObject:wrappers[sweep + 2]];
                [indices addObject:@(sweep + 1)];
            }
            id request = [requestClass requestWithInputs:inputs inputIndices:indices
                outputs:@[wrappers[1 - state]] outputIndices:@[@0]
                weightsBuffer:nil perfStats:nil procedureIndex:@0];
            if (request == nil) { fprintf(stderr, "request creation failed\n"); return 2; }
            [requests addObject:request];
        }
        owner.requests = requests;

        void *program = (__bridge_retained void *)owner;
        char nativeError[1024];

        // Threshold layout matches ane_bridge.h: sweep-major, then
        // channel-major, 128 lanes per channel. Sweep 0 uses threshold 0 on
        // every lane, so the flip decision depends only on margin = own *
        // raw. Sweep 1 uses 255 (skip), the same "disable every flip"
        // convention the production tail sweep uses (README.md:188), so
        // the sweep-0 result is the final result.
        int8_t *spinsFlat = malloc(inputElements);
        uint8_t *thresholds = malloc(inputElements * kSweeps);
        if (spinsFlat == NULL || thresholds == NULL) { fprintf(stderr, "spin or threshold allocation failed\n"); return 2; }
        for (size_t c = 0; c < kChannels; ++c) for (size_t l = 0; l < 128; ++l) spinsFlat[c * 128 + l] = x[c];
        for (size_t i = 0; i < inputElements; ++i) thresholds[i] = 0;
        for (size_t i = inputElements; i < inputElements * kSweeps; ++i) thresholds[i] = 255;

        // === Step 2: evaluate and record the output ===
        if (quip_ane_reset(program, spinsFlat, inputElements, nativeError, sizeof(nativeError)) != 0) {
            fprintf(stderr, "reset failed: %s\n", nativeError);
            return 2;
        }
        QuipAneTimes times;
        if (quip_ane_evaluate(program, thresholds, inputElements * kSweeps, &times, nativeError, sizeof(nativeError)) != 0) {
            fprintf(stderr, "evaluate failed: %s\n", nativeError);
            return 2;
        }
        // quip_ane_read returns the full lane-replicated surface, count ==
        // inputElements (ane_bridge.m:438-463), not one value per node.
        // Every one of the 128 lanes per channel carries the same value
        // here, since every lane uses the same constant threshold.
        int8_t *outA = malloc(inputElements);
        if (outA == NULL) { fprintf(stderr, "outA allocation failed\n"); return 2; }
        if (quip_ane_read(program, outA, inputElements, nativeError, sizeof(nativeError)) != 0) {
            fprintf(stderr, "read A failed: %s\n", nativeError);
            return 2;
        }
        printf("outA node[0..3]=[%d,%d,%d,%d]\n", outA[0 * 128], outA[1 * 128], outA[2 * 128], outA[3 * 128]);

        // === Step 3: overwrite the blob and reload ===
        int8_t *weightsB = malloc(weightCount);
        if (weightsB == NULL) { fprintf(stderr, "weightsB allocation failed\n"); return 2; }
        for (size_t i = 0; i < weightCount; ++i) weightsB[i] = (int8_t)(-weightsA[i]);
        NSData *blobB = makeWeightBlob(weightsB, kChannels, kLengths, kTiles, weightCount);
        // The no-change result below only means something if blob B is
        // provably not the same bytes as blob A. Check that before writing,
        // not by inspecting makeWeightBlob's logic.
        uint64_t hashB = fnv1a64(blobB.bytes, blobB.length);
        if (hashB == hashA) {
            fprintf(stderr, "weight blob B has the same checksum as blob A; negating the couplings produced identical bytes\n");
            return 2;
        }
        if (![blobB writeToFile:weightPath options:NSDataWritingAtomic error:&error]) {
            fprintf(stderr, "weight blob overwrite failed: %s\n", error.description.UTF8String);
            return 2;
        }
        NSData *onDiskB = [NSData dataWithContentsOfFile:weightPath];
        if (onDiskB == nil || fnv1a64(onDiskB.bytes, onDiskB.length) != hashB) {
            fprintf(stderr, "weight blob B readback does not match what this probe wrote\n");
            return 2;
        }
        printf("blobB bytes=%llu fnv1a64=%016llx (readback confirmed, differs from blobA's %016llx)\n",
            (unsigned long long)blobB.length, (unsigned long long)hashB, (unsigned long long)hashA);

        t0 = now();
        BOOL unloadOK = [owner.model unloadWithQoS:21 error:&error];
        double unload_ms = tb_ms(now() - t0);
        printf("unload: ok=%d unload_ms=%.3f\n", unloadOK, unload_ms);
        if (!unloadOK) { fprintf(stderr, "unload failed: %s\n", error.description.UTF8String); return 2; }
        owner.loaded = NO;

        t0 = now();
        BOOL reloadOK = [owner.model loadWithQoS:21 options:@{} error:&error];
        double reload_ms = tb_ms(now() - t0);
        printf("reload: ok=%d reload_ms=%.3f\n", reloadOK, reload_ms);
        if (!reloadOK) {
            fprintf(stderr, "reload after overwrite failed: %s\n", error.description.UTF8String);
            printf("outcome=RELOAD_FAILED\n");
            return 1;
        }
        owner.loaded = YES;

        // === Step 4: evaluate again with the same input state and compare ===
        if (quip_ane_reset(program, spinsFlat, inputElements, nativeError, sizeof(nativeError)) != 0) {
            fprintf(stderr, "reset before second evaluate failed: %s\n", nativeError);
            return 2;
        }
        if (quip_ane_evaluate(program, thresholds, inputElements * kSweeps, &times, nativeError, sizeof(nativeError)) != 0) {
            fprintf(stderr, "evaluate after reload failed: %s\n", nativeError);
            return 2;
        }
        int8_t *outB = malloc(inputElements);
        if (outB == NULL) { fprintf(stderr, "outB allocation failed\n"); return 2; }
        if (quip_ane_read(program, outB, inputElements, nativeError, sizeof(nativeError)) != 0) {
            fprintf(stderr, "read B failed: %s\n", nativeError);
            return 2;
        }
        printf("outB node[0..3]=[%d,%d,%d,%d]\n", outB[0 * 128], outB[1 * 128], outB[2 * 128], outB[3 * 128]);

        int8_t *expectedA = malloc(kChannels);
        int8_t *expectedB = malloc(kChannels);
        if (expectedA == NULL || expectedB == NULL) { fprintf(stderr, "expected-output allocation failed\n"); return 2; }
        hostStep(weightsA, kChannels, x, expectedA);
        hostStep(weightsB, kChannels, x, expectedB);
        BOOL outAMatchesExpectedA = matchesReplicated(outA, kChannels, expectedA);
        if (!outAMatchesExpectedA) {
            fprintf(stderr, "warning: outA does not match the host prediction for the unswapped weights; the sweep model assumed by this probe may not match ane_bridge.m\n");
        }

        BOOL changedFromA = memcmp(outA, outB, inputElements) != 0;
        BOOL matchesExpectedB = matchesReplicated(outB, kChannels, expectedB);

        // === Step 5 in the harness's own terms: report the outcome ===
        if (!changedFromA) {
            printf("outcome=3 reason=output_unchanged\n");
            printf("The runtime cached the couplings at compile time. The swap does not work.\n");
        } else if (matchesExpectedB) {
            printf("outcome=1 reason=matches_host_prediction\n");
            printf("The swap works. speedup=%.2fx (compile_ms_baseline=%.3f vs unload_plus_reload_ms=%.3f)\n",
                kCompileMsBaseline / (unload_ms + reload_ms), kCompileMsBaseline, unload_ms + reload_ms);
        } else {
            printf("outcome=2 reason=changed_but_wrong\n");
            size_t mismatches = 0;
            for (size_t c = 0; c < kChannels; ++c) if (outB[c * 128] != expectedB[c]) ++mismatches;
            printf("mismatched_nodes=%zu of %zu\n", mismatches, kChannels);
            printf("The swap is unsound.\n");
        }

        printf("{\"unload_ms\":%.3f,\"reload_ms\":%.3f,\"compile_ms\":%.3f,\"compile_ms_baseline\":%.3f}\n",
            unload_ms, reload_ms, compile_ms, kCompileMsBaseline);

        // === Positive control: a fresh compile with the negated weights,
        // built independently of the reload path above through the real,
        // unmodified quip_ane_create export (not this probe's step-1
        // scaffold). A pass here proves three things at once: expectedB is
        // correct and achievable on this device, the device honours
        // negated couplings at all, and this probe can detect a coupling
        // change when the runtime really makes one. It does not depend on
        // anything the reload path above measured.
        void *controlProgram = NULL;
        char controlError[1024];
        if (quip_ane_create(kChannels, kLengths, kTiles, kSweeps, weightsB, weightCount,
                fields, kChannels, &controlProgram, controlError, sizeof(controlError)) != 0) {
            fprintf(stderr, "positive control: fresh compile with negated weights failed: %s\n", controlError);
            printf("positive_control=FAILED reason=compile\n");
            return 2;
        }
        if (quip_ane_reset(controlProgram, spinsFlat, inputElements, controlError, sizeof(controlError)) != 0) {
            fprintf(stderr, "positive control: reset failed: %s\n", controlError);
            printf("positive_control=FAILED reason=reset\n");
            quip_ane_destroy(controlProgram, controlError, sizeof(controlError));
            return 2;
        }
        QuipAneTimes controlTimes;
        if (quip_ane_evaluate(controlProgram, thresholds, inputElements * kSweeps, &controlTimes,
                controlError, sizeof(controlError)) != 0) {
            fprintf(stderr, "positive control: evaluate failed: %s\n", controlError);
            printf("positive_control=FAILED reason=evaluate\n");
            quip_ane_destroy(controlProgram, controlError, sizeof(controlError));
            return 2;
        }
        int8_t *outControl = malloc(inputElements);
        if (outControl == NULL) { fprintf(stderr, "outControl allocation failed\n"); return 2; }
        if (quip_ane_read(controlProgram, outControl, inputElements, controlError, sizeof(controlError)) != 0) {
            fprintf(stderr, "positive control: read failed: %s\n", controlError);
            printf("positive_control=FAILED reason=read\n");
            free(outControl);
            quip_ane_destroy(controlProgram, controlError, sizeof(controlError));
            return 2;
        }
        BOOL controlMatchesExpectedB = matchesReplicated(outControl, kChannels, expectedB);
        printf("positive_control node[0..3]=[%d,%d,%d,%d] matches_expectedB=%d\n",
            outControl[0 * 128], outControl[1 * 128], outControl[2 * 128], outControl[3 * 128], controlMatchesExpectedB);
        if (quip_ane_destroy(controlProgram, controlError, sizeof(controlError)) != 0) {
            fprintf(stderr, "warning: positive control destroy failed: %s\n", controlError);
        }
        free(outControl);
        if (!controlMatchesExpectedB) {
            fprintf(stderr, "POSITIVE CONTROL FAILED: a fresh compile with the negated weights does not "
                "match expectedB. This calls the outcome-3 conclusion into question. Do not trust it "
                "without re-examining the host model and this probe.\n");
            printf("positive_control=FAILED reason=output_mismatch\n");
            return 3;
        }
        printf("positive_control=PASS\n");

        // Cleanup for the step-1 program. quip_ane_destroy consumes the
        // bridge retain and unloads. QuipAneProgram's dealloc
        // (ane_bridge.m:110-129) then removes the staging directory this
        // probe deliberately kept. dealloc fires when the `owner` local
        // below drops its own last strong reference, which ARC does at the
        // end of this scope: strictly after every Step 2, 3, and 4
        // measurement above, and after the positive control just run. The
        // "No such file or directory" NSLog line some runs print during
        // this cleanup is therefore harmless by construction; see the task
        // report for why it happens anyway.
        if (quip_ane_destroy(program, nativeError, sizeof(nativeError)) != 0) {
            fprintf(stderr, "warning: destroy failed: %s\n", nativeError);
        }
        owner = nil;
        free(weightsA);
        free(weightsB);
        free(fields);
        free(x);
        free(spinsFlat);
        free(thresholds);
        free(outA);
        free(outB);
        free(expectedA);
        free(expectedB);
        return 0;
    } @catch (NSException *exception) {
        fprintf(stderr, "probe_failed=%s\n", exception.reason.UTF8String);
        return 1;
    }
    }
}
