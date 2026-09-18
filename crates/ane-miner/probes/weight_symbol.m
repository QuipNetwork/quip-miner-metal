// Closed. The production couplings moved to the sparse encoding on
// 2026-09-18 (constexpr_sparse_to_dense, see ane_bridge.m makeWeightBlob),
// so this probe's arithmetic over the dense fp16 blob describes the layout
// at the time of its measurement. Bead quip-miner-metal-kyk records the
// outcome; the probe is kept as that record and is not maintained.
//
// Bead quip-miner-metal-kyk, experiment 2: can a compiled program be given
// different couplings by loading a new instance bound to per-job weight
// files?
//
// The runtime shape, from crates/ane-miner/probes/weight_symbol_probe.m:
// _ANEWeight binds a weight symbol to a file URL, _ANEProcedureData groups
// weights under a procedure symbol, _ANEModelInstanceParameters carries the
// procedure data, and _ANEClient takes
// loadModelNewInstance:options:modelInstParams:qos:error:. The notes
// describe exactly this as the mechanism behind swappable weights without
// recompilation.
//
// Experiment 1 ruled out the other candidate: the weightsBuffer argument on
// _ANERequest is ignored for a program whose weights are compiled in as
// BLOBFILE constants. See crates/ane-miner/probes/weights_buffer.m.
//
// The weight symbol is a guess, so it is the first argument. The MIL names
// "@model_path/weights/weight_data.bin" at ane_bridge.m:160, while the
// runtime's own default is "weight.bin", so try several.
//
// Usage: weight-symbol [SYMBOL] [PROCEDURE]
//
// Outcomes printed as outcome=<name>:
//   INSTANCE_HONOURED - the new instance computes with the file's couplings.
//   INSTANCE_IGNORED  - the load succeeded and the couplings did not change.
//   LOAD_REFUSED      - the load failed. The error text says why, and an
//                       entitlement refusal is a definite answer.
//   INDETERMINATE     - a control failed, so nothing is proven.
//
// This reuses ane_bridge.m's static helpers unchanged, the same way
// weight_reload.m and setup_profile.m do. Production entry points are
// unchanged.
#include "../native/ane_bridge.m"
#include <objc/message.h>
#include <stdlib.h>

static const size_t kChannels = 512;
static const size_t kLengths[1] = {512};
static const size_t kTiles = 1;
static const size_t kSweeps = 2;

static void buildRingWeights(int8_t *weights, size_t channels) {
    memset(weights, 0, channels * channels);
    for (size_t row = 0; row < channels; ++row) weights[row * channels + (row + 1) % channels] = 1;
}

// margin = threshold + own * raw, flip when margin >= 0. Fields are zero.
static void hostStep(const int8_t *weights, size_t channels, const int8_t *x, int8_t *out) {
    for (size_t row = 0; row < channels; ++row) {
        int raw = 0;
        for (size_t col = 0; col < channels; ++col) raw += (int)weights[row * channels + col] * (int)x[col];
        int own = x[row];
        out[row] = (int8_t)((own * raw) >= 0 ? -own : own);
    }
}

static BOOL matchesReplicated(const int8_t *actual, size_t channels, const int8_t *expectedPerNode) {
    for (size_t c = 0; c < channels; ++c)
        for (size_t l = 0; l < 128; ++l)
            if (actual[c * 128 + l] != expectedPerNode[c]) return NO;
    return YES;
}

static void buildSpins(int8_t *x, size_t channels) {
    uint32_t state = 0x2545F491u;
    for (size_t i = 0; i < channels; ++i) {
        state = state * 1664525u + 1013904223u;
        x[i] = (state & 0x80000000u) ? (int8_t)1 : (int8_t)-1;
    }
}

int main(int argc, const char **argv) {
    setbuf(stdout, NULL);
    NSString *symbol = argc >= 2 ? @(argv[1]) : @"weight_data.bin";
    NSString *procedureSymbol = argc >= 3 ? @(argv[2]) : @"main";

    @autoreleasepool {
    @try {
        NSError *error = nil;
        size_t weightCount = kChannels * kChannels;
        int8_t *weightsA = malloc(weightCount);
        int8_t *weightsB = malloc(weightCount);
        int8_t *fields = calloc(kChannels, 1);
        int8_t *x = malloc(kChannels);
        int8_t *expectA = malloc(kChannels);
        int8_t *expectB = malloc(kChannels);
        if (!weightsA || !weightsB || !fields || !x || !expectA || !expectB) {
            fprintf(stderr, "allocation failed\n");
            return 2;
        }
        buildRingWeights(weightsA, kChannels);
        for (size_t i = 0; i < weightCount; ++i) weightsB[i] = (int8_t)(-weightsA[i]);
        buildSpins(x, kChannels);
        hostStep(weightsA, kChannels, x, expectA);
        hostStep(weightsB, kChannels, x, expectB);

        BOOL differ = NO;
        for (size_t c = 0; c < kChannels; ++c) if (expectA[c] != expectB[c]) { differ = YES; break; }
        if (!differ) { fprintf(stderr, "the two coupling sets predict the same result\n"); return 2; }

        char nativeError[1024];
        void *program = NULL;
        if (quip_ane_create(kChannels, kLengths, kTiles, kSweeps, weightsA, weightCount,
                fields, kChannels, &program, nativeError, sizeof(nativeError)) != 0) {
            fprintf(stderr, "create with couplings A failed: %s\n", nativeError);
            return 2;
        }

        size_t inputElements = kChannels * 128;
        int8_t *spinsFlat = malloc(inputElements);
        uint8_t *thresholds = malloc(inputElements * kSweeps);
        int8_t *out = malloc(inputElements);
        if (!spinsFlat || !thresholds || !out) { fprintf(stderr, "buffer allocation failed\n"); return 2; }
        for (size_t c = 0; c < kChannels; ++c)
            for (size_t l = 0; l < 128; ++l) spinsFlat[c * 128 + l] = x[c];
        memset(thresholds, 0, inputElements);
        memset(thresholds + inputElements, 255, inputElements);

        QuipAneTimes times;
        if (quip_ane_reset(program, spinsFlat, inputElements, nativeError, sizeof(nativeError)) != 0 ||
            quip_ane_evaluate(program, thresholds, inputElements * kSweeps, &times, nativeError, sizeof(nativeError)) != 0 ||
            quip_ane_read(program, out, inputElements, nativeError, sizeof(nativeError)) != 0) {
            fprintf(stderr, "baseline evaluation failed: %s\n", nativeError);
            return 2;
        }
        if (!matchesReplicated(out, kChannels, expectA)) {
            fprintf(stderr, "baseline does not match couplings A; the harness is wrong\n");
            printf("outcome=INDETERMINATE\n");
            return 1;
        }
        printf("baseline: matches_A=1\n");

        // Write couplings B where a weight symbol can point at them, in the
        // same blob format the compiler consumed for couplings A.
        NSData *blobB = makeWeightBlob(weightsB, kChannels, kLengths, kTiles);
        NSString *directory = [NSTemporaryDirectory() stringByAppendingPathComponent:
            [NSString stringWithFormat:@"quip-ane-weight-symbol-%d", getpid()]];
        if (![NSFileManager.defaultManager createDirectoryAtPath:directory
                withIntermediateDirectories:YES attributes:nil error:&error]) {
            fprintf(stderr, "weight directory creation failed: %s\n", error.description.UTF8String);
            return 2;
        }
        NSString *weightPath = [directory stringByAppendingPathComponent:@"weight_data.bin"];
        if (![blobB writeToFile:weightPath options:NSDataWritingAtomic error:&error]) {
            fprintf(stderr, "weight file write failed: %s\n", error.description.UTF8String);
            return 2;
        }
        printf("weight_file=%s bytes=%llu symbol=%s procedure=%s\n",
            weightPath.UTF8String, (unsigned long long)blobB.length,
            symbol.UTF8String, procedureSymbol.UTF8String);

        id (*call2)(id, SEL, id, id) = (id (*)(id, SEL, id, id))objc_msgSend;
        id weight = call2(NSClassFromString(@"_ANEWeight"),
            NSSelectorFromString(@"weightWithSymbolAndURL:weightURL:"),
            symbol, [NSURL fileURLWithPath:weightPath]);
        id procedureData = call2(NSClassFromString(@"_ANEProcedureData"),
            NSSelectorFromString(@"procedureDataWithSymbol:weightArray:"),
            procedureSymbol, @[weight]);
        id parameters = call2(NSClassFromString(@"_ANEModelInstanceParameters"),
            NSSelectorFromString(@"withProcedureData:procedureArray:"),
            procedureData, @[procedureData]);
        if (weight == nil || procedureData == nil || parameters == nil) {
            fprintf(stderr, "instance parameter construction failed\n");
            printf("outcome=INDETERMINATE reason=parameters_nil\n");
            return 1;
        }

        QuipAneProgram *owner = (__bridge QuipAneProgram *)program;
        id client = [owner.model respondsToSelector:NSSelectorFromString(@"sharedConnection")]
            ? ((id (*)(id, SEL))objc_msgSend)(owner.model, NSSelectorFromString(@"sharedConnection"))
            : nil;
        if (client == nil) {
            Class clientClass = NSClassFromString(@"_ANEClient");
            if ([clientClass respondsToSelector:NSSelectorFromString(@"sharedConnection")])
                client = ((id (*)(id, SEL))objc_msgSend)(clientClass, NSSelectorFromString(@"sharedConnection"));
        }
        printf("client=%s\n", client != nil ? object_getClassName(client) : "<nil>");
        if (client == nil) { printf("outcome=INDETERMINATE reason=no_client\n"); return 1; }

        SEL loadSelector = NSSelectorFromString(@"loadModelNewInstance:options:modelInstParams:qos:error:");
        if (![client respondsToSelector:loadSelector]) {
            printf("outcome=INDETERMINATE reason=client_lacks_selector\n");
            return 1;
        }
        BOOL (*loadCall)(id, SEL, id, id, id, unsigned int, NSError **) =
            (BOOL (*)(id, SEL, id, id, id, unsigned int, NSError **))objc_msgSend;
        // _ANEClient wants an _ANEModel, not the _ANEInMemoryModel wrapper.
        // connectionForLoadingModel: calls getUUID, which only _ANEModel
        // implements, so pass the wrapper's underlying model.
        id target = owner.model;
        if ([owner.model respondsToSelector:NSSelectorFromString(@"model")]) {
            id inner = ((id (*)(id, SEL))objc_msgSend)(owner.model, NSSelectorFromString(@"model"));
            if (inner != nil) target = inner;
        }
        printf("load_target=%s responds_to_get_uuid=%d\n", object_getClassName(target),
            [target respondsToSelector:NSSelectorFromString(@"getUUID")]);
        if (![target respondsToSelector:NSSelectorFromString(@"getUUID")]) {
            printf("outcome=INDETERMINATE reason=no_model_with_uuid\n");
            return 1;
        }

        error = nil;
        BOOL loaded = loadCall(client, loadSelector, target, @{}, parameters, 21, &error);
        printf("load_new_instance=%d\n", loaded);
        if (!loaded) {
            printf("load_error=%s\n", error != nil ? error.description.UTF8String : "<none>");
            printf("outcome=LOAD_REFUSED\n");
            return 1;
        }

        if (quip_ane_reset(program, spinsFlat, inputElements, nativeError, sizeof(nativeError)) != 0 ||
            quip_ane_evaluate(program, thresholds, inputElements * kSweeps, &times, nativeError, sizeof(nativeError)) != 0 ||
            quip_ane_read(program, out, inputElements, nativeError, sizeof(nativeError)) != 0) {
            fprintf(stderr, "evaluation after the instance load failed: %s\n", nativeError);
            return 2;
        }
        BOOL matchesA = matchesReplicated(out, kChannels, expectA);
        BOOL matchesB = matchesReplicated(out, kChannels, expectB);
        printf("after_instance_load: matches_A=%d matches_B=%d\n", matchesA, matchesB);

        void *controlProgram = NULL;
        if (quip_ane_create(kChannels, kLengths, kTiles, kSweeps, weightsB, weightCount,
                fields, kChannels, &controlProgram, nativeError, sizeof(nativeError)) != 0) {
            fprintf(stderr, "control create failed: %s\n", nativeError);
            return 2;
        }
        int8_t *controlOut = malloc(inputElements);
        if (!controlOut) { fprintf(stderr, "control allocation failed\n"); return 2; }
        if (quip_ane_reset(controlProgram, spinsFlat, inputElements, nativeError, sizeof(nativeError)) != 0 ||
            quip_ane_evaluate(controlProgram, thresholds, inputElements * kSweeps, &times, nativeError, sizeof(nativeError)) != 0 ||
            quip_ane_read(controlProgram, controlOut, inputElements, nativeError, sizeof(nativeError)) != 0) {
            fprintf(stderr, "control evaluation failed: %s\n", nativeError);
            return 2;
        }
        if (!matchesReplicated(controlOut, kChannels, expectB)) {
            fprintf(stderr, "a fresh compile with couplings B does not predict B\n");
            printf("outcome=INDETERMINATE reason=control_failed\n");
            return 1;
        }
        printf("control_fresh_compile_B: matches_B=1\n");

        if (matchesB && !matchesA) printf("outcome=INSTANCE_HONOURED\n");
        else if (matchesA && !matchesB) printf("outcome=INSTANCE_IGNORED\n");
        else printf("outcome=INDETERMINATE reason=matched_neither\n");

        quip_ane_destroy(controlProgram, nativeError, sizeof(nativeError));
        quip_ane_destroy(program, nativeError, sizeof(nativeError));
        [NSFileManager.defaultManager removeItemAtPath:directory error:NULL];
        return 0;
    } @catch (NSException *exception) {
        fprintf(stderr, "probe_failed=%s\n", exception.reason.UTF8String);
        return 1;
    }
    }
}
