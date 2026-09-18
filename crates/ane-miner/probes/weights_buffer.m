// Bead quip-miner-metal-kyk, first experiment: does the unused
// weightsBuffer argument on _ANERequest let one compiled program run with
// different couplings?
//
// The topology is fixed, so only h and J change per job. If a compiled
// program can take its couplings from a buffer supplied per dispatch, then
// one compile serves every job on that topology and the 155 ms per-job
// compile disappears. Unlike bead quip-miner-metal-yba, which moved the
// couplings to a graph input and lost both speed and sweep fusing, this
// leaves the graph untouched.
//
// _ANERequest already takes the argument, and the property is typed
// _ANEIOSurfaceObject, the same wrapper ane_bridge.m builds for its state
// and threshold surfaces. quip_ane_create passes nil for it at
// ane_bridge.m:360. This probe passes a real surface instead.
//
// Two earlier negative results this does not repeat. Task 4 overwrote the
// source blob weights/weight_data.bin and reloaded, which did not change
// the couplings. Bead yba moved the couplings into the graph as an input,
// which worked but cost more than the compile it removed. Neither touched
// the compiled program's own weights.
//
// This mirrors the create path in quip_ane_create, ane_bridge.m:251, and
// reuses that file's static makeMIL, makeWeightBlob, makeSurface, and
// QuipAneProgram helpers unchanged, the same way weight_reload.m and
// setup_profile.m do. Production bridge entry points are unchanged.
//
// Outcomes printed as outcome=<name>:
//   BUFFER_HONOURED - the device result matches the buffer's couplings.
//                     The route works.
//   BUFFER_IGNORED  - the result matches the compiled couplings. The
//                     argument does not rebind weights this way.
//   BUFFER_REJECTED - the evaluation failed with a non-nil buffer, which
//                     says the argument is read but wants another shape.
//   INDETERMINATE   - the positive control failed, so nothing is proven.
#include "../native/ane_bridge.m"
#include <mach/mach_time.h>
#include <stdlib.h>

static mach_timebase_info_data_t g_tb;
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static uint64_t now(void) { return mach_absolute_time(); }

// Small shape: one tile covering every channel, so the weight matrix is
// square and every row's conv sum is a plain matrix-vector product with no
// padding rows to trim.
static const size_t kChannels = 512;
static const size_t kLengths[1] = {512};
static const size_t kTiles = 1;
static const size_t kSweeps = 2;

// Ring topology: node i couples to node i+1 (mod channels) with weight +1.
// Row-major [output_channel, input_channel] matches the conv weight layout
// makeMIL emits at ane_bridge.m:160.
static void buildRingWeights(int8_t *weights, size_t channels) {
    memset(weights, 0, channels * channels);
    for (size_t row = 0; row < channels; ++row) weights[row * channels + (row + 1) % channels] = 1;
}

// Host emulation of one MIL sweep tile, the same arithmetic weight_reload.m
// uses: margin = threshold + own * raw, flip when margin >= 0. Fields are
// zero throughout, so h is omitted. Integer arithmetic, so "the output
// changed" is never accepted as a pass on its own.
static void hostStep(const int8_t *weights, size_t channels, const int8_t *x, int8_t *out) {
    for (size_t row = 0; row < channels; ++row) {
        int raw = 0;
        for (size_t col = 0; col < channels; ++col) raw += (int)weights[row * channels + col] * (int)x[col];
        int own = x[row];
        int margin = own * raw;
        out[row] = (int8_t)(margin >= 0 ? -own : own);
    }
}

static BOOL matchesReplicated(const int8_t *actual, size_t channels, const int8_t *expectedPerNode) {
    for (size_t c = 0; c < channels; ++c)
        for (size_t l = 0; l < 128; ++l)
            if (actual[c * 128 + l] != expectedPerNode[c]) return NO;
    return YES;
}

static uint64_t fnv1a64(const void *data, size_t length) {
    const uint8_t *bytes = data;
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < length; ++i) {
        hash ^= bytes[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

static void buildSpins(int8_t *x, size_t channels) {
    uint32_t state = 0x2545F491u;
    for (size_t i = 0; i < channels; ++i) {
        state = state * 1664525u + 1013904223u;
        x[i] = (state & 0x80000000u) ? (int8_t)1 : (int8_t)-1;
    }
}

// Copies bytes into a fresh IOSurface big enough to hold them. makeSurface
// takes an element count of fp16 values and rounds the allocation up to a
// 64 KB boundary, so ask for half the byte count, rounded up.
static IOSurfaceRef surfaceWithBytes(const void *data, size_t length) {
    IOSurfaceRef surface = makeSurface((length + 1) / 2);
    if (surface == NULL) return NULL;
    if (IOSurfaceLock(surface, 0, NULL) != kIOReturnSuccess) {
        CFRelease(surface);
        return NULL;
    }
    memset(IOSurfaceGetBaseAddress(surface), 0, IOSurfaceGetAllocSize(surface));
    memcpy(IOSurfaceGetBaseAddress(surface), data, length);
    IOSurfaceUnlock(surface, 0, NULL);
    return surface;
}

int main(int argc, const char **argv) {
    setbuf(stdout, NULL);
    mach_timebase_info(&g_tb);

    // Which bytes to put in the weights buffer. "blob" is the same format
    // makeWeightBlob writes for the compiler, chunk headers included. "raw"
    // is the bare fp16 matrix with no headers. The right format is unknown,
    // so try both rather than guessing once.
    // "zero" is the discriminator that does not depend on knowing the
    // tiling. A buffer of zeros decodes to zero weights under any
    // permutation, so raw becomes 0, margin becomes own * 0 which is at
    // least zero, and every node flips. An all-flipped result therefore
    // proves the buffer feeds the convolution even while its layout is
    // still unknown.
    // "none" is the control that makes the others readable. It builds the
    // same hand-made request and passes nil, exactly what the bridge does.
    // If that alone stops reproducing the baseline, then a changed result
    // with a buffer says nothing about the buffer, only about this probe's
    // request construction.
    const char *form = argc >= 2 ? argv[1] : "blob";
    BOOL rawForm = strcmp(form, "raw") == 0;
    BOOL zeroForm = strcmp(form, "zero") == 0;
    BOOL noneForm = strcmp(form, "none") == 0;
    if (!rawForm && !zeroForm && !noneForm && strcmp(form, "blob") != 0) {
        fprintf(stderr, "usage: weights-buffer [blob|raw|zero|none]\n");
        return 2;
    }

    @autoreleasepool {
    @try {
        NSError *error = nil;
        size_t weightCount = kChannels * kChannels;

        int8_t *weightsA = malloc(weightCount);
        int8_t *weightsB = malloc(weightCount);
        int8_t *fields = calloc(kChannels, 1); // zero fields: this tests the coupling swap alone
        int8_t *x = malloc(kChannels);
        if (weightsA == NULL || weightsB == NULL || fields == NULL || x == NULL) {
            fprintf(stderr, "allocation failed\n");
            return 2;
        }
        buildRingWeights(weightsA, kChannels);
        for (size_t i = 0; i < weightCount; ++i) weightsB[i] = (int8_t)(-weightsA[i]);
        buildSpins(x, kChannels);

        int8_t *expectA = malloc(kChannels);
        int8_t *expectB = malloc(kChannels);
        if (expectA == NULL || expectB == NULL) { fprintf(stderr, "expectation allocation failed\n"); return 2; }
        hostStep(weightsA, kChannels, x, expectA);
        hostStep(weightsB, kChannels, x, expectB);

        // The whole probe is meaningless if the two coupling sets produce
        // the same answer, so prove they do not before touching the device.
        BOOL predictionsDiffer = NO;
        for (size_t c = 0; c < kChannels; ++c) if (expectA[c] != expectB[c]) { predictionsDiffer = YES; break; }
        if (!predictionsDiffer) {
            fprintf(stderr, "negating the couplings does not change the host prediction; this probe cannot tell the two apart\n");
            return 2;
        }

        char nativeError[1024];
        void *program = NULL;
        uint64_t t0 = now();
        if (quip_ane_create(kChannels, kLengths, kTiles, kSweeps, weightsA, weightCount,
                fields, kChannels, &program, nativeError, sizeof(nativeError)) != 0) {
            fprintf(stderr, "create with couplings A failed: %s\n", nativeError);
            return 2;
        }
        printf("compile_and_load_ms=%.3f\n", tb_ms(now() - t0));

        NSData *blobB = makeWeightBlob(weightsB, kChannels, kLengths, kTiles, weightCount);
        NSData *blobA = makeWeightBlob(weightsA, kChannels, kLengths, kTiles, weightCount);
        uint64_t hashA = fnv1a64(blobA.bytes, blobA.length), hashB = fnv1a64(blobB.bytes, blobB.length);
        if (hashA == hashB) {
            fprintf(stderr, "blob A and blob B are the same bytes; negation produced no difference\n");
            return 2;
        }
        printf("blobA fnv1a64=%016llx blobB fnv1a64=%016llx bytes=%llu form=%s\n",
            (unsigned long long)hashA, (unsigned long long)hashB,
            (unsigned long long)blobB.length, form);

        // Raw form: the bare fp16 matrix, no chunk headers.
        size_t rawBytes = weightCount * sizeof(_Float16);
        _Float16 *rawB = malloc(rawBytes);
        if (rawB == NULL) { fprintf(stderr, "raw buffer allocation failed\n"); return 2; }
        for (size_t i = 0; i < weightCount; ++i) rawB[i] = (_Float16)weightsB[i];

        size_t inputElements = kChannels * 128;
        int8_t *spinsFlat = malloc(inputElements);
        uint8_t *thresholds = malloc(inputElements * kSweeps);
        if (spinsFlat == NULL || thresholds == NULL) { fprintf(stderr, "spin or threshold allocation failed\n"); return 2; }
        for (size_t c = 0; c < kChannels; ++c)
            for (size_t l = 0; l < 128; ++l) spinsFlat[c * 128 + l] = x[c];
        // Sweep 0 uses threshold 0 on every lane, so the flip decision
        // depends only on margin = own * raw. Sweep 1 uses 255, the same
        // "disable every flip" convention the production tail sweep uses,
        // so the sweep-0 result is the final result.
        memset(thresholds, 0, inputElements);
        memset(thresholds + inputElements, 255, inputElements);

        int8_t *out = malloc(inputElements);
        if (out == NULL) { fprintf(stderr, "output allocation failed\n"); return 2; }

        // === Step 1: baseline, weightsBuffer nil, the production path ===
        QuipAneTimes times;
        if (quip_ane_reset(program, spinsFlat, inputElements, nativeError, sizeof(nativeError)) != 0 ||
            quip_ane_evaluate(program, thresholds, inputElements * kSweeps, &times, nativeError, sizeof(nativeError)) != 0 ||
            quip_ane_read(program, out, inputElements, nativeError, sizeof(nativeError)) != 0) {
            fprintf(stderr, "baseline evaluation failed: %s\n", nativeError);
            return 2;
        }
        BOOL baselineMatchesA = matchesReplicated(out, kChannels, expectA);
        printf("baseline: matches_A=%d matches_B=%d\n",
            baselineMatchesA, matchesReplicated(out, kChannels, expectB));
        if (!baselineMatchesA) {
            fprintf(stderr, "baseline does not match the host prediction for couplings A; the harness is wrong\n");
            printf("outcome=INDETERMINATE\n");
            return 1;
        }

        // === Step 2: same program, couplings B supplied through the request ===
        QuipAneProgram *owner = (__bridge QuipAneProgram *)program;
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        Class requestClass = NSClassFromString(@"_ANERequest");
        IOSurfaceRef weightSurface = NULL;
        if (noneForm) {
            weightSurface = NULL;
        } else if (zeroForm) {
            void *zeros = calloc(1, blobB.length);
            if (zeros == NULL) { fprintf(stderr, "zero buffer allocation failed\n"); return 2; }
            weightSurface = surfaceWithBytes(zeros, blobB.length);
            free(zeros);
        } else if (rawForm) {
            weightSurface = surfaceWithBytes(rawB, rawBytes);
        } else {
            weightSurface = surfaceWithBytes(blobB.bytes, blobB.length);
        }
        if (!noneForm && weightSurface == NULL) { fprintf(stderr, "weight surface allocation failed\n"); return 2; }
        id weightWrapper = nil;
        if (!noneForm) {
            weightWrapper = [surfaceClass objectWithIOSurface:weightSurface];
            if (weightWrapper == nil) { fprintf(stderr, "weight surface wrapping failed\n"); return 2; }
        }

        // Rebuild the same request the bridge builds, changing only
        // weightsBuffer. Surface 0 holds the state that quip_ane_reset just
        // staged, and surfaces 2 onward hold the thresholds.
        NSArray *wrappers = owner.wrappers;
        NSMutableArray *inputs = [NSMutableArray arrayWithObject:wrappers[0]];
        NSMutableArray *indices = [NSMutableArray arrayWithObject:@0];
        for (size_t sweep = 0; sweep < kSweeps; ++sweep) {
            [inputs addObject:wrappers[sweep + 2]];
            [indices addObject:@(sweep + 1)];
        }
        id request = [requestClass requestWithInputs:inputs inputIndices:indices
            outputs:@[wrappers[1]] outputIndices:@[@0]
            weightsBuffer:weightWrapper perfStats:nil procedureIndex:@0];
        if (request == nil) {
            fprintf(stderr, "request creation with a weights buffer returned nil\n");
            printf("outcome=BUFFER_REJECTED reason=request_nil\n");
            return 1;
        }

        if (quip_ane_reset(program, spinsFlat, inputElements, nativeError, sizeof(nativeError)) != 0) {
            fprintf(stderr, "reset before the buffer evaluation failed: %s\n", nativeError);
            return 2;
        }
        error = nil;
        BOOL evaluated = [owner.model evaluateWithQoS:21 options:@{} request:request error:&error];
        if (!evaluated) {
            printf("evaluate_with_buffer_failed=%s\n", error.description.UTF8String);
            printf("outcome=BUFFER_REJECTED reason=evaluate_failed\n");
            return 1;
        }
        // quip_ane_evaluate flips current after a successful dispatch, and
        // quip_ane_read reads surfaces[current]. This probe dispatches the
        // model directly, so it has to flip current itself. Without this the
        // read returns the input surface, which looks like a plausible
        // result: every lane agrees and about half the nodes differ from the
        // prediction, because the staged spins differ from the swept spins
        // exactly where a flip happened.
        owner->current = 1 - owner->current;
        if (quip_ane_read(program, out, inputElements, nativeError, sizeof(nativeError)) != 0) {
            fprintf(stderr, "read after the buffer evaluation failed: %s\n", nativeError);
            return 2;
        }
        BOOL bufferMatchesA = matchesReplicated(out, kChannels, expectA);
        BOOL bufferMatchesB = matchesReplicated(out, kChannels, expectB);

        // Zero weights make every margin own * 0, which is at least zero, so
        // every node flips. This prediction needs no knowledge of the tiling.
        int8_t *expectZero = malloc(kChannels);
        if (expectZero == NULL) { fprintf(stderr, "zero expectation allocation failed\n"); return 2; }
        for (size_t c = 0; c < kChannels; ++c) expectZero[c] = (int8_t)(-x[c]);
        BOOL bufferMatchesZero = matchesReplicated(out, kChannels, expectZero);

        // How far off is it? A count of differing nodes separates "a
        // different weight layout" from "unchanged" or "garbage".
        size_t differFromA = 0, laneDisagreements = 0;
        for (size_t c = 0; c < kChannels; ++c) {
            if (out[c * 128] != expectA[c]) ++differFromA;
            for (size_t l = 1; l < 128; ++l)
                if (out[c * 128 + l] != out[c * 128]) { ++laneDisagreements; break; }
        }
        printf("with_buffer: matches_A=%d matches_B=%d matches_all_flipped=%d\n",
            bufferMatchesA, bufferMatchesB, bufferMatchesZero);
        printf("with_buffer: nodes_differing_from_A=%zu of %zu lanes_disagreeing=%zu\n",
            differFromA, kChannels, laneDisagreements);
        printf("with_buffer: node[0..7]=[%d,%d,%d,%d,%d,%d,%d,%d] expectA[0..7]=[%d,%d,%d,%d,%d,%d,%d,%d]\n",
            out[0], out[128], out[256], out[384], out[512], out[640], out[768], out[896],
            expectA[0], expectA[1], expectA[2], expectA[3], expectA[4], expectA[5], expectA[6], expectA[7]);
        if (bufferMatchesZero) {
            printf("outcome=BUFFER_HONOURED form=zero note=zero weights flipped every node, so the buffer feeds the convolution\n");
        }

        // === Step 3: positive control, compile B and confirm B is visible ===
        void *controlProgram = NULL;
        if (quip_ane_create(kChannels, kLengths, kTiles, kSweeps, weightsB, weightCount,
                fields, kChannels, &controlProgram, nativeError, sizeof(nativeError)) != 0) {
            fprintf(stderr, "control create with couplings B failed: %s\n", nativeError);
            return 2;
        }
        int8_t *controlOut = malloc(inputElements);
        if (controlOut == NULL) { fprintf(stderr, "control output allocation failed\n"); return 2; }
        if (quip_ane_reset(controlProgram, spinsFlat, inputElements, nativeError, sizeof(nativeError)) != 0 ||
            quip_ane_evaluate(controlProgram, thresholds, inputElements * kSweeps, &times, nativeError, sizeof(nativeError)) != 0 ||
            quip_ane_read(controlProgram, controlOut, inputElements, nativeError, sizeof(nativeError)) != 0) {
            fprintf(stderr, "control evaluation failed: %s\n", nativeError);
            return 2;
        }
        BOOL controlMatchesB = matchesReplicated(controlOut, kChannels, expectB);
        printf("control_fresh_compile_B: matches_A=%d matches_B=%d\n",
            matchesReplicated(controlOut, kChannels, expectA), controlMatchesB);
        if (!controlMatchesB) {
            fprintf(stderr, "a fresh compile with couplings B does not produce B's prediction; nothing is proven\n");
            printf("outcome=INDETERMINATE\n");
            return 1;
        }

        if (bufferMatchesB && !bufferMatchesA) printf("outcome=BUFFER_HONOURED\n");
        else if (bufferMatchesA && !bufferMatchesB) printf("outcome=BUFFER_IGNORED\n");
        else printf("outcome=INDETERMINATE reason=matched_neither\n");

        quip_ane_destroy(controlProgram, nativeError, sizeof(nativeError));
        quip_ane_destroy(program, nativeError, sizeof(nativeError));
        if (weightSurface != NULL) CFRelease(weightSurface);
        return 0;
    } @catch (NSException *exception) {
        fprintf(stderr, "probe_failed=%s\n", exception.reason.UTF8String);
        return 1;
    }
    }
}
