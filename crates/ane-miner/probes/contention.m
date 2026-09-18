// Contention probe for Task 2 of the 2026-09-17 ANE throughput plan: settles
// whether the Apple Neural Engine (ANE) runs more than one program at a
// time. Compiles one program, then issues 200 back-to-back
// `evaluateWithQoS:options:request:error:` calls through the real
// quip_ane_create/quip_ane_reset/quip_ane_evaluate bridge calls, unchanged,
// same pattern setup_profile.m and single_call.m already use in this
// directory. Running N copies of this same binary at once, each with its
// own compiled program and its own IOSurfaces, is how step 3 in the brief
// measures concurrent device access. Production bridge entry points are
// unchanged.
//
// Two shapes are available (second argv, default "small"):
//   small       - one tile, 512 channels, two sweeps. Keeps compile short;
//                 characterizes round-trip overhead more than engine work.
//   production  - the real tile layout from
//                 tests/fixtures/advantage2-system1.edges, the same eight
//                 lengths setup_profile.m uses: 4,608 channels, tiles
//                 {857, 849, 817, 740, 685, 480, 136, 13}, two sweeps
//                 (BLOCK_SWEEPS in native.rs). Compile here costs about
//                 283.887 ms per docs/perf/2026-09-17-ane-setup-profile.md;
//                 that cost sits outside the timed loop below.
//
// `loop_start_us`/`loop_end_us` are absolute values from monotonicUS(),
// which ane_bridge.m defines as clock_gettime_nsec_np(CLOCK_UPTIME_RAW): a
// system-wide monotonic clock (time since boot, not per-process), so these
// timestamps are directly comparable across concurrently-launched
// processes. That comparison is what tells apart two different
// explanations for an aggregate rate that looks too high: real overlap
// filling each process's idle wait between dispatches, versus an arithmetic
// trap where per-process rates computed over non-overlapping windows are
// summed as though they were concurrent.
#include "../native/ane_bridge.m"
#include <stdlib.h>
#include <string.h>

static const size_t kSmallChannels = 512;
static const size_t kSmallLengths[1] = {512};
static const size_t kSmallTiles = 1;

// Same eight colors setup_profile.m uses: descending degree then ascending
// index over tests/fixtures/advantage2-system1.edges, exactly as graph.rs
// colors it. docs/perf/2026-09-16-ane-local-routing.md:13.
static const size_t kProdChannels = 4608;
static const size_t kProdLengths[8] = {857, 849, 817, 740, 685, 480, 136, 13};
static const size_t kProdTiles = 8;

// BLOCK_SWEEPS in crates/ane-miner/src/native.rs: every production compile
// call, --solve included, builds the program with exactly this many sweeps.
static const size_t kSweeps = 2;
static const size_t kDefaultCalls = 200;

int main(int argc, const char **argv) {
    setbuf(stdout, NULL);
    if (argc > 3) { fprintf(stderr, "usage: contention [call-count] [small|production]\n"); return 2; }
    size_t calls = argc >= 2 ? (size_t)strtoul(argv[1], NULL, 10) : kDefaultCalls;
    if (calls == 0) { fprintf(stderr, "call count must be positive\n"); return 2; }
    const char *shapeName = argc == 3 ? argv[2] : "small";
    BOOL production = strcmp(shapeName, "production") == 0;
    if (!production && strcmp(shapeName, "small") != 0) { fprintf(stderr, "unknown shape: %s\n", shapeName); return 2; }
    size_t channels = production ? kProdChannels : kSmallChannels;
    const size_t *lengths = production ? kProdLengths : kSmallLengths;
    size_t tiles = production ? kProdTiles : kSmallTiles;

    @autoreleasepool {
    @try {
        char error[256];

        // Same synthetic construction setup_profile.m uses: couplings in
        // {-1, 1} from seed 123, zero fields. Real neighbor data is not
        // needed; this probe measures dispatch throughput, not results.
        size_t weightCount = 0;
        for (size_t i = 0; i < tiles; ++i) weightCount += ((lengths[i] + 31) / 32 * 32) * channels;
        int8_t *weights = malloc(weightCount);
        int8_t *fields = calloc(channels, 1);
        if (weights == NULL || fields == NULL) { fprintf(stderr, "weight or field allocation failed\n"); return 2; }
        srandom(123);
        for (size_t i = 0; i < weightCount; ++i) weights[i] = (random() & 1) ? (int8_t)1 : (int8_t)-1;

        void *program = NULL;
        if (quip_ane_create(channels, lengths, tiles, kSweeps, weights, weightCount, fields, channels,
                &program, error, sizeof(error)) != 0) {
            fprintf(stderr, "create failed: %s\n", error);
            return 2;
        }
        free(weights);
        free(fields);

        size_t inputElements = channels * 128;
        int8_t *spins = malloc(inputElements);
        if (spins == NULL) { fprintf(stderr, "spin allocation failed\n"); return 2; }
        srandom(456);
        for (size_t i = 0; i < inputElements; ++i) spins[i] = (random() & 1) ? (int8_t)1 : (int8_t)-1;
        if (quip_ane_reset(program, spins, inputElements, error, sizeof(error)) != 0) {
            fprintf(stderr, "reset failed: %s\n", error);
            return 2;
        }
        free(spins);

        // Fixed thresholds reused for every call: this probe times dispatch
        // throughput, not the solver's search behavior.
        size_t thresholdCount = inputElements * kSweeps;
        uint8_t *thresholds = malloc(thresholdCount);
        if (thresholds == NULL) { fprintf(stderr, "threshold allocation failed\n"); return 2; }
        srandom(789);
        for (size_t i = 0; i < thresholdCount; ++i) thresholds[i] = (uint8_t)(random() % 64);

        QuipAneTimes times;
        uint64_t loopStart = monotonicUS();
        for (size_t call = 0; call < calls; ++call) {
            if (quip_ane_evaluate(program, thresholds, thresholdCount, &times, error, sizeof(error)) != 0) {
                fprintf(stderr, "evaluate failed at call %zu: %s\n", call, error);
                return 2;
            }
        }
        uint64_t loopEnd = monotonicUS();
        uint64_t elapsedUS = loopEnd - loopStart;
        free(thresholds);

        if (quip_ane_destroy(program, error, sizeof(error)) != 0) {
            fprintf(stderr, "destroy failed: %s\n", error);
            return 2;
        }

        double totalMS = elapsedUS / 1000.0;
        double callsPerSecond = (double)calls / ((double)elapsedUS / 1e6);
        printf("{\"pid\":%d,\"shape\":\"%s\",\"channels\":%zu,\"calls\":%zu,\"total_ms\":%.3f,"
            "\"calls_per_second\":%.3f,\"loop_start_us\":%llu,\"loop_end_us\":%llu}\n",
            getpid(), production ? "production" : "small", channels, calls, totalMS, callsPerSecond,
            (unsigned long long)loopStart, (unsigned long long)loopEnd);
        return 0;
    } @catch (NSException *exception) {
        fprintf(stderr, "probe_failed=%s\n", exception.reason.UTF8String);
        return 1;
    }
    }
}
