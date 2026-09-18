// Compile contention probe for bead quip-miner-metal-6jd: settles whether
// the Apple Neural Engine (ANE) compile path serializes across processes.
//
// contention.m answers the same question for evaluateWithQoS: and found
// overlap exactly two deep. It compiles once, outside its timed loop, so it
// says nothing about compile. Real mining compiles on every job, so if
// compile serializes then it, not evaluate, caps how many workers are worth
// running. This probe times quip_ane_create itself, which is the call that
// builds the Model Intermediate Language text, writes the weight blob, and
// runs compileWithQoS: and loadWithQoS:.
//
// Usage: compile-contention [start-at-us|now] [small|production]
//
// start-at-us is an absolute monotonicUS() deadline. Every copy allocates
// its inputs, waits for that deadline, and only then calls
// quip_ane_create, so N copies launched from a shell enter the compile
// together instead of staggered by process startup. Pass 0 to start at
// once, which is correct only for a single process.
//
// create_start_us and create_end_us come from monotonicUS(), which
// ane_bridge.m defines as clock_gettime_nsec_np(CLOCK_UPTIME_RAW): a
// system-wide monotonic clock, so the intervals are directly comparable
// across concurrently-launched processes. Comparing them is the whole
// measurement. Two readings distinguish the outcomes:
//
//   serialized  - the intervals do not overlap, and wall time from the
//                 first start to the last end grows about N times a single
//                 compile.
//   concurrent  - the intervals overlap, and each create_ms stays near a
//                 single compile's cost.
//
// A third outcome is possible and worth naming: the intervals overlap but
// every create_ms inflates, which means the compiles share a resource
// without queueing on a lock.
#include "../native/ane_bridge.m"
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const size_t kSmallChannels = 512;
static const size_t kSmallLengths[1] = {512};
static const size_t kSmallTiles = 1;

// The real tile layout from tests/fixtures/advantage2-system1.edges, the
// same eight lengths setup_profile.m and contention.m use.
static const size_t kProdChannels = 4608;
static const size_t kProdLengths[8] = {857, 849, 817, 740, 685, 480, 136, 13};
static const size_t kProdTiles = 8;

// BLOCK_SWEEPS in crates/ane-miner/src/native.rs. Unlike contention.m, which
// is pinned to 2 so Task 2's numbers reproduce, this probe measures the
// compile as production runs it today, so it tracks the current value.
static const size_t kSweeps = 1;

int main(int argc, const char **argv) {
    setbuf(stdout, NULL);
    if (argc > 3) {
        fprintf(stderr, "usage: compile-contention [start-at-us] [small|production]\n");
        return 2;
    }
    // `now` prints the current monotonicUS() reading and exits. A runner
    // needs it to pick a shared deadline, because CLOCK_UPTIME_RAW is not
    // reachable from a shell.
    if (argc >= 2 && strcmp(argv[1], "now") == 0) {
        printf("%llu\n", (unsigned long long)monotonicUS());
        return 0;
    }
    uint64_t startAt = argc >= 2 ? strtoull(argv[1], NULL, 10) : 0;
    const char *shapeName = argc == 3 ? argv[2] : "small";
    BOOL production = strcmp(shapeName, "production") == 0;
    if (!production && strcmp(shapeName, "small") != 0) {
        fprintf(stderr, "unknown shape: %s\n", shapeName);
        return 2;
    }
    size_t channels = production ? kProdChannels : kSmallChannels;
    const size_t *lengths = production ? kProdLengths : kSmallLengths;
    size_t tiles = production ? kProdTiles : kSmallTiles;

    @autoreleasepool {
    @try {
        char error[256];

        // Same synthetic construction setup_profile.m and contention.m use:
        // couplings in {-1, 1} from seed 123, zero fields. Built before the
        // barrier so allocation never lands inside the timed window.
        size_t weightCount = 0;
        for (size_t i = 0; i < tiles; ++i) weightCount += ((lengths[i] + 31) / 32 * 32) * channels;
        int8_t *weights = malloc(weightCount);
        int8_t *fields = calloc(channels, 1);
        if (weights == NULL || fields == NULL) {
            fprintf(stderr, "weight or field allocation failed\n");
            return 2;
        }
        srandom(123);
        for (size_t i = 0; i < weightCount; ++i) weights[i] = (random() & 1) ? (int8_t)1 : (int8_t)-1;

        // Barrier. Sleep in short slices rather than spinning, so a waiting
        // copy does not burn a core that a compiling copy could use.
        uint64_t waitStart = monotonicUS();
        while (startAt != 0 && monotonicUS() < startAt) {
            struct timespec slice = {.tv_sec = 0, .tv_nsec = 200000};
            nanosleep(&slice, NULL);
        }
        uint64_t barrierWaitUS = monotonicUS() - waitStart;

        uint64_t createStart = monotonicUS();
        void *program = NULL;
        int created = quip_ane_create(channels, lengths, tiles, kSweeps, weights, weightCount,
            fields, channels, &program, error, sizeof(error));
        uint64_t createEnd = monotonicUS();
        if (created != 0) {
            fprintf(stderr, "create failed: %s\n", error);
            return 2;
        }
        free(weights);
        free(fields);

        if (quip_ane_destroy(program, error, sizeof(error)) != 0) {
            fprintf(stderr, "destroy failed: %s\n", error);
            return 2;
        }

        printf("{\"pid\":%d,\"shape\":\"%s\",\"channels\":%zu,\"sweeps\":%zu,"
            "\"create_ms\":%.3f,\"create_start_us\":%llu,\"create_end_us\":%llu,"
            "\"barrier_wait_us\":%llu,\"late\":%s}\n",
            getpid(), production ? "production" : "small", channels, kSweeps,
            (createEnd - createStart) / 1000.0,
            (unsigned long long)createStart, (unsigned long long)createEnd,
            (unsigned long long)barrierWaitUS,
            (startAt != 0 && createStart > startAt + 5000) ? "true" : "false");
        return 0;
    } @catch (NSException *exception) {
        fprintf(stderr, "probe_failed=%s\n", exception.reason.UTF8String);
        return 1;
    }
    }
}
