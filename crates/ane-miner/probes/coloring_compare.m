// Bead quip-miner-metal-fjo, ANE side: does the four-colouring beat the
// greedy eight-colouring on the Apple Neural Engine?
//
// Each sweep walks its colour classes in strict sequence, because every
// class reads what the previous class wrote. The ANE crate colours greedily
// by descending degree, which gives eight classes of 857, 849, 817, 740,
// 685, 480, 136 and 13 nodes. The last two are too small to amortise a
// dispatch, and /tmp/quip-ane-architecture.txt:10529 records that the
// compiler spreads output channels across engine cores by strided
// round-robin, so a 13-node class leaves most of the array idle.
//
// A proper four-colouring of the same topology already exists on the Metal
// side, src/topology.rs advantage2_color, with classes of 1148, 1145, 1145
// and 1139 nodes. Its own test proves no edge joins two nodes of one
// colour, and both colourings cover the same 4,577 nodes.
//
// Four colours should win twice over. The chain per sweep halves, from
// eight dependent steps to four. The padded work also falls slightly,
// because padding each class to a multiple of 32 costs 4,704 rows under
// greedy and 4,608 under four colours.
//
// This probe times both layouts back to back through the real
// quip_ane_create and quip_ane_evaluate bridge calls, unchanged, the same
// way contention.m and setup_profile.m do. It measures dispatch cost only.
// Solution quality is a separate question that this probe does not touch,
// and the bead's acceptance criteria require it to be answered separately.
//
// Usage: coloring-compare [call-count] [sweeps]
//
// Production bridge entry points are unchanged.
#include "../native/ane_bridge.m"
#include <stdlib.h>
#include <string.h>

static const size_t kChannels = 4608;

// Greedy, descending degree then ascending index, as crates/ane-miner's
// graph.rs colours it. docs/perf/2026-09-16-ane-local-routing.md:13.
static const size_t kGreedyLengths[8] = {857, 849, 817, 740, 685, 480, 136, 13};
static const size_t kGreedyTiles = 8;

// src/topology.rs advantage2_color, verified proper by
// advantage2_four_colors_cover_all_edges_and_preserve_csr.
static const size_t kFourLengths[4] = {1148, 1145, 1145, 1139};
static const size_t kFourTiles = 4;

static const size_t kDefaultCalls = 200;

static size_t paddedRows(const size_t *lengths, size_t tiles) {
    size_t total = 0;
    for (size_t i = 0; i < tiles; ++i) total += (lengths[i] + 31) / 32 * 32;
    return total;
}

// Times `calls` back-to-back evaluations of one compiled program. Returns
// milliseconds for the whole loop, and reports compile cost separately.
static BOOL timeLayout(const char *name, const size_t *lengths, size_t tiles,
    size_t sweeps, size_t calls, double *loopMS, double *compileMS) {
    char error[1024];
    size_t nodes = 0, weightCount = 0;
    for (size_t i = 0; i < tiles; ++i) {
        nodes += lengths[i];
        weightCount += ((lengths[i] + 31) / 32 * 32) * kChannels;
    }

    int8_t *weights = malloc(weightCount);
    int8_t *fields = calloc(kChannels, 1);
    if (weights == NULL || fields == NULL) { fprintf(stderr, "allocation failed\n"); return NO; }
    // Couplings in {-1, 1} from seed 123, the same synthetic construction
    // setup_profile.m and contention.m use. Values do not change the cost of
    // a dense convolution, which is what this probe times.
    srandom(123);
    for (size_t i = 0; i < weightCount; ++i) weights[i] = (random() & 1) ? (int8_t)1 : (int8_t)-1;

    void *program = NULL;
    uint64_t compileStart = monotonicUS();
    if (quip_ane_create(kChannels, 128, lengths, tiles, sweeps, weights, weightCount,
            fields, kChannels, &program, error, sizeof(error)) != 0) {
        fprintf(stderr, "%s: create failed: %s\n", name, error);
        return NO;
    }
    *compileMS = (monotonicUS() - compileStart) / 1000.0;
    free(weights);
    free(fields);

    size_t inputElements = kChannels * 128;
    int8_t *spins = malloc(inputElements);
    if (spins == NULL) { fprintf(stderr, "spin allocation failed\n"); return NO; }
    srandom(456);
    for (size_t i = 0; i < inputElements; ++i) spins[i] = (random() & 1) ? (int8_t)1 : (int8_t)-1;
    if (quip_ane_reset(program, spins, inputElements, error, sizeof(error)) != 0) {
        fprintf(stderr, "%s: reset failed: %s\n", name, error);
        return NO;
    }
    free(spins);

    size_t thresholdCount = inputElements * sweeps;
    uint8_t *thresholds = malloc(thresholdCount);
    if (thresholds == NULL) { fprintf(stderr, "threshold allocation failed\n"); return NO; }
    srandom(789);
    for (size_t i = 0; i < thresholdCount; ++i) thresholds[i] = (uint8_t)(random() % 64);

    QuipAneTimes times;
    uint64_t loopStart = monotonicUS();
    for (size_t call = 0; call < calls; ++call) {
        if (quip_ane_evaluate(program, thresholds, thresholdCount, &times, error, sizeof(error)) != 0) {
            fprintf(stderr, "%s: evaluate failed at call %zu: %s\n", name, call, error);
            return NO;
        }
    }
    *loopMS = (monotonicUS() - loopStart) / 1000.0;
    free(thresholds);

    if (quip_ane_destroy(program, error, sizeof(error)) != 0) {
        fprintf(stderr, "%s: destroy failed: %s\n", name, error);
        return NO;
    }

    printf("{\"layout\":\"%s\",\"tiles\":%zu,\"nodes\":%zu,\"padded_rows\":%zu,"
        "\"sweeps\":%zu,\"calls\":%zu,\"compile_ms\":%.3f,\"loop_ms\":%.3f,"
        "\"ms_per_sweep\":%.4f}\n",
        name, tiles, nodes, paddedRows(lengths, tiles), sweeps, calls,
        *compileMS, *loopMS, *loopMS / (double)(calls * sweeps));
    return YES;
}

int main(int argc, const char **argv) {
    setbuf(stdout, NULL);
    if (argc > 3) { fprintf(stderr, "usage: coloring-compare [call-count] [sweeps]\n"); return 2; }
    size_t calls = argc >= 2 ? (size_t)strtoul(argv[1], NULL, 10) : kDefaultCalls;
    size_t sweeps = argc >= 3 ? (size_t)strtoul(argv[2], NULL, 10) : 1;
    if (calls == 0 || sweeps == 0 || sweeps > 8) {
        fprintf(stderr, "call count must be positive and sweeps must be 1 through 8\n");
        return 2;
    }

    @autoreleasepool {
    @try {
        // Both colourings must cover the same node count, or the comparison
        // is between two different problems.
        size_t greedyNodes = 0, fourNodes = 0;
        for (size_t i = 0; i < kGreedyTiles; ++i) greedyNodes += kGreedyLengths[i];
        for (size_t i = 0; i < kFourTiles; ++i) fourNodes += kFourLengths[i];
        if (greedyNodes != fourNodes) {
            fprintf(stderr, "the two colourings cover %zu and %zu nodes; not comparable\n",
                greedyNodes, fourNodes);
            return 2;
        }
        printf("{\"nodes\":%zu,\"greedy_padded_rows\":%zu,\"four_padded_rows\":%zu}\n",
            greedyNodes, paddedRows(kGreedyLengths, kGreedyTiles), paddedRows(kFourLengths, kFourTiles));

        double greedyLoop = 0, greedyCompile = 0, fourLoop = 0, fourCompile = 0;
        if (!timeLayout("greedy8", kGreedyLengths, kGreedyTiles, sweeps, calls, &greedyLoop, &greedyCompile)) return 2;
        if (!timeLayout("four4", kFourLengths, kFourTiles, sweeps, calls, &fourLoop, &fourCompile)) return 2;

        printf("{\"sweep_speedup\":%.4f,\"compile_ratio\":%.4f}\n",
            greedyLoop / fourLoop, fourCompile / greedyCompile);
        return 0;
    } @catch (NSException *exception) {
        fprintf(stderr, "probe_failed=%s\n", exception.reason.UTF8String);
        return 1;
    }
    }
}
