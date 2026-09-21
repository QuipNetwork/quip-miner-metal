// Bead quip-miner-metal-kyk, experiment 3: where do the compiled couplings
// live, and in what layout?
//
// We choose h and J, so we can plant a signature in them that cannot occur
// by chance, compile, and then search this process's address space for it.
// That is a known-plaintext attack on the compiled weight layout. It
// answers two questions at once: whether the decoded weight region is in
// our address space at all, and, from the distance between successive
// pieces of the signature, what tiling it uses.
//
// The notes claim a convolution weight sits in a 0xC0-stride layout and
// that editing the values leaves the program descriptor unchanged, so a
// host can patch weights in place without recompiling. Section 7.5, Table
// 23.10. Nothing here tests the patch yet. This locates the target.
//
// The couplings must be -1, 0, or 1, which ane_bridge.m:276 enforces, so
// the signature is a pseudorandom sequence over those three values. As
// fp16 they encode 0xBC00, 0x0000 and 0x3C00, so a few hundred consecutive
// elements give a byte string that will not appear by accident. This keeps
// the probe on the unmodified production create path.
//
// Two searches run, because they answer different questions:
//   contiguous - the whole signature in order. A hit means a region holds
//                the weights laid out the way the source blob is.
//   windows    - many short runs. Hits where the contiguous search fails
//                mean the weights are present but interleaved, and the gap
//                between consecutive window hits is the tiling stride.
//
// Production bridge entry points are unchanged.
#include "../native/ane_bridge.m"
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <stdlib.h>
#include <string.h>

static const size_t kChannels = 512;
static const size_t kLengths[1] = {512};
static const size_t kTiles = 1;
static const size_t kSweeps = 2;

// How much of the signature to search for in one piece. 256 elements is
// 512 bytes of fp16, far beyond what chance produces.
static const size_t kContiguousElements = 256;
// A short run that survives interleaving, used to measure the stride.
static const size_t kWindowElements = 16;
static const size_t kMaxHits = 64;

static void buildSignature(int8_t *weights, size_t count) {
    uint64_t state = 0x9E3779B97F4A7C15ULL;
    for (size_t i = 0; i < count; ++i) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        weights[i] = (int8_t)((int)(state % 3) - 1); // -1, 0, or 1
    }
}

static void encodeFp16(const int8_t *values, size_t count, uint8_t *out) {
    for (size_t i = 0; i < count; ++i) {
        _Float16 value = (_Float16)values[i];
        uint16_t bits;
        memcpy(&bits, &value, sizeof(bits));
        bits = CFSwapInt16HostToLittle(bits);
        memcpy(out + i * 2, &bits, sizeof(bits));
    }
}

// Walks this process's readable regions and counts occurrences of a byte
// pattern, recording the first addresses it finds.
// Addresses to ignore, because they are this probe's own copies of the
// pattern. Without this every scan finds its own needle and reports a hit
// that proves nothing.
#define kMaxExclusions 16
static mach_vm_address_t g_excludeStart[kMaxExclusions];
static mach_vm_address_t g_excludeEnd[kMaxExclusions];
static size_t g_excludeCount = 0;

// Overflowing this silently would leave one of the probe's own buffers in
// the search space, and it would match itself and be reported as a find.
// That happened once, so overflow aborts rather than returning quietly.
static void excludeRange(const void *base, size_t length) {
    if (g_excludeCount >= kMaxExclusions) {
        fprintf(stderr, "exclusion table overflow; a self-match would be reported as a hit\n");
        exit(2);
    }
    g_excludeStart[g_excludeCount] = (mach_vm_address_t)base;
    g_excludeEnd[g_excludeCount] = (mach_vm_address_t)base + length;
    ++g_excludeCount;
}

static BOOL excluded(mach_vm_address_t address) {
    for (size_t i = 0; i < g_excludeCount; ++i)
        if (address >= g_excludeStart[i] && address < g_excludeEnd[i]) return YES;
    return NO;
}

static size_t scanSelf(const uint8_t *pattern, size_t patternLength,
    mach_vm_address_t *hits, size_t maxHits, size_t *regionsScanned, uint64_t *bytesScanned) {
    size_t found = 0;
    *regionsScanned = 0;
    *bytesScanned = 0;
    mach_vm_address_t address = 0;
    kern_return_t status;
    for (;;) {
        mach_vm_size_t size = 0;
        vm_region_basic_info_data_64_t info;
        mach_msg_type_number_t infoCount = VM_REGION_BASIC_INFO_COUNT_64;
        mach_port_t object = MACH_PORT_NULL;
        status = mach_vm_region(mach_task_self(), &address, &size, VM_REGION_BASIC_INFO_64,
            (vm_region_info_t)&info, &infoCount, &object);
        if (status != KERN_SUCCESS) break;
        if ((info.protection & VM_PROT_READ) != 0 && size > 0 && size < (mach_vm_size_t)1 << 32) {
            uint8_t *copy = malloc((size_t)size);
            if (copy != NULL) {
                mach_vm_size_t copied = 0;
                if (mach_vm_read_overwrite(mach_task_self(), address, size,
                        (mach_vm_address_t)copy, &copied) == KERN_SUCCESS && copied >= patternLength) {
                    *regionsScanned += 1;
                    *bytesScanned += copied;
                    uint8_t *cursor = copy;
                    size_t remaining = (size_t)copied;
                    for (;;) {
                        uint8_t *hit = memmem(cursor, remaining, pattern, patternLength);
                        if (hit == NULL) break;
                        mach_vm_address_t where = address + (mach_vm_address_t)(hit - copy);
                        if (!excluded(where)) {
                            if (found < maxHits) hits[found] = where;
                            ++found;
                        }
                        size_t advance = (size_t)(hit - cursor) + 1;
                        if (advance >= remaining) break;
                        cursor += advance;
                        remaining -= advance;
                    }
                }
                free(copy);
            }
        }
        address += size;
    }
    return found;
}

int main(void) {
    setbuf(stdout, NULL);
    @autoreleasepool {
    @try {
        size_t weightCount = kChannels * kChannels;
        int8_t *weights = malloc(weightCount);
        int8_t *fields = calloc(kChannels, 1);
        if (weights == NULL || fields == NULL) { fprintf(stderr, "allocation failed\n"); return 2; }
        buildSignature(weights, weightCount);

        uint8_t *contiguous = malloc(kContiguousElements * 2);
        uint8_t *window = malloc(kWindowElements * 2);
        // The compiled form need not be little-endian fp16. Search the two
        // other plausible encodings as well, so an absent result means the
        // values are not here rather than that the encoding was guessed
        // wrong: the same values byte-swapped, and the raw int8 bytes.
        uint8_t *swapped = malloc(kContiguousElements * 2);
        uint8_t *rawBytes = malloc(kContiguousElements);
        if (contiguous == NULL || window == NULL || swapped == NULL || rawBytes == NULL) {
            fprintf(stderr, "pattern allocation failed\n");
            return 2;
        }
        encodeFp16(weights, kContiguousElements, contiguous);
        encodeFp16(weights, kWindowElements, window);
        for (size_t i = 0; i < kContiguousElements; ++i) {
            swapped[i * 2] = contiguous[i * 2 + 1];
            swapped[i * 2 + 1] = contiguous[i * 2];
            rawBytes[i] = (uint8_t)weights[i];
        }
        excludeRange(contiguous, kContiguousElements * 2);
        excludeRange(window, kWindowElements * 2);
        excludeRange(swapped, kContiguousElements * 2);
        excludeRange(rawBytes, kContiguousElements);
        excludeRange(weights, weightCount);

        char error[1024];
        void *program = NULL;
        if (quip_ane_create(kChannels, 128, kLengths, kTiles, kSweeps, weights, weightCount,
                fields, kChannels, &program, error, sizeof(error)) != 0) {
            fprintf(stderr, "create failed: %s\n", error);
            return 2;
        }
        printf("compiled_and_loaded=1 channels=%zu weight_elements=%zu\n", kChannels, weightCount);

        mach_vm_address_t hits[kMaxHits];
        size_t regions = 0;
        uint64_t bytes = 0;

        // Positive control. A live copy of the pattern that is deliberately
        // not excluded, so the scanner must find it. Without this, zero hits
        // below could equally mean the scanner does not work.
        uint8_t *control = malloc(kContiguousElements * 2);
        if (control == NULL) { fprintf(stderr, "control allocation failed\n"); return 2; }
        memcpy(control, contiguous, kContiguousElements * 2);
        size_t controlHits = scanSelf(contiguous, kContiguousElements * 2, hits, kMaxHits, &regions, &bytes);
        printf("scanner_control_hits=%zu (must be at least 1)\n", controlHits);
        if (controlHits == 0) {
            fprintf(stderr, "the scanner cannot find a pattern known to be mapped; absence proves nothing\n");
            printf("outcome=INDETERMINATE reason=scanner_broken\n");
            return 1;
        }
        excludeRange(control, kContiguousElements * 2);

        size_t contiguousHits = scanSelf(contiguous, kContiguousElements * 2, hits, kMaxHits, &regions, &bytes);
        printf("scan: regions=%zu bytes_scanned=%llu\n", regions, (unsigned long long)bytes);
        printf("contiguous_pattern_bytes=%zu hits=%zu\n", kContiguousElements * 2, contiguousHits);
        for (size_t i = 0; i < contiguousHits && i < 8; ++i)
            printf("  contiguous_hit[%zu]=0x%llx\n", i, (unsigned long long)hits[i]);

        size_t windowHits = scanSelf(window, kWindowElements * 2, hits, kMaxHits, &regions, &bytes);
        printf("window_pattern_bytes=%zu hits=%zu\n", kWindowElements * 2, windowHits);
        size_t shown = windowHits < 8 ? windowHits : 8;
        for (size_t i = 0; i < shown; ++i)
            printf("  window_hit[%zu]=0x%llx%s\n", i, (unsigned long long)hits[i],
                i > 0 ? "" : "");
        // Gaps between consecutive window hits are the tiling stride, if the
        // weights are interleaved rather than contiguous.
        for (size_t i = 1; i < shown; ++i)
            printf("  window_gap[%zu]=%lld (0x%llx)\n", i,
                (long long)(hits[i] - hits[i - 1]), (unsigned long long)(hits[i] - hits[i - 1]));

        size_t swappedHits = scanSelf(swapped, kContiguousElements * 2, hits, kMaxHits, &regions, &bytes);
        printf("byte_swapped_fp16_hits=%zu\n", swappedHits);
        size_t rawHits = scanSelf(rawBytes, kContiguousElements, hits, kMaxHits, &regions, &bytes);
        printf("raw_int8_hits=%zu\n", rawHits);

        if (contiguousHits == 0 && windowHits == 0) printf("outcome=SIGNATURE_ABSENT\n");
        else if (contiguousHits > 0) printf("outcome=SIGNATURE_CONTIGUOUS\n");
        else printf("outcome=SIGNATURE_INTERLEAVED\n");

        quip_ane_destroy(program, error, sizeof(error));
        return 0;
    } @catch (NSException *exception) {
        fprintf(stderr, "probe_failed=%s\n", exception.reason.UTF8String);
        return 1;
    }
    }
}
