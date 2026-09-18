// Bead quip-miner-metal-fjo.7: what fraction of the engine does a sweep use,
// and can the coupling stream shrink?
//
// The reads study fitted sweep = 0.425 ms + 1.75 us x reads and read the
// fixed part as the 42.5 MB coupling stream. That names the bottleneck but
// never states utilization, which needs two ceilings measured on this host:
// the engine's MAC rate when weights fit on chip, and its weight-stream rate
// when they do not. This probe measures both with bare 1x1 convolutions,
// then runs the production sweep under different read layouts and weight
// encodings and reports each one's MAC rate and weight bytes.
//
// Layouts: reads sit in the last two tensor dimensions as [1, C, H, W]. The
// production program uses H=1, W=128 and the cost per read doubles past
// W=128. Whether that is a limit on W alone or on H x W decides whether one
// model can carry more reads per coupling stream.
//
// Encodings: fp16 is production. int8 halves the bytes through
// constexpr_blockwise_shift_scale. lut2 packs {0, +1, -1} into two bits
// through constexpr_lut_to_dense. sparse stores a one-bit mask and the
// nonzero values through constexpr_sparse_to_dense. The engine either
// streams the compressed form or densifies it at load, and the sweep time
// tells which. Every encoding must hash to the same output as fp16.
//
// Couplings are synthetic but shaped like the testnet's: 41,514 edges over
// 4,577 nodes in the Advantage2 four-colouring, never within a class,
// values in {-1, +1}, fields zero.
//
// Usage: utilization roofline BUDGET_MS
//        utilization sweep BUDGET_MS ENCODING HxW [HxW ...]
//        utilization dump ENCODING HxW
// Each timed loop runs until BUDGET_MS has elapsed, at least 20 calls.
#include "../native/ane_bridge.m"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static const size_t kChannels = 4608;
static const size_t kNodes = 4577;
static const size_t kEdges = 41514;
// The Advantage2 four-colouring, src/topology.rs advantage2_color.
static const size_t kLengths[4] = {1148, 1145, 1145, 1139};
static const size_t kTiles = 4;
static const size_t kMinCalls = 20;

typedef enum { EncFP16, EncInt8, EncLUT2, EncSparse } Encoding;

static const char *encodingName(Encoding enc) {
    static const char *names[] = {"fp16", "int8", "lut2", "sparse"};
    return names[enc];
}

static BOOL parseEncoding(const char *text, Encoding *enc) {
    for (Encoding e = EncFP16; e <= EncSparse; ++e) {
        if (strcmp(text, encodingName(e)) == 0) { *enc = e; return YES; }
    }
    return NO;
}

static uint64_t wallUS(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

// coremltools blob storage, version 2: a 64-byte header, then per blob a
// 64-byte metadata record followed by its payload, each 64-byte aligned.
// The MIL text references a blob by the offset of its metadata record.
typedef struct { NSMutableData *data; uint32_t count; size_t payloadBytes; } BlobWriter;

static BlobWriter blobBegin(void) {
    NSMutableData *data = [NSMutableData dataWithLength:64];
    uint32_t version = CFSwapInt32HostToLittle(2);
    memcpy((uint8_t *)data.mutableBytes + 4, &version, sizeof(version));
    return (BlobWriter){data, 0, 0};
}

static uint64_t blobAppend(BlobWriter *writer, uint32_t dtype, const void *bytes, size_t size, uint64_t paddingBits) {
    uint64_t metadataOffset = writer->data.length;
    uint8_t metadata[64] = {0};
    uint32_t sentinel = CFSwapInt32HostToLittle(0xDEADBEEF), littleType = CFSwapInt32HostToLittle(dtype);
    uint64_t littleSize = CFSwapInt64HostToLittle(size), littleOffset = CFSwapInt64HostToLittle(metadataOffset + 64);
    uint64_t littlePadding = CFSwapInt64HostToLittle(paddingBits);
    memcpy(metadata, &sentinel, 4);
    memcpy(metadata + 4, &littleType, 4);
    memcpy(metadata + 8, &littleSize, 8);
    memcpy(metadata + 16, &littleOffset, 8);
    memcpy(metadata + 24, &littlePadding, 8);
    [writer->data appendBytes:metadata length:64];
    [writer->data appendBytes:bytes length:size];
    size_t tail = (64 - writer->data.length % 64) % 64;
    if (tail) [writer->data increaseLengthBy:tail];
    writer->count += 1;
    writer->payloadBytes += size;
    return metadataOffset;
}

static NSData *blobFinish(BlobWriter *writer) {
    uint32_t count = CFSwapInt32HostToLittle(writer->count);
    memcpy(writer->data.mutableBytes, &count, sizeof(count));
    return writer->data;
}

// kBlobFP16 and kBlobUInt1 come from ane_bridge.m.
static const uint32_t kBlobInt8 = 4, kBlobUInt2 = 10;
static NSString *const kBlobPath = @"@model_path/weights/weight_data.bin";

// Appends the weight tensor `name`, shape [k, c, 1, 1], to the blob under
// the requested encoding and emits the MIL that produces it as fp16.
static void emitWeight(NSMutableString *mil, BlobWriter *blob, NSString *name, const int8_t *block, size_t k, size_t c, Encoding enc) {
    size_t n = k * c;
    switch (enc) {
    case EncFP16: {
        _Float16 *half = malloc(n * sizeof(_Float16));
        for (size_t i = 0; i < n; ++i) half[i] = (_Float16)block[i];
        uint64_t offset = blobAppend(blob, kBlobFP16, half, n * sizeof(_Float16), 0);
        free(half);
        [mil appendFormat:@"    tensor<fp16, [%zu, %zu, 1, 1]> %@ = const()[name=string(\"%@\"), val=tensor<fp16, [%zu, %zu, 1, 1]>(BLOBFILE(path=string(\"%@\"), offset=uint64(%llu)))];\n",
            k, c, name, name, k, c, kBlobPath, (unsigned long long)offset];
        return;
    }
    case EncInt8: {
        uint64_t offset = blobAppend(blob, kBlobInt8, block, n, 0);
        [mil appendFormat:@"    tensor<int8, [%zu, %zu, 1, 1]> %@q = const()[name=string(\"%@q\"), val=tensor<int8, [%zu, %zu, 1, 1]>(BLOBFILE(path=string(\"%@\"), offset=uint64(%llu)))];\n",
            k, c, name, name, k, c, kBlobPath, (unsigned long long)offset];
        [mil appendFormat:@"    tensor<fp16, [1, 1, 1, 1]> %@s = const()[name=string(\"%@s\"), val=tensor<fp16, [1, 1, 1, 1]>([1.0])];\n", name, name];
        [mil appendFormat:@"    tensor<fp16, [%zu, %zu, 1, 1]> %@ = constexpr_blockwise_shift_scale(data=%@q, scale=%@s)[name=string(\"%@\")];\n",
            k, c, name, name, name, name];
        return;
    }
    case EncLUT2: {
        // Palette index 0 -> 0, 1 -> +1, 2 -> -1. Four indices per byte,
        // element i in bits 2(i mod 4) upward, matching MILBlob's packing.
        size_t bytes = (n + 3) / 4;
        uint8_t *packed = calloc(bytes, 1);
        for (size_t i = 0; i < n; ++i) {
            uint8_t index = block[i] == 0 ? 0 : (block[i] > 0 ? 1 : 2);
            packed[i / 4] |= (uint8_t)(index << (2 * (i % 4)));
        }
        uint64_t offset = blobAppend(blob, kBlobUInt2, packed, bytes, bytes * 8 - n * 2);
        free(packed);
        [mil appendFormat:@"    tensor<uint2, [%zu, %zu, 1, 1]> %@i = const()[name=string(\"%@i\"), val=tensor<uint2, [%zu, %zu, 1, 1]>(BLOBFILE(path=string(\"%@\"), offset=uint64(%llu)))];\n",
            k, c, name, name, k, c, kBlobPath, (unsigned long long)offset];
        [mil appendFormat:@"    tensor<fp16, [1, 1, 1, 1, 4, 1]> %@l = const()[name=string(\"%@l\"), val=tensor<fp16, [1, 1, 1, 1, 4, 1]>([0.0, 1.0, -1.0, 0.0])];\n", name, name];
        [mil appendFormat:@"    tensor<fp16, [%zu, %zu, 1, 1]> %@ = constexpr_lut_to_dense(indices=%@i, lut=%@l)[name=string(\"%@\")];\n",
            k, c, name, name, name, name];
        return;
    }
    case EncSparse: {
        // One mask bit per element, element i in bit i mod 8 of byte i / 8,
        // and the nonzero values in mask order.
        size_t maskBytes = (n + 7) / 8;
        uint8_t *mask = calloc(maskBytes, 1);
        NSMutableData *values = [NSMutableData new];
        size_t nonzero = 0;
        for (size_t i = 0; i < n; ++i) {
            if (block[i] == 0) continue;
            mask[i / 8] |= (uint8_t)(1 << (i % 8));
            _Float16 value = (_Float16)block[i];
            [values appendBytes:&value length:sizeof(value)];
            nonzero += 1;
        }
        uint64_t maskOffset = blobAppend(blob, kBlobUInt1, mask, maskBytes, maskBytes * 8 - n);
        uint64_t dataOffset = blobAppend(blob, kBlobFP16, values.bytes, values.length, 0);
        free(mask);
        [mil appendFormat:@"    tensor<uint1, [%zu, %zu, 1, 1]> %@m = const()[name=string(\"%@m\"), val=tensor<uint1, [%zu, %zu, 1, 1]>(BLOBFILE(path=string(\"%@\"), offset=uint64(%llu)))];\n",
            k, c, name, name, k, c, kBlobPath, (unsigned long long)maskOffset];
        [mil appendFormat:@"    tensor<fp16, [%zu]> %@d = const()[name=string(\"%@d\"), val=tensor<fp16, [%zu]>(BLOBFILE(path=string(\"%@\"), offset=uint64(%llu)))];\n",
            nonzero, name, name, nonzero, kBlobPath, (unsigned long long)dataOffset];
        [mil appendFormat:@"    tensor<fp16, [%zu, %zu, 1, 1]> %@ = constexpr_sparse_to_dense(nonzero_data=%@d, mask=%@m)[name=string(\"%@\")];\n",
            k, c, name, name, name, name];
        return;
    }
    }
}

static NSString *programHeader(void) {
    return [NSString stringWithFormat:@"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}, {\"quip-ane-utilization\", \"%@\"}})]\n{\n", NSUUID.UUID.UUIDString];
}

static NSString *convConstants(void) {
    return @"    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n";
}

// A bare 1x1 convolution: weight [k, c, 1, 1] over an input [1, c, h, w].
static NSString *makeConvMIL(size_t k, size_t c, size_t h, size_t w, const int8_t *block, Encoding enc, BlobWriter *blob) {
    NSMutableString *mil = [NSMutableString stringWithString:programHeader()];
    [mil appendFormat:@"  func main<ios18>(tensor<fp16, [1, %zu, %zu, %zu]> x) {\n", c, h, w];
    [mil appendString:convConstants()];
    emitWeight(mil, blob, @"w0", block, k, c, enc);
    [mil appendFormat:@"    tensor<fp16, [1, %zu, %zu, %zu]> y = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w0, x=x)[name=string(\"y\")];\n", k, h, w];
    [mil appendString:@"  } -> (y);\n}\n"];
    return mil;
}

static NSString *shapeHW(size_t channels, size_t h, size_t w) {
    return [NSString stringWithFormat:@"tensor<fp16, [1, %zu, %zu, %zu]>", channels, h, w];
}

static void sliceHW(NSMutableString *mil, NSString *name, NSString *source, size_t begin, size_t count, size_t h, size_t w) {
    [mil appendFormat:@"    tensor<int32, [4]> %@begin = const()[name=string(\"%@begin\"), val=tensor<int32, [4]>([0,%zu,0,0])];\n", name, name, begin];
    [mil appendFormat:@"    tensor<int32, [4]> %@size = const()[name=string(\"%@size\"), val=tensor<int32, [4]>([1,%zu,%zu,%zu])];\n", name, name, count, h, w];
    [mil appendFormat:@"    %@ %@ = slice_by_size(x=%@, begin=%@begin, size=%@size)[name=string(\"%@\")];\n", shapeHW(count, h, w), name, source, name, name, name];
}

// The production sweep from ane_bridge.m makeMIL, one sweep, zero fields,
// with the read layout and the weight encoding lifted out. The operator
// chain is copied so that a difference from production is layout or
// encoding alone. `couplings` is [kChannels][kChannels], row = node.
static NSString *makeSweepMIL(size_t h, size_t w, Encoding enc, const int8_t *couplings, BlobWriter *blob, size_t *macPerCall) {
    NSMutableString *mil = [NSMutableString stringWithString:programHeader()];
    [mil appendFormat:@"  func main<ios18>(%@ a_state, %@ t0) {\n", shapeHW(kChannels, h, w), shapeHW(kChannels, h, w)];
    [mil appendString:convConstants()];
    [mil appendString:@"    int32 axis = const()[name=string(\"axis\"), val=int32(1)];\n"
        "    bool interleave = const()[name=string(\"interleave\"), val=bool(false)];\n"
        "    fp16 zero = const()[name=string(\"zero\"), val=fp16(0.0)];\n"
        "    fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];\n"
        "    fp16 minusTwo = const()[name=string(\"minusTwo\"), val=fp16(-2.0)];\n"];
    size_t begin = 0;
    *macPerCall = 0;
    for (size_t tile = 0; tile < kTiles; ++tile) {
        size_t count = kLengths[tile], padded = (count + 31) / 32 * 32;
        int8_t *block = calloc(padded * kChannels, 1);
        memcpy(block, couplings + begin * kChannels, count * kChannels);
        emitWeight(mil, blob, [NSString stringWithFormat:@"w%zu", tile], block, padded, kChannels, enc);
        free(block);
        *macPerCall += padded * kChannels * h * w;
        NSMutableArray *values = [NSMutableArray new];
        for (size_t row = 0; row < count; ++row) [values addObject:@"0.0"];
        [mil appendFormat:@"    tensor<fp16, [%zu]> hFlat%zu = const()[name=string(\"hFlat%zu\"), val=tensor<fp16, [%zu]>([%@])];\n", count, tile, tile, count, [values componentsJoinedByString:@","]];
        [mil appendFormat:@"    tensor<int32, [4]> hShape%zu = const()[name=string(\"hShape%zu\"), val=tensor<int32, [4]>([1,%zu,1,1])];\n", tile, tile, count];
        [mil appendFormat:@"    tensor<fp16, [1,%zu,1,1]> h%zu = reshape(x=hFlat%zu, shape=hShape%zu)[name=string(\"h%zu\")];\n", count, tile, tile, tile, tile];
        begin += count;
    }
    NSString *state = @"a_state";
    begin = 0;
    for (size_t tile = 0; tile < kTiles; ++tile) {
        size_t count = kLengths[tile], padded = (count + 31) / 32 * 32;
        NSString *prefix = [NSString stringWithFormat:@"s0c%zu", tile];
        NSString *own = [prefix stringByAppendingString:@"own"];
        NSString *threshold = [prefix stringByAppendingString:@"threshold"];
        NSString *raw = [prefix stringByAppendingString:@"raw"];
        NSString *js = [prefix stringByAppendingString:@"js"];
        sliceHW(mil, own, state, begin, count, h, w);
        sliceHW(mil, threshold, @"t0", begin, count, h, w);
        [mil appendFormat:@"    %@ %@ = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w%zu, x=%@)[name=string(\"%@\")];\n", shapeHW(padded, h, w), raw, tile, state, raw];
        sliceHW(mil, js, raw, 0, count, h, w);
        NSArray *ops = @[
            @[@"field", [NSString stringWithFormat:@"add(x=%@, y=h%zu)", js, tile]],
            @[@"signed", [NSString stringWithFormat:@"mul(x=%@, y=%@field)", own, prefix]],
            @[@"margin", [NSString stringWithFormat:@"add(x=%@, y=%@signed)", threshold, prefix]],
            @[@"shifted", [NSString stringWithFormat:@"add(x=%@margin, y=one)", prefix]],
            @[@"accept", [NSString stringWithFormat:@"clip(x=%@shifted, alpha=zero, beta=one)", prefix]],
            @[@"negative", [NSString stringWithFormat:@"mul(x=%@accept, y=minusTwo)", prefix]],
            @[@"factor", [NSString stringWithFormat:@"add(x=one, y=%@negative)", prefix]],
            @[@"updated", [NSString stringWithFormat:@"mul(x=%@, y=%@factor)", own, prefix]]
        ];
        for (NSArray *op in ops)
            [mil appendFormat:@"    %@ %@%@ = %@[name=string(\"%@%@\")];\n", shapeHW(count, h, w), prefix, op[0], op[1], prefix, op[0]];
        NSMutableArray *parts = [NSMutableArray new];
        if (begin > 0) {
            NSString *head = [prefix stringByAppendingString:@"head"];
            sliceHW(mil, head, state, 0, begin, h, w);
            [parts addObject:head];
        }
        [parts addObject:[prefix stringByAppendingString:@"updated"]];
        if (begin + count < kChannels) {
            NSString *tail = [prefix stringByAppendingString:@"tail"];
            sliceHW(mil, tail, state, begin + count, kChannels - begin - count, h, w);
            [parts addObject:tail];
        }
        state = [prefix stringByAppendingString:@"state"];
        [mil appendFormat:@"    %@ %@ = concat(values=(%@), axis=axis, interleave=interleave)[name=string(\"%@\")];\n", shapeHW(kChannels, h, w), state, [parts componentsJoinedByString:@", "], state];
        begin += count;
    }
    [mil appendFormat:@"  } -> (%@);\n}\n", state];
    return mil;
}

// Synthetic couplings shaped like the testnet's problems: kEdges edges over
// kNodes nodes, none within a colour class, values in {-1, +1}, symmetric.
// Padding channels past kNodes stay zero.
static int8_t *makeCouplings(void) {
    int8_t *couplings = calloc(kChannels * kChannels, 1);
    size_t *classOf = malloc(kChannels * sizeof(size_t));
    size_t begin = 0;
    for (size_t tile = 0; tile < kTiles; ++tile) {
        for (size_t row = 0; row < kLengths[tile]; ++row) classOf[begin + row] = tile;
        begin += kLengths[tile];
    }
    srandom(123);
    size_t placed = 0;
    while (placed < kEdges) {
        size_t u = (size_t)random() % kNodes, v = (size_t)random() % kNodes;
        if (classOf[u] == classOf[v] || couplings[u * kChannels + v] != 0) continue;
        int8_t value = (random() & 1) ? (int8_t)1 : (int8_t)-1;
        couplings[u * kChannels + v] = value;
        couplings[v * kChannels + u] = value;
        placed += 1;
    }
    free(classOf);
    return couplings;
}

static int8_t *makeDense(size_t k, size_t c, unsigned seed) {
    int8_t *block = malloc(k * c);
    srandom(seed);
    for (size_t i = 0; i < k * c; ++i) block[i] = (random() & 1) ? (int8_t)1 : (int8_t)-1;
    return block;
}

typedef struct {
    size_t calls;
    double compileMS, msPerCall;
    uint64_t loopStartUS, loopEndUS, wallStartUS, wallEndUS, outputHash;
    size_t nonSpin;
} RunResult;

static BOOL loadFramework(void) {
    static void *framework;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        framework = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    });
    return framework != NULL;
}

// Compiles and loads one program, stages its inputs, times evaluations
// until the budget elapses, and hashes the output surface. Input 0 is
// staged as spins, input 1 as thresholds. Returns NO on any failure,
// which a rejected encoding produces, and that rejection is a result.
static BOOL runProgram(NSString *milText, NSData *blob, size_t inputs, size_t inputElements, size_t outputElements, double budgetMS, RunResult *result) {
    @autoreleasepool {
        NSError *error = nil;
        if (!loadFramework()) { fprintf(stderr, "AppleNeuralEngine framework unavailable\n"); return NO; }
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        Class requestClass = NSClassFromString(@"_ANERequest");
        NSData *plist = [NSPropertyListSerialization dataWithPropertyList:@{} format:NSPropertyListXMLFormat_v1_0 options:0 error:&error];
        if (plist == nil) { fprintf(stderr, "plist failed\n"); return NO; }
        NSData *mil = [milText dataUsingEncoding:NSUTF8StringEncoding];
        id descriptor = [[descriptorClass alloc] initWithNetworkText:mil weights:@{} optionsPlist:plist isMILModel:YES];
        if (descriptor == nil) { fprintf(stderr, "descriptor failed\n"); return NO; }
        id model = [modelClass inMemoryModelWithDescriptor:descriptor];
        if (model == nil) { fprintf(stderr, "model failed\n"); return NO; }

        // The compiler resolves "@model_path" against a directory named for
        // the model's identifier; any other name fails as InvalidMILProgram.
        NSString *directory = [NSTemporaryDirectory() stringByAppendingPathComponent:[model hexStringIdentifier]];
        if (mkdir(directory.fileSystemRepresentation, 0700) != 0) { fprintf(stderr, "staging dir failed\n"); return NO; }
        NSString *weightDirectory = [directory stringByAppendingPathComponent:@"weights"];
        [NSFileManager.defaultManager createDirectoryAtPath:weightDirectory withIntermediateDirectories:NO attributes:nil error:&error];
        [mil writeToFile:[directory stringByAppendingPathComponent:@"model.mil"] options:NSDataWritingAtomic error:&error];
        [blob writeToFile:[weightDirectory stringByAppendingPathComponent:@"weight_data.bin"] options:NSDataWritingAtomic error:&error];

        uint64_t compileStart = monotonicUS();
        BOOL compiled = [model compileWithQoS:21 options:@{} error:&error];
        if (!compiled) {
            fprintf(stderr, "compile failed: %s\n", error.description.UTF8String);
            [NSFileManager.defaultManager removeItemAtPath:directory error:NULL];
            return NO;
        }
        result->compileMS = (monotonicUS() - compileStart) / 1000.0;
        if (![model loadWithQoS:21 options:@{} error:&error]) {
            fprintf(stderr, "load failed: %s\n", error.description.UTF8String);
            [NSFileManager.defaultManager removeItemAtPath:directory error:NULL];
            return NO;
        }
        [NSFileManager.defaultManager removeItemAtPath:directory error:NULL];

        IOSurfaceRef surfaces[3] = {0};
        NSMutableArray *inputWrappers = [NSMutableArray new], *inputIndices = [NSMutableArray new];
        for (size_t i = 0; i < inputs; ++i) {
            surfaces[i] = makeSurface(inputElements);
            if (surfaces[i] == NULL) { fprintf(stderr, "surface failed\n"); return NO; }
            [inputWrappers addObject:[surfaceClass objectWithIOSurface:surfaces[i]]];
            [inputIndices addObject:@(i)];
        }
        surfaces[inputs] = makeSurface(outputElements);
        if (surfaces[inputs] == NULL) { fprintf(stderr, "output surface failed\n"); return NO; }
        id outputWrapper = [surfaceClass objectWithIOSurface:surfaces[inputs]];

        int8_t *spins = malloc(inputElements);
        uint8_t *thresholds = malloc(inputElements);
        if (spins == NULL || thresholds == NULL) { fprintf(stderr, "stage alloc failed\n"); return NO; }
        srandom(456);
        for (size_t i = 0; i < inputElements; ++i) spins[i] = (random() & 1) ? (int8_t)1 : (int8_t)-1;
        srandom(789);
        for (size_t i = 0; i < inputElements; ++i) thresholds[i] = (uint8_t)(random() % 64);
        NSString *stagingError = nil;
        if (!stageSurface(surfaces[0], spins, inputElements, NO, &stagingError) ||
            (inputs > 1 && !stageSurface(surfaces[1], thresholds, inputElements, YES, &stagingError))) {
            fprintf(stderr, "staging failed: %s\n", stagingError.UTF8String);
            return NO;
        }
        free(spins);
        free(thresholds);

        id request = [requestClass requestWithInputs:inputWrappers inputIndices:inputIndices
            outputs:@[outputWrapper] outputIndices:@[@0]
            weightsBuffer:nil perfStats:nil procedureIndex:@0];
        if (request == nil) { fprintf(stderr, "request failed\n"); return NO; }

        uint64_t budgetUS = (uint64_t)(budgetMS * 1000.0);
        result->wallStartUS = wallUS();
        result->loopStartUS = monotonicUS();
        result->calls = 0;
        do {
            if (![model evaluateWithQoS:21 options:@{} request:request error:&error]) {
                fprintf(stderr, "evaluate failed: %s\n", error.description.UTF8String);
                return NO;
            }
            result->calls += 1;
        } while (monotonicUS() - result->loopStartUS < budgetUS || result->calls < kMinCalls);
        result->loopEndUS = monotonicUS();
        result->wallEndUS = wallUS();
        result->msPerCall = (result->loopEndUS - result->loopStartUS) / 1000.0 / (double)result->calls;

        // FNV-1a over the output bytes: identical programs must agree exactly.
        IOSurfaceRef output = surfaces[inputs];
        if (IOSurfaceLock(output, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) { fprintf(stderr, "output lock failed\n"); return NO; }
        const _Float16 *values = IOSurfaceGetBaseAddress(output);
        const uint8_t *bytes = (const uint8_t *)values;
        uint64_t hash = 1469598103934665603ULL;
        for (size_t i = 0; i < outputElements * sizeof(_Float16); ++i) { hash ^= bytes[i]; hash *= 1099511628211ULL; }
        result->outputHash = hash;
        result->nonSpin = 0;
        for (size_t i = 0; i < outputElements; ++i) if (values[i] != (_Float16)1 && values[i] != (_Float16)-1) result->nonSpin += 1;
        IOSurfaceUnlock(output, kIOSurfaceLockReadOnly, NULL);

        [model unloadWithQoS:21 error:&error];
        for (size_t i = 0; i <= inputs; ++i) if (surfaces[i]) CFRelease(surfaces[i]);
        return YES;
    }
}

static void printResult(const char *kind, const char *name, Encoding enc, size_t k, size_t c, size_t h, size_t w,
                        size_t macPerCall, size_t weightBytes, const RunResult *r) {
    printf("{\"kind\":\"%s\",\"name\":\"%s\",\"encoding\":\"%s\",\"k\":%zu,\"c\":%zu,\"h\":%zu,\"w\":%zu,\"reads\":%zu,"
        "\"calls\":%zu,\"compile_ms\":%.3f,\"ms_per_call\":%.4f,\"mac_per_call\":%zu,\"weight_bytes\":%zu,"
        "\"loop_start_us\":%llu,\"loop_end_us\":%llu,\"wall_start_us\":%llu,\"wall_end_us\":%llu,"
        "\"output_hash\":\"%016llx\",\"non_spin\":%zu}\n",
        kind, name, encodingName(enc), k, c, h, w, h * w, r->calls, r->compileMS, r->msPerCall, macPerCall, weightBytes,
        (unsigned long long)r->loopStartUS, (unsigned long long)r->loopEndUS,
        (unsigned long long)r->wallStartUS, (unsigned long long)r->wallEndUS,
        (unsigned long long)r->outputHash, r->nonSpin);
}

static void printRejected(const char *kind, const char *name, Encoding enc, size_t h, size_t w) {
    printf("{\"kind\":\"%s\",\"name\":\"%s\",\"encoding\":\"%s\",\"h\":%zu,\"w\":%zu,\"reads\":%zu,\"status\":\"rejected\"}\n",
        kind, name, encodingName(enc), h, w, h * w);
}

static BOOL parseLayout(const char *text, size_t *h, size_t *w) {
    return sscanf(text, "%zux%zu", h, w) == 2 && *h > 0 && *w > 0;
}

// Ceilings. Compute-bound: weights that fit on chip over a wide activation,
// so the MAC array is the limit. Stream-bound: the production weight size
// over a narrow activation, so the weight path is the limit. Both are bare
// convolutions, so each rate is a floor on the engine's ceiling rather
// than the ceiling itself.
static int runRoofline(double budgetMS) {
    struct { const char *name; size_t k, c, h, w; } kernels[] = {
        {"compute_1024x1024_h32", 1024, 1024, 32, 128},
        {"compute_2048x2048_h32", 2048, 2048, 32, 128},
        {"compute_2048x2048_h64", 2048, 2048, 64, 128},
        {"compute_4096x4096_h8", 4096, 4096, 8, 128},
        {"stream_4608x4608_w1", 4608, 4608, 1, 1},
        {"stream_4608x4608_w32", 4608, 4608, 1, 32},
        {"stream_4608x4608_w128", 4608, 4608, 1, 128},
    };
    for (size_t i = 0; i < sizeof(kernels) / sizeof(kernels[0]); ++i) {
        @autoreleasepool {
            int8_t *block = makeDense(kernels[i].k, kernels[i].c, 321);
            BlobWriter blob = blobBegin();
            NSString *mil = makeConvMIL(kernels[i].k, kernels[i].c, kernels[i].h, kernels[i].w, block, EncFP16, &blob);
            free(block);
            NSData *blobData = blobFinish(&blob);
            RunResult r = {0};
            size_t positions = kernels[i].h * kernels[i].w;
            if (!runProgram(mil, blobData, 1, kernels[i].c * positions, kernels[i].k * positions, budgetMS, &r)) {
                printRejected("roofline", kernels[i].name, EncFP16, kernels[i].h, kernels[i].w);
                continue;
            }
            printResult("roofline", kernels[i].name, EncFP16, kernels[i].k, kernels[i].c, kernels[i].h, kernels[i].w,
                kernels[i].k * kernels[i].c * positions, blob.payloadBytes, &r);
        }
    }
    return 0;
}

static int runSweeps(double budgetMS, Encoding enc, int layoutCount, const char **layouts) {
    int8_t *couplings = makeCouplings();
    for (int i = 0; i < layoutCount; ++i) {
        @autoreleasepool {
            size_t h, w;
            if (!parseLayout(layouts[i], &h, &w)) { fprintf(stderr, "bad layout %s\n", layouts[i]); continue; }
            BlobWriter blob = blobBegin();
            size_t macPerCall = 0;
            NSString *mil = makeSweepMIL(h, w, enc, couplings, &blob, &macPerCall);
            NSData *blobData = blobFinish(&blob);
            RunResult r = {0};
            size_t elements = kChannels * h * w;
            if (!runProgram(mil, blobData, 2, elements, elements, budgetMS, &r)) {
                printRejected("sweep", "sweep", enc, h, w);
                continue;
            }
            printResult("sweep", "sweep", enc, kChannels, kChannels, h, w, macPerCall, blob.payloadBytes, &r);
        }
    }
    free(couplings);
    return 0;
}

int main(int argc, const char **argv) {
    setbuf(stdout, NULL);
    if (argc < 3) { fprintf(stderr, "usage: utilization roofline BUDGET_MS | sweep BUDGET_MS ENCODING HxW... | dump ENCODING HxW\n"); return 2; }
    @try {
        if (strcmp(argv[1], "dump") == 0) {
            Encoding enc;
            size_t h, w;
            if (argc < 4 || !parseEncoding(argv[2], &enc) || !parseLayout(argv[3], &h, &w)) { fprintf(stderr, "dump ENCODING HxW\n"); return 2; }
            int8_t *couplings = makeCouplings();
            BlobWriter blob = blobBegin();
            size_t mac = 0;
            printf("%s", makeSweepMIL(h, w, enc, couplings, &blob, &mac).UTF8String);
            fprintf(stderr, "mac_per_call=%zu weight_bytes=%zu\n", mac, blob.payloadBytes);
            free(couplings);
            return 0;
        }
        double budgetMS = strtod(argv[2], NULL);
        if (budgetMS <= 0) { fprintf(stderr, "budget must be positive\n"); return 2; }
        if (strcmp(argv[1], "roofline") == 0) return runRoofline(budgetMS);
        if (strcmp(argv[1], "sweep") == 0) {
            Encoding enc;
            if (argc < 5 || !parseEncoding(argv[3], &enc)) { fprintf(stderr, "sweep BUDGET_MS ENCODING HxW...\n"); return 2; }
            return runSweeps(budgetMS, enc, argc - 4, argv + 4);
        }
        fprintf(stderr, "unknown command %s\n", argv[1]);
        return 2;
    } @catch (NSException *exception) {
        fprintf(stderr, "probe_failed=%s\n", exception.reason.UTF8String);
        return 1;
    }
}
