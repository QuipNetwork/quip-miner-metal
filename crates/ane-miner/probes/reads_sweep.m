// Bead quip-miner-metal-fjo.6: how does a sweep's cost move with the read
// count?
//
// The program pins 128 reads. crates/ane-miner/src/graph.rs sets LANES to
// 128 and ane_bridge.m's shape() hardcodes the last tensor dimension, so
// every tensor in the program is [1, channels, 1, 128]. That count came
// from the CUDA implementation and has never been measured on this engine.
//
// Reads are independent replicas. Each lane anneals its own copy of the
// problem against its own thresholds, and no operation in the sweep mixes
// lanes. So the arithmetic falls proportionally with the count, and the
// question is whether the engine follows it or pads to a fixed width.
//
// This matters for reaching about 16,384 sweeps per job. At the
// four-colouring's 0.8818 ms per sweep that is 14.4 s, and a four-fold cut
// would bring it near 3.6 s.
//
// The probe builds its own program text, because the lane count is not a
// parameter of the production builder, and its own dense fp16 weight blob,
// because production moved to the sparse encoding on 2026-09-18 and this
// probe measures the dense program its report describes. Everything else
// follows ane_bridge.m's create path, and makeSurface and stageSurface come
// from that file unchanged. Production entry points are untouched.
//
// This measures dispatch cost only. Fewer reads means fewer parallel
// replicas, so time to a valid solution can worsen even when time per sweep
// improves. That half belongs with quip-miner-metal-fjo.2.
//
// Usage: reads-sweep [call-count] [lanes...]
#include "../native/ane_bridge.m"
#include <dlfcn.h>
#include <stdlib.h>
#include <string.h>

static const size_t kChannels = 4608;
// The Advantage2 four-colouring, src/topology.rs advantage2_color.
static const size_t kLengths[4] = {1148, 1145, 1145, 1139};
static const size_t kTiles = 4;
static const size_t kSweeps = 1;
static const size_t kDefaultCalls = 200;

// The dense fp16 blob the production bridge wrote before the sparse
// encoding: one fp16 chunk per tile behind a 64-byte record.
static NSData *makeDenseWeightBlob(const int8_t *weights, size_t channels, const size_t *lengths, size_t tiles, size_t count) {
    NSMutableData *blob = [NSMutableData dataWithLength:64 + 64 * tiles + count * sizeof(_Float16)];
    uint8_t *bytes = blob.mutableBytes;
    uint32_t chunkCount = CFSwapInt32HostToLittle((uint32_t)tiles);
    memcpy(bytes, &chunkCount, sizeof(chunkCount));
    bytes[4] = 2;
    size_t offset = 64, weightOffset = 0;
    for (size_t tile = 0; tile < tiles; ++tile) {
        size_t elements = ((lengths[tile] + 31) / 32 * 32) * channels;
        writeBlobRecord(bytes + offset, kBlobFP16, elements * sizeof(_Float16), offset + 64);
        for (size_t i = 0; i < elements; ++i) {
            _Float16 value = (_Float16)weights[weightOffset + i];
            uint16_t bits;
            memcpy(&bits, &value, sizeof(bits));
            bits = CFSwapInt16HostToLittle(bits);
            memcpy(bytes + offset + 64 + i * sizeof(bits), &bits, sizeof(bits));
        }
        weightOffset += elements;
        offset += 64 + elements * sizeof(_Float16);
    }
    return blob;
}

// shape() and slice() from ane_bridge.m with the lane count lifted out of
// the literal. Named apart from that file's statics, which this includes.
static NSString *shapeL(size_t channels, size_t lanes) {
    return [NSString stringWithFormat:@"tensor<fp16, [1, %zu, 1, %zu]>", channels, lanes];
}

static void sliceL(NSMutableString *mil, NSString *name, NSString *source,
    size_t begin, size_t count, size_t lanes) {
    [mil appendFormat:@"    tensor<int32, [4]> %@begin = const()[name=string(\"%@begin\"), val=tensor<int32, [4]>([0,%zu,0,0])];\n", name, name, begin];
    [mil appendFormat:@"    tensor<int32, [4]> %@size = const()[name=string(\"%@size\"), val=tensor<int32, [4]>([1,%zu,1,%zu])];\n", name, name, count, lanes];
    [mil appendFormat:@"    %@ %@ = slice_by_size(x=%@, begin=%@begin, size=%@size)[name=string(\"%@\")];\n", shapeL(count, lanes), name, source, name, name, name];
}

// makeMIL from ane_bridge.m:142 with the lane count parameterised. The
// operator chain is copied rather than adapted, so a difference between
// this and production is a difference in lanes alone.
static NSString *makeMILL(size_t channels, const size_t *lengths, size_t tiles,
    size_t sweeps, const int8_t *fields, size_t lanes) {
    NSMutableString *mil = [NSMutableString stringWithFormat:
        @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, {\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, {\"coremltools-version\", \"9.0\"}, {\"quip-ane-msa\", \"%@\"}})]\n{\n  func main<ios18>(%@ a_state", NSUUID.UUID.UUIDString, shapeL(channels, lanes)];
    for (size_t sweep = 0; sweep < sweeps; ++sweep)
        [mil appendFormat:@", %@ t%zu", shapeL(channels, lanes), sweep];
    [mil appendString:@") {\n"
        "    string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"
        "    tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"
        "    tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"
        "    tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"
        "    int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"
        "    int32 axis = const()[name=string(\"axis\"), val=int32(1)];\n"
        "    bool interleave = const()[name=string(\"interleave\"), val=bool(false)];\n"
        "    fp16 zero = const()[name=string(\"zero\"), val=fp16(0.0)];\n"
        "    fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];\n"
        "    fp16 minusTwo = const()[name=string(\"minusTwo\"), val=fp16(-2.0)];\n"];
    size_t begin = 0, blobOffset = 64;
    for (size_t tile = 0; tile < tiles; ++tile) {
        size_t count = lengths[tile], padded = (count + 31) / 32 * 32;
        [mil appendFormat:@"    tensor<fp16, [%zu, %zu, 1, 1]> w%zu = const()[name=string(\"w%zu\"), val=tensor<fp16, [%zu, %zu, 1, 1]>(BLOBFILE(path=string(\"@model_path/weights/weight_data.bin\"), offset=uint64(%zu)))];\n", padded, channels, tile, tile, padded, channels, blobOffset];
        blobOffset += 64 + padded * channels * sizeof(_Float16);
        NSMutableArray *values = [NSMutableArray new];
        for (size_t row = 0; row < count; ++row)
            [values addObject:[NSString stringWithFormat:@"%d.0", (int)fields[begin + row]]];
        [mil appendFormat:@"    tensor<fp16, [%zu]> hFlat%zu = const()[name=string(\"hFlat%zu\"), val=tensor<fp16, [%zu]>([%@])];\n", count, tile, tile, count, [values componentsJoinedByString:@","]];
        [mil appendFormat:@"    tensor<int32, [4]> hShape%zu = const()[name=string(\"hShape%zu\"), val=tensor<int32, [4]>([1,%zu,1,1])];\n", tile, tile, count];
        [mil appendFormat:@"    tensor<fp16, [1,%zu,1,1]> h%zu = reshape(x=hFlat%zu, shape=hShape%zu)[name=string(\"h%zu\")];\n", count, tile, tile, tile, tile];
        begin += count;
    }
    NSString *state = @"a_state";
    for (size_t sweep = 0; sweep < sweeps; ++sweep) {
        begin = 0;
        for (size_t tile = 0; tile < tiles; ++tile) {
            size_t count = lengths[tile], padded = (count + 31) / 32 * 32;
            NSString *prefix = [NSString stringWithFormat:@"s%zuc%zu", sweep, tile];
            NSString *own = [prefix stringByAppendingString:@"own"];
            NSString *threshold = [prefix stringByAppendingString:@"threshold"];
            NSString *raw = [prefix stringByAppendingString:@"raw"];
            NSString *js = [prefix stringByAppendingString:@"js"];
            sliceL(mil, own, state, begin, count, lanes);
            sliceL(mil, threshold, [NSString stringWithFormat:@"t%zu", sweep], begin, count, lanes);
            [mil appendFormat:@"    %@ %@ = conv(dilations=dl, groups=gr, pad=pd, pad_type=pt, strides=st, weight=w%zu, x=%@)[name=string(\"%@\")];\n", shapeL(padded, lanes), raw, tile, state, raw];
            sliceL(mil, js, raw, 0, count, lanes);
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
                [mil appendFormat:@"    %@ %@%@ = %@[name=string(\"%@%@\")];\n", shapeL(count, lanes), prefix, op[0], op[1], prefix, op[0]];
            NSMutableArray *parts = [NSMutableArray new];
            if (begin > 0) {
                NSString *head = [prefix stringByAppendingString:@"head"];
                sliceL(mil, head, state, 0, begin, lanes);
                [parts addObject:head];
            }
            [parts addObject:[prefix stringByAppendingString:@"updated"]];
            if (begin + count < channels) {
                NSString *tail = [prefix stringByAppendingString:@"tail"];
                sliceL(mil, tail, state, begin + count, channels - begin - count, lanes);
                [parts addObject:tail];
            }
            if (parts.count == 1) {
                state = parts[0];
            } else {
                state = [prefix stringByAppendingString:@"state"];
                [mil appendFormat:@"    %@ %@ = concat(values=(%@), axis=axis, interleave=interleave)[name=string(\"%@\")];\n", shapeL(channels, lanes), state, [parts componentsJoinedByString:@", "], state];
            }
            begin += count;
        }
    }
    [mil appendFormat:@"  } -> (%@);\n}\n", state];
    return mil;
}

// Compiles one program at the given lane count and times `calls`
// evaluations. Returns NO on any failure, which a lane count the engine
// rejects will produce, and that rejection is itself a result.
static BOOL timeLanes(size_t lanes, size_t calls, double *msPerSweep, double *compileMS,
                      uint64_t *loopStartUS, uint64_t *loopEndUS) {
    @autoreleasepool {
        NSError *error = nil;
        size_t weightCount = 0;
        for (size_t i = 0; i < kTiles; ++i) weightCount += ((kLengths[i] + 31) / 32 * 32) * kChannels;
        int8_t *weights = malloc(weightCount);
        int8_t *fields = calloc(kChannels, 1);
        if (weights == NULL || fields == NULL) { fprintf(stderr, "allocation failed\n"); return NO; }
        srandom(123);
        for (size_t i = 0; i < weightCount; ++i) weights[i] = (random() & 1) ? (int8_t)1 : (int8_t)-1;

        // ane_bridge.m dlopens the framework inside quip_ane_create, which
        // this probe does not call, so the classes would otherwise be nil
        // and every lane count would report a descriptor failure.
        static void *framework;
        static dispatch_once_t once;
        dispatch_once(&once, ^{
            framework = dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
        });
        if (framework == NULL) { fprintf(stderr, "AppleNeuralEngine framework unavailable\n"); return NO; }
        Class descriptorClass = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class modelClass = NSClassFromString(@"_ANEInMemoryModel");
        Class surfaceClass = NSClassFromString(@"_ANEIOSurfaceObject");
        Class requestClass = NSClassFromString(@"_ANERequest");
        NSData *plist = [NSPropertyListSerialization dataWithPropertyList:@{} format:NSPropertyListXMLFormat_v1_0 options:0 error:&error];
        if (plist == nil) { fprintf(stderr, "plist failed\n"); return NO; }

        NSData *mil = [makeMILL(kChannels, kLengths, kTiles, kSweeps, fields, lanes)
            dataUsingEncoding:NSUTF8StringEncoding];
        id descriptor = [[descriptorClass alloc] initWithNetworkText:mil weights:@{} optionsPlist:plist isMILModel:YES];
        if (descriptor == nil) { fprintf(stderr, "lanes=%zu descriptor failed\n", lanes); return NO; }
        id model = [modelClass inMemoryModelWithDescriptor:descriptor];
        if (model == nil) { fprintf(stderr, "lanes=%zu model failed\n", lanes); return NO; }

        // The directory must be named for the model's own identifier. The
        // compiler resolves the program text's "@model_path" against it, so
        // any other name leaves weight_data.bin unfindable and the compile
        // fails with InvalidMILProgram even though the text is correct.
        NSString *directory = [NSTemporaryDirectory()
            stringByAppendingPathComponent:[model hexStringIdentifier]];
        if (mkdir(directory.fileSystemRepresentation, 0700) != 0) { fprintf(stderr, "staging dir failed\n"); return NO; }
        NSString *weightDirectory = [directory stringByAppendingPathComponent:@"weights"];
        [NSFileManager.defaultManager createDirectoryAtPath:weightDirectory withIntermediateDirectories:NO attributes:nil error:&error];
        [mil writeToFile:[directory stringByAppendingPathComponent:@"model.mil"] options:NSDataWritingAtomic error:&error];
        NSData *blob = makeDenseWeightBlob(weights, kChannels, kLengths, kTiles, weightCount);
        [blob writeToFile:[weightDirectory stringByAppendingPathComponent:@"weight_data.bin"] options:NSDataWritingAtomic error:&error];
        free(weights);
        free(fields);

        uint64_t compileStart = monotonicUS();
        if (![model compileWithQoS:21 options:@{} error:&error]) {
            fprintf(stderr, "lanes=%zu compile failed: %s\n", lanes, error.description.UTF8String);
            [NSFileManager.defaultManager removeItemAtPath:directory error:NULL];
            return NO;
        }
        *compileMS = (monotonicUS() - compileStart) / 1000.0;
        if (![model loadWithQoS:21 options:@{} error:&error]) {
            fprintf(stderr, "lanes=%zu load failed: %s\n", lanes, error.description.UTF8String);
            [NSFileManager.defaultManager removeItemAtPath:directory error:NULL];
            return NO;
        }
        [NSFileManager.defaultManager removeItemAtPath:directory error:NULL];

        size_t inputElements = kChannels * lanes;
        IOSurfaceRef surfaces[3] = {0};
        NSMutableArray *wrappers = [NSMutableArray new];
        for (size_t i = 0; i < kSweeps + 2; ++i) {
            surfaces[i] = makeSurface(inputElements);
            if (surfaces[i] == NULL) { fprintf(stderr, "surface failed\n"); return NO; }
            id wrapper = [surfaceClass objectWithIOSurface:surfaces[i]];
            if (wrapper == nil) { fprintf(stderr, "wrap failed\n"); return NO; }
            [wrappers addObject:wrapper];
        }
        // Stage a spin state and thresholds so the dispatch does real work.
        int8_t *spins = malloc(inputElements);
        uint8_t *thresholds = malloc(inputElements);
        if (spins == NULL || thresholds == NULL) { fprintf(stderr, "stage alloc failed\n"); return NO; }
        srandom(456);
        for (size_t i = 0; i < inputElements; ++i) spins[i] = (random() & 1) ? (int8_t)1 : (int8_t)-1;
        srandom(789);
        for (size_t i = 0; i < inputElements; ++i) thresholds[i] = (uint8_t)(random() % 64);
        NSString *stagingError = nil;
        if (!stageSurface(surfaces[0], spins, inputElements, NO, &stagingError) ||
            !stageSurface(surfaces[2], thresholds, inputElements, YES, &stagingError)) {
            fprintf(stderr, "staging failed: %s\n", stagingError.UTF8String);
            return NO;
        }
        free(spins);
        free(thresholds);

        id request = [requestClass requestWithInputs:@[wrappers[0], wrappers[2]] inputIndices:@[@0, @1]
            outputs:@[wrappers[1]] outputIndices:@[@0]
            weightsBuffer:nil perfStats:nil procedureIndex:@0];
        if (request == nil) { fprintf(stderr, "request failed\n"); return NO; }

        uint64_t loopStart = monotonicUS();
        for (size_t call = 0; call < calls; ++call) {
            if (![model evaluateWithQoS:21 options:@{} request:request error:&error]) {
                fprintf(stderr, "lanes=%zu evaluate failed: %s\n", lanes, error.description.UTF8String);
                return NO;
            }
        }
        uint64_t loopEnd = monotonicUS();
        double loopMS = (loopEnd - loopStart) / 1000.0;
        *loopStartUS = loopStart;
        *loopEndUS = loopEnd;
        *msPerSweep = loopMS / (double)(calls * kSweeps);

        [model unloadWithQoS:21 error:&error];
        for (size_t i = 0; i < kSweeps + 2; ++i) if (surfaces[i]) CFRelease(surfaces[i]);
        return YES;
    }
}

int main(int argc, const char **argv) {
    setbuf(stdout, NULL);
    // "dump N" prints the program text at N lanes, for diffing against the
    // production builder at 128, where the two must agree apart from the
    // build identifier.
    if (argc >= 2 && strcmp(argv[1], "dump") == 0) {
        size_t lanes = argc >= 3 ? (size_t)strtoul(argv[2], NULL, 10) : 128;
        int8_t *fields = calloc(kChannels, 1);
        printf("%s", makeMILL(kChannels, kLengths, kTiles, kSweeps, fields, lanes).UTF8String);
        return 0;
    }
    size_t calls = argc >= 2 ? (size_t)strtoul(argv[1], NULL, 10) : kDefaultCalls;
    if (calls == 0) { fprintf(stderr, "call count must be positive\n"); return 2; }
    size_t defaults[5] = {8, 16, 32, 64, 128};
    size_t laneList[16];
    size_t laneCount = 0;
    if (argc > 2) {
        for (int i = 2; i < argc && laneCount < 16; ++i)
            laneList[laneCount++] = (size_t)strtoul(argv[i], NULL, 10);
    } else {
        for (size_t i = 0; i < 5; ++i) laneList[laneCount++] = defaults[i];
    }

    @try {
        double baseline = 0;
        for (size_t i = 0; i < laneCount; ++i) {
            size_t lanes = laneList[i];
            double msPerSweep = 0, compileMS = 0;
            uint64_t loopStartUS = 0, loopEndUS = 0;
            if (!timeLanes(lanes, calls, &msPerSweep, &compileMS, &loopStartUS, &loopEndUS)) {
                printf("{\"lanes\":%zu,\"status\":\"rejected\"}\n", lanes);
                continue;
            }
            if (lanes == 128) baseline = msPerSweep;
            // Absolute loop bounds from monotonicUS() let concurrently launched
            // instances show how much of their loops overlapped.
            printf("{\"lanes\":%zu,\"calls\":%zu,\"compile_ms\":%.3f,\"ms_per_sweep\":%.4f,"
                "\"us_per_sweep_per_read\":%.4f,\"loop_start_us\":%llu,\"loop_end_us\":%llu}\n",
                lanes, calls, compileMS, msPerSweep, msPerSweep * 1000.0 / (double)lanes,
                (unsigned long long)loopStartUS, (unsigned long long)loopEndUS);
        }
        if (baseline > 0) printf("{\"baseline_128_ms_per_sweep\":%.4f}\n", baseline);
        return 0;
    } @catch (NSException *exception) {
        fprintf(stderr, "probe_failed=%s\n", exception.reason.UTF8String);
        return 1;
    }
}
