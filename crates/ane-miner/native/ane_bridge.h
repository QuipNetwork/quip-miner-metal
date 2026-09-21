#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct { uint64_t staging_us; uint64_t dispatch_us; } QuipAneTimes;

// Arrays remain valid for each call. Submit stages thresholds before returning;
// it retains no caller pointer while the dispatch continues asynchronously.
// Errors are zero-terminated when error_capacity is nonzero. Handles are confined
// to the owning thread. State and requests remain owned until destroy.
int32_t quip_ane_create(size_t channels, size_t lanes, const size_t *lengths, size_t tile_count,
    size_t sweeps, const int8_t *weights, size_t weight_count,
    const int8_t *fields, size_t field_count, void **program,
    char *error, size_t error_capacity);
int32_t quip_ane_reset(void *program, const int8_t *spins, size_t count,
    char *error, size_t error_capacity);
// Thresholds are sweep-major, then channel-major, with the compiled lane count per channel.
// 255 skips a sweep position; 0..63 are the exact MSA integer thresholds.
int32_t quip_ane_submit(void *program, const uint8_t *thresholds, size_t count,
    QuipAneTimes *times, char *error, size_t error_capacity);
int32_t quip_ane_finish(void *program, QuipAneTimes *times,
    char *error, size_t error_capacity);
int32_t quip_ane_evaluate(void *program, const uint8_t *thresholds, size_t count,
    QuipAneTimes *times, char *error, size_t error_capacity);
int32_t quip_ane_read(void *program, int8_t *output, size_t count,
    char *error, size_t error_capacity);
// Consumes the handle even when unloading fails. Never call twice on a handle.
int32_t quip_ane_destroy(void *program, char *error, size_t error_capacity);
uint32_t quip_ane_parent_pid(void);

// Converts thresholds to their exact IEEE 754 binary16 representation.
// The pointer arguments must each cover count non-overlapping elements.
void quip_ane_convert_thresholds(const uint8_t *thresholds, uint16_t *output_bits, size_t count);
