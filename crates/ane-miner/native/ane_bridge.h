#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct { uint64_t staging_us; uint64_t dispatch_us; } QuipAneTimes;

// Arrays remain valid for the supplied counts throughout each blocking call.
// Errors are zero-terminated when error_capacity is nonzero. Handles are confined
// to the owning thread. State and requests remain owned until destroy.
int32_t quip_ane_create(size_t channels, const size_t *lengths, size_t tile_count,
    size_t sweeps, const int8_t *weights, size_t weight_count,
    const int8_t *fields, size_t field_count, void **program,
    char *error, size_t error_capacity);
int32_t quip_ane_reset(void *program, const int8_t *spins, size_t count,
    char *error, size_t error_capacity);
// Thresholds are sweep-major, then channel-major, with 128 lanes per channel.
// 255 skips a sweep position; 0..63 are the exact MSA integer thresholds.
int32_t quip_ane_evaluate(void *program, const uint8_t *thresholds, size_t count,
    QuipAneTimes *times, char *error, size_t error_capacity);
int32_t quip_ane_read(void *program, int8_t *output, size_t count,
    char *error, size_t error_capacity);
// Consumes the handle even when unloading fails. Never call twice on a handle.
int32_t quip_ane_destroy(void *program, char *error, size_t error_capacity);
uint32_t quip_ane_parent_pid(void);
