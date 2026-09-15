#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint64_t staging_us;
    uint64_t dispatch_us;
} QuipAneTimes;

// All arrays must be valid for the supplied counts for the duration of the call.
// Error buffers must be writable for error_capacity bytes. Nonzero status means
// failure; when capacity is nonzero, the error is always zero-terminated.
int32_t quip_ane_create(
    size_t input_channels, size_t output_channels,
    const int8_t *weights, size_t weight_count,
    const int8_t *fields, size_t field_count,
    void **program, char *error, size_t error_capacity);

// Both handles must be live and on the same owning thread. Rebinds only the
// neighbor input; each program keeps its own output shape and other surfaces.
int32_t quip_ane_share_input(void *program, void *source, char *error, size_t error_capacity);

#define QUIP_ANE_INPUT_FULL 0
#define QUIP_ANE_INPUT_ROWS 1

// program must be a live handle returned by create, used on its owning thread.
// neighbors always has the full input shape. FULL uploads all entries and
// requires no changed rows. ROWS uploads only the listed 128-lane rows, after
// a successful full upload to the shared input. Unlisted entries are not read.
int32_t quip_ane_evaluate(
    void *program,
    const int8_t *neighbors, size_t neighbor_count,
    uint32_t input_mode, const size_t *changed_rows, size_t changed_row_count,
    const int8_t *spins, size_t spin_count,
    const uint8_t *thresholds, size_t threshold_count,
    int8_t *output, size_t output_count,
    QuipAneTimes *times, char *error, size_t error_capacity);

// Consumes the handle even when unloading fails. Never call twice on a handle.
int32_t quip_ane_destroy(void *program, char *error, size_t error_capacity);
uint32_t quip_ane_parent_pid(void);
