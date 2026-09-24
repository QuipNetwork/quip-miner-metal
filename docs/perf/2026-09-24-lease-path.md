# Lease path rate and local generation, 2026-09-24

L1 missed: lease-wide reached 0.186 of the direct rate. The gate was 0.95. The median rates were 550.48 salts/s for lease-wide and 2962.29 jobs/s for direct. The combined router checks eligibility for every job, while the direct arm bypasses the router.

## Runs

Each run lasted 300 s as one process. The direct arm reports jobs/s and CPU ms per job. Lease arms report salts/s and CPU ms per salt. The load values are the 1-, 5-, and 15-minute averages, in that order.

Both arms used the default adaptive configuration as whole-system runs. There was no deliberate warm-up interval. The direct timer starts before stream setup and producer dispatch. The lease timer starts after the coordinator sends the initial leases. Direct logs report per-window rates. Direct-1 began at 3416.6 jobs/s and reported 2269.4 jobs/s at 120 s. Lease logs report cumulative rates. Lease-wide-1 stayed near 550 salts/s from 10 s through 301 s.

| run | rate | CPU ms per salt or job | results | verified winners | leases finished | load at start | load at end | used |
|---|---:|---:|---:|---:|---:|---|---|---|
| direct-1 | 2962.29 jobs/s | 1.063 | - | - | - | 16.30 11.07 7.46 | 7.42 9.66 7.97 | yes |
| lease-42-1 | 521.74 salts/s | 2.750 | 0 | 0 | 16 | 7.42 9.66 7.97 | 6.05 7.75 7.63 | yes |
| lease-wide-1 | 550.48 salts/s | 2.625 | 0 | 0 | 15 | 6.05 7.75 7.63 | 6.00 6.89 7.30 | yes |
| lease-42-2 | 554.31 salts/s | 2.589 | 0 | 0 | 16 | 6.00 6.89 7.30 | 5.43 6.30 6.89 | yes |
| lease-wide-2 | 548.27 salts/s | 2.625 | 0 | 0 | 16 | 5.43 6.30 6.89 | 4.56 5.48 6.35 | yes |
| direct-2 | 2683.82 jobs/s | 1.076 | - | - | - | 4.56 5.48 6.35 | 38.65 15.96 10.16 | yes |
| lease-wide-3 | 297.32 salts/s | 4.276 | 0 | 0 | 12 | 38.65 15.96 10.16 | 52.64 37.79 22.10 | no: external load |
| direct-3 | 1700.26 jobs/s | 1.110 | - | - | - | 52.64 37.79 22.10 | 23.29 41.04 29.52 | no: external load |
| lease-42-3 | 520.48 salts/s | 2.758 | 0 | 0 | 11 | 23.29 41.04 29.52 | 5.77 18.81 22.62 | yes |
| lease-wide-3r | 573.76 salts/s | 2.537 | 0 | 0 | 16 | 4.07 16.76 21.70 | 5.54 9.49 16.76 | yes |
| direct-3r | 3309.27 jobs/s | 1.060 | - | - | - | 5.54 9.49 16.76 | 7.62 8.78 14.27 | yes |

The external load spike began near the end of direct-2 and continued through lease-wide-3 and direct-3. The medians exclude lease-wide-3 and direct-3 because of that spike. Direct-2 ended at a 1-minute load of 38.65. Lease-42-3 started at 23.29 while the spike decayed. Both runs remain included because their rates fall within the clean-run ranges. Excluding direct-2 would move the direct median from 2962.29 to 3135.78 and would not change any verdict.

The timed lease runs returned 0 results and 0 of 0 verified winners. Every direct `errors:` line reads `errors: 0`. Lease runs logged no errors.

## Medians and gates

| path | used rates | median | range |
|---|---|---:|---:|
| direct | 2683.82, 2962.29, 3309.27 jobs/s | 2962.29 jobs/s | 2683.82-3309.27 |
| lease-42 | 520.48, 521.74, 554.31 salts/s | 521.74 salts/s | 520.48-554.31 |
| lease-wide | 548.27, 550.48, 573.76 salts/s | 550.48 salts/s | 548.27-573.76 |

L1 is 550.48 / 2962.29 = 0.186, below the required 0.95.

L2 is 550.48 / 521.74 = 1.055. The projection threshold was greater than 1.1, so L2 is informational. The ratio is between 1.05 and 1.1. The wider window adds about 5.5 percent, so width is not the main limit.

L3 holds: no run logged errors. The nine timed runs found no winner at this target. The 120 s profile run found 1 winner, and it verified. All 7 Task 3 device tests pass and cover winner verification with `verify_lease_result`.

## Profile

The plan called for the `sample <pid> 30` command on the miner during a 120 s lease-wide run. The first try sampled the Apple Neural Engine (ANE) worker child, `quip-metal-msa --ane-worker`, instead of the session process. The saved `profile-lease-wide.ane-child.sample.txt.gz` was not used. The second try sampled both processes. The session process, pid 50892, used 144.2 percent CPU. The ANE worker, pid 51063, used 44.7 percent CPU. The sampled run reached 547.18 salts/s at 2.567 CPU ms per salt and found 1 winner, which verified.

In pid 50892, thread `quip-sampler` runs `combined::CombinedSampler::sample_stream` through `combined::Router::run`. Of its 22232 thread samples, 22002 fell inside `combined::eligibility` at `src/combined.rs:256`. This function inserts every job edge into a `HashSet<(usize, usize)>`. Each job has about 41k edges on Advantage2. The function also recomputes node degrees for every job when the miner runs the ANE engine and every coefficient is a unit value. One thread performs this work, which serializes every job handled by the router.

The session expander and preparation figures are top-of-stack counts summed across the process:

| work | samples |
|---|---:|
| `ChaCha8Rng::next_u32` | 1264 |
| `draw_ising_milli` | 168 |
| `convert_milli` | 110 |
| `fill_h_j_matching` | 878 |
| `Preparer::prepare` | 858 |
| `PreparedSchedule::new` | 1314 |

The direct arm builds `MetalSampler` and calls `sample_stream` in `tests/probe_screen.rs:682`. It does not use the combined router or the Apple Neural Engine. The lease arm runs the real `quip-metal-msa` binary with `CombinedSampler`. L1 compares the binary through the router with the Metal sampler without the router. The profile points to the router eligibility check, not the lease expander, as the cause of the gap.

## Local-generation decision

Local generation stays unbuilt. L1 missed, but the profile does not put the session expander or preparation conversion first. The plan directs the work to stop after reporting the profile when the miss has another cause.

Follow-up bead `quip-miner-metal-zse` is a P1 bug. It calls for computing the structural part of eligibility, including bounds, degrees, and duplicate edges, once per edge set. Reuse that result while the edges remain unchanged. Plain jobs in the shipped combined miner pay the same per-job cost. This also affects production without leases.

## Production note

Under leases, the coordinator receives one `Result` per winning salt instead of one per job. A protocol-2 coordinator exists as quip-miner merge request !290, which is open and not merged. It sizes each lease at 4 s of the miner's smoothed salt rate, never below one stream_width, and at most 2^20 salts. Its first lease uses 4 x stream_width. This report did not run that coordinator.

## Data

Files in `docs/perf/data/2026-09-24-lease-path/`. The direct arm's per-nonce CSV output, 290 MB in total, is not kept, because no gate reads it. Every rate comes from the logs.

- `direct-1.log`, `direct-1.uptime-end`, `direct-1.uptime-start`
- `direct-2.log`, `direct-2.uptime-end`, `direct-2.uptime-start`
- `direct-3.log`, `direct-3.uptime-end`, `direct-3.uptime-start`
- `direct-3r.log`, `direct-3r.uptime-end`, `direct-3r.uptime-start`
- `lease-42-1.log`, `lease-42-1.uptime-end`, `lease-42-1.uptime-start`
- `lease-42-2.log`, `lease-42-2.uptime-end`, `lease-42-2.uptime-start`
- `lease-42-3.log`, `lease-42-3.uptime-end`, `lease-42-3.uptime-start`
- `lease-wide-1.log`, `lease-wide-1.uptime-end`, `lease-wide-1.uptime-start`
- `lease-wide-2.log`, `lease-wide-2.uptime-end`, `lease-wide-2.uptime-start`
- `lease-wide-3.log`, `lease-wide-3.uptime-end`, `lease-wide-3.uptime-start`
- `lease-wide-3r.log`, `lease-wide-3r.uptime-end`, `lease-wide-3r.uptime-start`
- `profile-lease-wide.50892.sample.txt.gz`
- `profile-lease-wide.51063.sample.txt.gz`
- `profile-lease-wide.ane-child.sample.txt.gz`
- `profile-lease-wide.log`
- `profile.processes.txt`
- `profile.uptime-end`
- `profile.uptime-start`
