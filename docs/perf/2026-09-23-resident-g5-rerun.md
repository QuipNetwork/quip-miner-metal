# Gate G5 rerun and a production measurement, 2026-09-23

G5 still misses at commit `66ff170`. The whole-system rate is 2,795.56 nonces
per second against a gate of 4,615. This follows the [first
report](2026-09-23-resident-g5.md), which records 1,058 nonces per second at
`e1adb01`.

## Result

| gate | measured | pass | result |
|---|---|---|---|
| G5 at `66ff170` | 2,795.56 nonces per second | at least 4,615 | missed |

| run | test type | sweeps | timed interval | nonces per second | CPU ms per job | load average, start to end |
|---|---|---:|---:|---:|---:|---|
| `one-gate-32` | one gate | 32 | 60 s | 4,070 | 0.852 | 5.33 to 8.85 |
| `one-gate-14336` | one gate, then full-budget work | 14,336 | 120 s | 3,291 | 1.097 | 8.85 to 14.74 |
| `system` | whole system | 14,336 | 600 s | 2,795.56 | 1.142 | 14.74 to 19.13 across both streams |
| `system`, second stream | whole system, same sampler | 14,336 | 60 s | 2,848.6 | 1.155 | same measurement across both streams |

The progress notes use the last timed rate for both one-gate runs and the
second stream. Their summaries include draining and report 4,069.34, 3,273.17,
and 2,822.43 nonces per second. The first system row uses the summary rate
with drain time. Load averages use the first value in each `uptime` line.

All runs use 64 reads and run seed 20260923. The `one-gate-32` run meets the
one-gate definition in `AGENTS.md`: no later gate evaluates and the job ends
at 32 sweeps. The `one-gate-14336` run has stage `[32]` and keep 1 in 2,000.
Kept jobs continue to the full budget, so this run includes work beyond the
gate.

The first system stream returns 1,680,133 results, zero errors, and three
nonces at or below the chain target. The second returns 171,128 results, zero
errors, and no target hits. Neither one-gate run reaches the target.

### S4 after fix round 1

| round | pool, jobs per second | batch path, jobs per second | rate ratio | decode ratio |
|---|---:|---:|---:|---:|
| 1 | 5,154.92 | 4,583.61 | 1.1246 | 1.3365 |
| 2 | 4,974.87 | 4,692.86 | 1.0601 | 0.5782 |
| 3 | 5,470.76 | 4,417.88 | 1.2383 | 0.9779 |

S4 includes preparation and controller admission. Each round measures one
gate, with 32 sweeps, 64 reads, and no warmup. Load average rises from 4.64 to
5.73. All rate ratios pass the 0.95 gate. The first decode ratio misses the
separate 1.2 guard.

Fix round 2 caps decode leaves at 32 reads. Across ten S4 runs, nine pass
throughput and seven pass decode. The ruling closes the original decode
regression, from 123 microseconds per job to 20–60. Decode times still vary.
The final review must check that variance. G5 remains the binding gate.

## Changes since the first report

Commit `4d4beb0` computes one beta range per job and adds four preparation
workers. S4 now times that work too. Commit `dd82865` removes the shared
receiver lock. It also speeds unit-coefficient preparation and cuts decode
cost. Commit `66ff170` adds the decode leaf cap.

Commit `a1db19f` makes the resident runner the always-on MSA stream path. Jobs
whose coefficients are not whole units run once at full budget through the
batch path after resident slots drain.

## Rate while the controller settles

Timing starts with a fresh controller, without a warmup. Rates vary across the
run, as the 10-second windows show.

| interval | range of 10-second rates, nonces per second | total rate at interval end |
|---|---:|---:|
| 0 to 60 s | 1,856.7 to 2,950.4 | 2,502.5 |
| 60 to 300 s | 2,577.6 to 3,156.1 | 2,774.4 |
| 300 to 600 s | 2,059.9 to 3,324.3 | 2,799.7 |
| second stream, 0 to 60 s | 2,690.5 to 3,041.6 | 2,848.6 |

After the first stream's 600-second interval and drain, the second stream
reuses the saved controller. These windows show no clear time when the rate
settles. The logs include audit checks that loosen and restore cutoffs during
the first stream.

## Current limit in the test runner

The GPU now limits the test runner. The fix-round-1 profile measures a
40-second one-gate run at 4,019 nonces per second. Load average rises from
7.99 to 9.11. The progress notes report GPU busy time of 83 percent at 206
microseconds per probe job, with the runner thread 49 percent busy. Those
figures imply a probe-only ceiling of about 4,860 jobs per second under those
device conditions. They do not predict whole-system throughput.

Preparation workers are 73 to 86 percent busy. Coefficient filling accounts
for about 72 percent of preparation. The profile no longer shows the
beta-range calculation, and read decoding does not limit this run.

Kept full-budget jobs also consume device time. The saved CSV selects jobs
with at least 10 milliseconds of device time. Across both streams, its 2,345
rows account for 174.360276 seconds of device time. This is about 74.354
milliseconds per selected job, with one selected job per 789 results. The CSV
has no completed-sweep field, so the filter uses device time. It cannot prove
that every selected job reaches the full budget.

### Nonces at or below the chain target

The chain target for these tests is −14,625,068 milli.

| nonce | seed | best, milli |
|---:|---|---:|
| 21557 | `0e7488551277047666a77353a7caa71aa859800628ad256f1b33989bff577282` | −14,636,000 |
| 84198 | `518012ae2ffbc2051191c607390bf2db7378a9a608ad8c984356329ec6876bc8` | −14,626,000 |
| 1419019 | `e7e113197b9c6487e45f2d686d878d455b7ce0ea90cef22063ca67e8286b045d` | −14,632,000 |

## Device time per job

G5 and S4 both use command-buffer end time minus start time, in integer
microseconds. G5 divides each command's time among participating slots. S4
sums whole commands. The old division loses less than one microsecond per job
in the one-gate case. It cannot explain the historical 250 versus 156
microseconds per job.

The new split shares the spare microseconds, so slot totals equal command
totals exactly. The saved S4 rounds show equal totals of 876,502, 1,982,518,
and 902,856 microseconds. The totals now agree, but the old runs still have
different costs per job. Missing pool counts and device clock records prevent a fuller comparison. The first report gives no GPU ceiling until these
figures agree. The new probe estimate uses the later profile, not the
historical gap.

## Production measurement

The production limit is the coordinator's per-result validation. The
measurement uses `a1db19f` in `~/quip-data-3`, with coordinator `0.3.5-rc9`,
on 2026-09-23 at about 17:55 local time. The 120-second measurement spans load
averages 18.63 to 18.91.

Production progress shows about 186 jobs per second near 17:55. Across the
saved progress interval, 59,000 jobs over about 325.53 seconds give about 181
jobs per second. The rates use different time spans and methods.

The coordinator uses about 3 to 6.8 CPU cores. The miner uses about 0.5 to 1.5
cores. GPU load bursts between 0 and 80 percent. The script samples CPU use
for the coordinator first, then the miner, and reads `Device Utilization %`
every five seconds.

The coordinator's 10-second sample assigns 24,394 top-of-stack samples to
`quip_coordinator::validate::validate_results`. The listed non-waiting frames
total 27,537 samples, giving validation an 88.6 percent share. The next
largest busy frame, `validate_spins`, has 1,402 samples. Model generation's
`draw_ising_milli` has 29.

This count uses the sample's “Sort by top of stack” section, which lists
counts of at least five. It excludes condition waits, mutex waits, semaphore
waits, `swtch_pri`, `kevent`, `__recvfrom`, and `cthread_yield`. It measures
the listed busy samples, not total CPU time.

The miner sample shows waiting threads. Its largest frame is
`__psynch_cvwait`, with 140,375 samples. The process samples and load data
show that the coordinator limits the rate as it checks results. Neither the
miner nor model generation limits this run. Generated streams alone do not
raise this rate while the coordinator rescores all 64 reads of every result.

## Data

`docs/perf/data/2026-09-23-resident-g5-rerun/`:

- `one-gate-32.log.gz`, `one-gate-14336.log.gz`, `system.log.gz`: test
  output with 10-second rate lines and summaries.
- `one-gate-32.uptime`, `one-gate-14336.uptime`, `system.uptime`: load
  averages before and after each run.
- `system-kept.csv.gz`: jobs with at least 10 milliseconds of device time,
  columns `nonce,seed,best,reads,device_us,ok`.
- `profile-runner.txt.gz`: the one-gate runner call tree after fix round 1.
- `profile.uptime`: load averages before and after that profile run.
- `s4.log.gz`: the controller's S4 rounds after fix round 1.
- `production/measure.log`: load samples for each process and the GPU.
- `production/coordinator.sample.txt.gz`, `production/miner.sample.txt.gz`:
  10-second call trees from the production processes.
- `production/progress.log`: sampled production progress lines.
- `production/measure.sh`: the commands that collect the production measurements.
