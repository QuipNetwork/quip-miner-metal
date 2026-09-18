# Read count against the testnet's own blocks, 2026-09-18

## Result

On the 60 most recent Aglais testnet blocks, regenerated exactly, and run
at 16,384 sweeps, one job matches the winning energy in fewer than half of
its tries at every read count from 32 to 256. What reads buy is a higher
chance per job, and that chance grows slower than the cost.

| Reads | Valid proof | Beats the winner | Median best energy, milli |
| ---: | ---: | ---: | ---: |
| 32 | 0.30 | 0.16 | -14,614,000 |
| 64 | 0.40 | 0.20 | -14,618,000 |
| 128 | 0.54 | 0.32 | -14,622,000 |
| 256 | 0.68 | 0.47 | -14,626,000 |

A valid proof needs a best energy below the block's target, and an equal
energy fails. Beating
the winner means at or below the energy that won that block. Each cell is
300 jobs: 60 blocks, 5 seeds each.

Weighted by the GPU's measured jobs per second at 16,384 sweeps, fewer
reads produce more valid proofs per second, because the GPU trades reads
for models almost linearly while the chance per job does not keep up.

| Reads | Jobs per second | Valid proofs per second | Winner-beating proofs per second |
| ---: | ---: | ---: | ---: |
| 32, default streaming | 14.3 to 15.5 | 4.2 to 4.6 | 2.3 to 2.5 |
| 32, one batch at a time | 20.6 | 6.1 | 3.3 |
| 64 | 12.2 to 12.8 | 4.9 to 5.2 | 2.5 to 2.6 |
| 128 | 6.4 to 7.3 | 3.5 to 3.9 | 2.0 to 2.3 |
| 256 | 3.4 | 2.3 | 1.6 |

The jobs per second come from `docs/perf/2026-09-18-gpu-reads-and-models.md`
and a 256-read run of the same harness, 20 jobs in 5.8 s, twice.

The ANE moves the other way. Its sweep at 32 reads costs 0.77 of its sweep
at 128, so 32 reads gives 0.089 jobs per second against 0.068, and 0.026
valid proofs per second against 0.037. The ANE keeps 128 reads.

The answer to the question asked, how many reads reach the energies the
network found, is that 256 reads at 16,384 sweeps matches the winner on
average, with a mean gap of 207 milli, and 128 reads falls 3,700 milli
short. Below 128 the gap opens to 9,100 and 13,200 milli. Reaching the
winner in one job is not what mining needs, though. Mining needs the first
valid proof in a round and the lowest energy among proofs in one chain
block, and the second table is the one that decides read count for that.

## What the testnet mines

The chain is `AGLS (Quip Testnet)`, chain id `quip_testnet`, at block
233,412 with qblock 3,250 as the latest when fetched. Every one of the 60
blocks ran on topology `cbec1eb4`: 4,577 nodes, 41,514 edges, fields fixed
at zero, and couplings in `{-1, +1}`. That is the Advantage2 System 1 graph
with one edge fewer than `tests/fixtures/advantage2-system1.edges`, and
without the fields the benchmark harness draws.

The difficulty when fetched was `min_solutions: 1`, `max_energy_milli:
-14,630,583`, `min_diversity_milli: 0`. One read below target is a valid
proof. Over the 60 blocks the target moved from −14,599,017 to −14,629,583,
tightening as it goes.

A proof is valid when its best energy is lower than the active target, and
an equal energy fails, per `pallets/quantum-pow/src/lib.rs:996`. Among
valid proofs that arrive in the same six-second chain block, the lowest
energy wins, per `lib.rs:1021`.
Rounds ran a median of 21 chain blocks, about two minutes, and 16 distinct
miners won at least once. Winners landed a median of 4,417 milli below
target, and 14 of 60 cleared it by less than 2,000.

Winners self-report their compute time as `device_access_time_us`. Over the
60 blocks it ran from 1.0 to 3.9 s, median 1.5 s. The chain does not check
the value, but it says the winning job on the network is a one-to-four
second job rather than a search that spans the round.

## Why reads are not independent draws

Independent reads would turn 0.30 at 32 reads into 0.51 at 64, against
0.40 measured, because the blocks differ far more than the seeds do. At
128 reads, 18 blocks were valid on all 5 seeds and 9 on none. A block is
either within reach at this sweep count or out of it, and reads move that
boundary rather than the odds on each side of it.

The mean count of reads below target in a valid job is 2.4 at 32 reads and
5.8 at 128. `DifficultyConfig` defaults `min_solutions` to 5. If the network
returns to that default, 32-read jobs, with 2.4 reads below target on
average, would fail the count that 128-read jobs pass with 5.8. The read
count is not free to fall without watching that field.

## Method

Three tools under `scripts/testnet/`, all new on this branch.

**Fetch.** The bootnodes speak libp2p only and no public RPC exists, so
`scripts/testnet/fetch_qblocks.py` reads from a local node. The node is the
published `quip-network-node` image with the Aglais chain specification,
warp-synced to head in about two minutes. The script calls `QuantumPowApi`
through the generic `state_call` RPC method. It reads `latest_qblock_id`
and then `qblock_by_id` for each id downward. It fetches `topology_meta`
once per distinct hash. It decodes the SCALE bytes by hand. `QBlock` is a
fixed 180-byte layout plus a 32-byte nonce, so a wrong layout fails the
length check.

**Regenerate.** A qblock stores no fields, couplings, or spins. The chain
draws them from the nonce with ChaCha8 over the registered allowed sets,
one `next_u32` per node then per edge. `scripts/testnet/regen` calls
`quip_protocol::chacha8::draw_ising_milli`, the same call the coordinator
makes when it hands a miner its job. Before drawing, it recomputes each
nonce as BLAKE3 over the last proof block hash, `blake2_256` of the miner
account, and the salt, and stops on a mismatch. All 60 matched, which pins
the nonce byte order: the runtime API returns it SCALE little-endian, and
the seed is the reverse. Node ids compact to the position in the on-chain
node list, so each field and coupling stays with its drawn node.

**Run.** `scripts/testnet/run_reads_study.py` pipes each problem through
`quip-metal-msa --solve`, eight processes at a time, at 32, 64, 128, and
256 reads, five seeds per block and read count, 16,384 sweeps, and the
default beta schedule. 1,200 jobs took 409 s. Each job's seed hashes the
block id, read count and seed index, and the run order interleaves read
counts so drift falls on every count alike.

### Commands

```sh
scripts/testnet/fetch_qblocks.py 60 qblocks.json
cargo run --manifest-path scripts/testnet/regen/Cargo.toml --locked --release -- qblocks.json problems/
cargo build --release --bin quip-metal-msa
scripts/testnet/run_reads_study.py target/release/quip-metal-msa problems/ study.csv
```

The docstring of `fetch_qblocks.py` carries the `docker run` line for the
node.

## Limits of this measurement

The chance per job comes from `--solve`, one job per process at low
occupancy. The jobs per second come from the batched streaming harness on
the fixture graph with fields. Combining them assumes the chance per job
does not depend on the batch that carries the job, which holds because the
kernel, sweep count and schedule are the same, and that the fixture's cost
per job matches the testnet's. Both graphs have 4,577 nodes and within one
edge of the same count, and the kernel's cost does not depend on the
coupling values.

The study covers sixty blocks over one difficulty window, on one machine,
at one sweep count. The target tightened through the window and keeps
moving, so the chance per job is a snapshot. The tools exist so the
snapshot can be retaken.

The winners' compute times are self-reported and unverifiable.

This study holds sweeps at 16,384 by instruction and does not measure the
sweep count against the same blocks.
