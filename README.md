# quip-miner-metal

Metal Ising miners for the [quip.network](https://gitlab.com/quip.network) v0.3
mining protocol: simulated annealing (`quip-metal-sa`), multi-spin coded SA
(`quip-metal-msa`) and heat-bath Gibbs (`quip-metal-gibbs`), shipped as
separate binaries.

**macOS arm64 only (Apple Silicon).** Metal and IOKit exist on no other
platform, so this crate does not build anywhere else and offers no stub or CPU
fallback. A non-macOS build fails while compiling the Apple-only dependencies.
Build and run on Apple Silicon.

Energies are scored on the host with the canonical
`quip_solver_core::quip_protocol::scoring::energy_milli` so results match
consensus; there is no GPU energy kernel (Metal Shading Language has no
`double`).

## Binaries

| binary | algorithm |
|--------|-----------|
| `quip-metal-sa` | simulated annealing (Metropolis) |
| `quip-metal-msa` | multi-spin coded simulated annealing (32 replicas per word) |
| `quip-metal-gibbs` | heat-bath Gibbs |

Prebuilt `arm64` binaries are attached to each
[Release](https://gitlab.com/quip.network/quip-miner-metal/-/releases)
(built best-effort on a macOS CI runner; see [`.gitlab-ci.yml`](.gitlab-ci.yml)).

## Build

```sh
cargo build --release
```

The solver contract (`quip-proto`, `quip-protocol`, `quip-solver-core`) comes
from crates.io at a pinned version, published from
[quip-solver-core](https://gitlab.com/quip.network/quip-solver-core).

## Running

**Connect to a coordinator** (production):

```sh
quip-metal-sa --quip-coordinator unix:///run/quip/coord.sock
```

**Driver / fixed-input (run in isolation, no chain).** Use the coordinator's
`drive` harness pointed at the binary — `--source random` for golden-drawn
problems, `--source list <jsonl>` for a fixed replay:

```sh
quip-coordinator drive --miner ./quip-metal-sa \
  --source random --topology-preset advantage2-system1 \
  --count 8 --num-reads 16 --num-sweeps 1030 --report out.jsonl
```

**Introspection:**

```sh
quip-metal-sa --capabilities   # capabilities JSON
quip-metal-sa --check          # probe the backend is runnable
```

## Engine selection

All binaries accept `enable_ane` and `enable_metal` in `backend_toml`. Both default to `true`.
Only MSA implements ANE execution. See [engine selection](docs/combined-engines.md) for modes, limits, and invalid settings.

## Multi-spin kernel (`quip-metal-msa`)

`kernels/msa.metal` is a Metal port of the multi-spin coded simulated
annealing in `quip-miner-cuda`'s `quip-cuda-msa` and `quip-miner-cpu`'s
`quip-cpu-msa` (Isakov, Zintchenko, Rønnow, Troyer, *Optimised simulated
annealing for Ising spin glasses*, Comput. Phys. Commun. 192, 2015). 32
replicas share one 32-bit word per spin. The Metropolis test is an integer
comparison against a per-rung geometric threshold table. The kernel updates
spins one colour class at a time (the greedy colouring the Gibbs kernel
uses). One threadgroup anneals one 32-replica word of one problem in
threadgroup memory. A 128-read job dispatches four threadgroups per problem.

The CUDA kernel packs 64 replicas per `u64` in 99 KB of shared memory. Apple
GPUs cap threadgroup memory at 32 KB per threadgroup, which one 64-bit word
per spin exceeds for Advantage2's 4577 spins, so this port uses 32-bit
words. See `docs/metal-msa-design.md` for the
full comparison.

The host rescores every sample with `energy_milli`, as for the other
kernels, because the device does not compute energies. Couplings must be in
`{-1, 1}` and fields in `{-1, 0, 1}`, which v0.3 problems meet. The CSR
degree must be at most 20. The miner rejects denser graphs as over capacity
so the coordinator routes them elsewhere.

The 2026-09-15 runs used Apple M4 Max with 40 GPU cores and
`tests/fixtures/advantage2-system1.edges`, the Advantage2 System 1 working
graph. The fixture has 4577 nodes, 41515 edges, and eight greedy classes.
At one threadgroup per core, 40 jobs, 128 reads, and 16384 sweeps, the
multi-spin kernel reached 7.27 jobs/s. The `quip-metal-sa` reference
reached 1.02 jobs/s at 2048 sweeps and 256 reads. The multi-spin sweep range
remains 4096 to 16384, with 128 reads. The occupancy curve uses safety 0.2.
Three rounds at 7392 sweeps covered 1, 2, 5, 10, and 40 jobs.
Their largest `max_chunk_ms` was 261. The sweep-range runs peaked at 258 ms.
See `tests/msa_bench.rs` for the commands to repeat each run.

## Tests

```sh
cargo test --release
```

Conformance/golden and handshake tests drive the binary in isolation via
`quip-solver-conformance`'s scripted driver and check energies against
`conformance/golden_vectors.json`. The whole suite runs on a Mac with a Metal
device; CI runs it on a macOS runner, which is the only configuration this
crate builds in.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
