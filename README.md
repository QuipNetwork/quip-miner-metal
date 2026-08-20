# quip-miner-metal

Metal Ising miners for the [quip.network](https://gitlab.com/quip.network) v0.3
mining protocol: simulated annealing (`quip-metal-sa`) and heat-bath Gibbs
(`quip-metal-gibbs`), shipped as separate binaries.

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
