# quip-miner-metal

Metal Ising miners for the [quip.network](https://gitlab.com/quip.network) v0.3
mining protocol: simulated annealing (`quip-metal-sa`) and heat-bath Gibbs
(`quip-metal-gibbs`), shipped as separate binaries.

**macOS arm64 only (Apple Silicon).** The Metal/IOKit dependencies are gated
behind `cfg(target_os = "macos")`, so the crate compiles on Linux (the gated
code is simply excluded), but the binaries are non-functional off macOS —
there is no CPU fallback. Build and run on Apple Silicon to get a working
miner.

Energies are scored on the host with the canonical
`quip_protocol::scoring::energy_milli` so results match consensus; there is no
GPU energy kernel (Metal Shading Language has no `double`).

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
cargo build --release        # needs protoc on PATH (protobuf-compiler)
```

Shared protocol crates (`quip-proto`, `quip-protocol`, `quip-miner-core`) are
git dependencies pinned to a `shared-vX.Y.Z` tag of `quip-protocol`.

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
`quip-mock-coordinator` and check energies against
`conformance/golden_vectors.json`. The Metal-device golden-parity tests
(`tests/golden_parity.rs`) are `cfg(target_os = "macos")`-gated and only run
on a Mac with a Metal device.

## License

AGPL-3.0-or-later. See [LICENSE](LICENSE).
