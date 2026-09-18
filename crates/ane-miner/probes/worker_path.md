# Worker path measurement note

Task 8 of the 2026-09-17 ANE throughput plan. Full method, tables, and gap
accounting are in the "Worker path" section of
`docs/perf/2026-09-17-ane-setup-profile.md`. This note is a short pointer
to that section and the commands behind it, not a duplicate of it.

## What this note measures

`crates/ane-miner/src/process.rs` times six stages of one out-of-process
job: directory and file creation, the request write, `Command::spawn`,
the wait for the child to exit, the reply read, and teardown.
`crates/ane-miner/src/worker.rs` and
`crates/ane-miner/src/bin/quip_ane_msa.rs` add a seventh counter,
`child_arg_parse_us`. It times `main` entry to `worker_main` entry, which
is argument parsing, not the child's `dyld` and runtime startup: `dyld`
finishes before `main` ever runs, outside this window. The child prints
this value, and whatever spawned it captures the value. All seven are
counters only. No control flow changed.

## How to reproduce

```bash
cargo build --manifest-path crates/ane-miner/Cargo.toml --locked --release \
  --bin quip-ane-msa --tests
for i in 1 2 3 4 5; do
  cargo test --manifest-path crates/ane-miner/Cargo.toml --locked --release \
    --lib hardware_worker_path_stage_medians_advantage2_system1 \
    -- --ignored --nocapture --exact \
    process::tests::hardware_worker_path_stage_medians_advantage2_system1
  if [ "$i" -lt 5 ]; then sleep 3; fi
done
```

No shared device guard exists in this repository or in `/tmp`. Run one
process at a time, with a 3-second sleep between runs, in place of a
guard.

## Headline numbers

Full run median, spawn to reply consumed, five runs: 398.093 ms. Wrapper
cost alone, the parts of a job outside the child's own compute:
19.082 ms. Reconciled against the 309.462 ms native sum, Task 1, and the
5.647 ms Rust sum, Task 7, the redone total misses the 508.162 ms budget
by 110.069 ms, 21.7 percent of budget. The doc section gives two ways to
compute this and explains why they differ. It also says which one this
task recommends.
