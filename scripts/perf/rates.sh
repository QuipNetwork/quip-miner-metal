#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 QUIP Protocol Contributors
#
# One rate measurement: rates.sh LABEL JOBS READS SWEEPS OUT_DIR
# Writes OUT_DIR/LABEL.log (debug log) and OUT_DIR/LABEL.json (parse.py).
set -euo pipefail

label=$1 jobs=$2 reads=$3 sweeps=$4 out=$5
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
uptime >"$out/$label.uptime"
QUIP_BENCH_KERNEL=msa QUIP_BENCH_JOBS="$jobs" QUIP_BENCH_READS="$reads" \
	QUIP_BENCH_SWEEPS="$sweeps" RUST_LOG=quip_solver_metal=debug \
	cargo test --release --test msa_bench -- --ignored --nocapture \
	>"$out/$label.log" 2>&1
python3 "$here/parse.py" <"$out/$label.log" >"$out/$label.json"
cat "$out/$label.json"
