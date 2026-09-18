#!/bin/sh
# Run the utilization probe's roofline kernels and sweep grid for ROUNDS
# rounds while macmon samples ANE and DRAM power, then dump the aned log
# for the run so the summary can count other clients' program creations
# inside each timed window. Must run inside one scripts/ane-guard
# acquisition.
# Usage: utilization.sh PROBE BUDGET_MS OUT_DIR ROUNDS
set -eu
probe=$1
budget=$2
out=$3
rounds=$4
mkdir -p "$out"
macmon pipe -i 250 -s 0 >"$out/power.jsonl" &
macmon_pid=$!
trap 'kill "$macmon_pid" 2>/dev/null || true' EXIT
started=$(date '+%Y-%m-%d %H:%M:%S')
sleep 5
: >"$out/results.jsonl"
round=1
while [ "$round" -le "$rounds" ]; do
	{
		"$probe" roofline "$budget"
		for enc in fp16 sparse; do
			"$probe" sweep "$budget" "$enc" 1x32 1x64 1x128 2x128 4x128 8x128 1x256 1x512
		done
	} | sed "s/^{/{\"round\":$round,/" >>"$out/results.jsonl"
	round=$((round + 1))
done
sleep 5
/usr/bin/log show --start "$started" --predicate 'process == "aned"' --style compact >"$out/aned.log" 2>&1 || true
