#!/bin/sh
# Launch N copies of the utilization probe's sweep at once, for each
# encoding and layout in the grid, and record every JSON line tagged with
# the launch. macmon samples power throughout. Must run inside one
# scripts/ane-guard acquisition.
# Usage: utilization_concurrency.sh PROBE BUDGET_MS OUT_DIR ROUNDS
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
	for config in "fp16 1x128" "sparse 1x64" "sparse 1x128" "sparse 8x128"; do
		enc=${config% *}
		layout=${config#* }
		for n in 1 2 4; do
			# Wait on the probe pipelines by pid: a bare wait would also wait
			# for macmon, which runs until the script exits.
			pids=""
			i=0
			while [ "$i" -lt "$n" ]; do
				"$probe" sweep "$budget" "$enc" "$layout" | sed "s/^{/{\"round\":$round,\"n\":$n,\"proc\":$i,/" >>"$out/results.jsonl" &
				pids="$pids $!"
				i=$((i + 1))
			done
			for pid in $pids; do
				wait "$pid"
			done
			sleep 2
		done
	done
	round=$((round + 1))
done
sleep 5
/usr/bin/log show --start "$started" --predicate 'process == "aned"' --style compact >"$out/aned.log" 2>&1 || true
