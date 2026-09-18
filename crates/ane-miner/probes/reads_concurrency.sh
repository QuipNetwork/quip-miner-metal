#!/bin/sh
# Launch N copies of the reads probe at once and record every JSON line.
# Usage: reads_concurrency.sh PROBE CALLS OUT_FILE ROUNDS
# Must run inside one scripts/ane-guard acquisition.
set -eu
probe=$1
calls=$2
out=$3
rounds=$4
: >"$out"
round=1
while [ "$round" -le "$rounds" ]; do
	for reads in 128 64 32; do
		for n in 1 2 4; do
			i=0
			while [ "$i" -lt "$n" ]; do
				"$probe" "$calls" "$reads" | sed "s/^/{\"round\":$round,\"n\":$n,\"proc\":$i,/; s/^\({[^{]*\){/\1/" >>"$out" &
				i=$((i + 1))
			done
			wait
			sleep 2
		done
	done
	round=$((round + 1))
done
