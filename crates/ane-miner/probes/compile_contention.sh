#!/bin/sh
# Run N copies of compile-contention against one synchronized start and
# report whether their compiles overlapped.
#
# Usage: compile_contention.sh PROBE N [small|production]
#
# Run this through scripts/ane-guard, once around the whole experiment.
# Do not guard the individual copies: this probe measures concurrent device
# access, so serializing the copies would destroy the measurement.
#
#   scripts/ane-guard -- crates/ane-miner/probes/compile_contention.sh \
#       /path/to/compile-contention 4 production
#
# Reports each copy's compile interval, then three derived figures:
#   span_ms        first start to last end across every copy.
#   union_ms       wall time with overlaps counted once.
#   overlap_depth  the most copies compiling at any instant.
#
# overlap_depth is the answer. 1 means compile serializes across processes.
# N means it runs fully concurrently.

set -eu

[ $# -ge 2 ] || {
	echo "usage: compile_contention.sh PROBE N [small|production]" >&2
	exit 64
}

probe=$1
copies=$2
shape=${3:-small}

[ -x "$probe" ] || {
	echo "not executable: $probe" >&2
	exit 64
}
case $copies in
'' | *[!0-9]*)
	echo "copy count must be a positive integer" >&2
	exit 64
	;;
esac
[ "$copies" -gt 0 ] || {
	echo "copy count must be positive" >&2
	exit 64
}

# One second of headroom, so every copy is allocated and parked at the
# barrier before the deadline passes. The probe reports late=true if it
# missed, which invalidates that copy's reading.
now=$("$probe" now)
start=$((now + 1000000))

out=$(mktemp -d)
trap 'rm -f "$out"/*.json; rmdir "$out" 2>/dev/null || true' EXIT

i=0
while [ "$i" -lt "$copies" ]; do
	"$probe" "$start" "$shape" >"$out/$i.json" &
	i=$((i + 1))
done
wait

cat "$out"/*.json

cat "$out"/*.json | awk -v copies="$copies" '
# n must start at 0. An unset awk variable used as a subscript indexes the
# empty string, not 0, so the first record would land in st[""] and every
# later index would be off by one.
BEGIN { n = 0; late = 0; total = 0 }
{
  match($0, /"create_start_us":[0-9]+/); s = substr($0, RSTART+18, RLENGTH-18) + 0
  match($0, /"create_end_us":[0-9]+/);   e = substr($0, RSTART+16, RLENGTH-16) + 0
  match($0, /"create_ms":[0-9.]+/);      m = substr($0, RSTART+12, RLENGTH-12) + 0
  if (index($0, "\"late\":true")) late++
  st[n] = s; en[n] = e; ms[n] = m; n++
  if (first == 0 || s < first) first = s
  if (e > last) last = e
  total += m
}
END {
  if (n == 0) { print "no readings"; exit 1 }

  # Sweep every endpoint and count how many intervals cover it. The maximum
  # is the deepest concurrency actually reached.
  depth = 0
  for (i = 0; i < n; i++) {
    c = 0
    for (j = 0; j < n; j++) if (st[j] <= st[i] && st[i] < en[j]) c++
    if (c > depth) depth = c
  }

  # Union of the intervals, overlaps counted once.
  for (i = 0; i < n; i++) order[i] = i
  for (i = 0; i < n; i++)
    for (j = i+1; j < n; j++)
      if (st[order[j]] < st[order[i]]) { t = order[i]; order[i] = order[j]; order[j] = t }
  union = 0; cs = st[order[0]]; ce = en[order[0]]
  for (i = 1; i < n; i++) {
    k = order[i]
    if (st[k] > ce) { union += ce - cs; cs = st[k]; ce = en[k] }
    else if (en[k] > ce) ce = en[k]
  }
  union += ce - cs

  printf "\ncopies=%d late=%d\n", copies, late + 0
  printf "mean_create_ms=%.3f\n", total / n
  printf "span_ms=%.3f\n", (last - first) / 1000.0
  printf "union_ms=%.3f\n", union / 1000.0
  printf "sum_create_ms=%.3f\n", total
  printf "overlap_depth=%d\n", depth
}'
