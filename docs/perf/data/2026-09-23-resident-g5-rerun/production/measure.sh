#!/bin/zsh
S=/private/tmp/claude-501/-Users-carback1-Code-quip-quip-miner-metal/834bffcf-b9be-46c7-8cc7-064fca302acb/scratchpad/prod-measure
C=$(pgrep -f "bin/quip-coordinator --config"); M=$(pgrep -f "quip-metal-msa --quip-coordinator")
echo "coord=$C miner=$M"; uptime
/usr/bin/sample $C 10 1 -mayDie -file $S/coord.sample.txt >/dev/null 2>&1 &
/usr/bin/sample $M 10 1 -mayDie -file $S/miner.sample.txt >/dev/null 2>&1 &
for i in $(seq 1 24); do
  echo "t=$((i*5)) $(ps -o %cpu= -p $C) $(ps -o %cpu= -p $M) gpu=$(ioreg -r -d 1 -c IOAccelerator | grep -o '"Device Utilization %"=[0-9]*' | head -1)"
  /bin/sleep 5
done
wait; uptime
