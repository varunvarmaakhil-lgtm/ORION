#!/bin/bash
# Live progress for all 6 ORION runs. Ctrl+C to exit (training keeps running).
cd ~/habitat-lab
while true; do
  clear
  echo "ORION batch progress   $(date '+%H:%M:%S')"
  echo "------------------------------------------------------------"
  for name in ppo_s1 orion_s1 ppo_s2 orion_s2 ppo_s3 orion_s3 ppo_s4 orion_s4 ppo_s5 orion_s5; do
    f=$(ls -t logs/runs/${name}_a*.log 2>/dev/null | head -1)
    if grep -q "$name attempt [0-9] exited with 0" logs/runs/batch_status.txt 2>/dev/null; then
      u=10000; status="DONE"
    elif [ -z "$f" ]; then
      u=0; status="waiting"
    else
      u=$(grep -o "Num updates: [0-9]*" "$f" | tail -1 | grep -o "[0-9]*$"); u=${u:-0}
      fps=$(grep -o "fps: [0-9.]*" "$f" | tail -1 | grep -o "[0-9.]*$")
      status="running ${fps:-?} fps"
    fi
    pct=$((u * 100 / 10000)); n=$((pct / 4))
    bar=$(printf '%*s' $n | tr ' ' '#')$(printf '%*s' $((25 - n)) | tr ' ' '.')
    printf "%-9s [%s] %3d%%  %5d/10000  %s\n" "$name" "$bar" "$pct" "$u" "$status"
  done
  echo "------------------------------------------------------------"
  tail -1 logs/runs/batch_status.txt 2>/dev/null
  sleep 15
done
