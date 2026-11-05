#!/bin/bash
# PPO vs ORION, seeds 1-3, ONE run at a time (Habitat on this Mac can't run 2 sims at once).
cd ~/habitat-lab
mkdir -p logs/runs
S=logs/runs/batch_status.txt

run() {
  method=$1; seed=$2; cfg=$3; name=${method}_s${seed}
  for attempt in 1 2 3; do
    if [ -f data/runs/$name/ckpt.9.pth ]; then echo "$(date) $name done" >> $S; return; fi
    rm -rf data/runs/$name tb/runs/$name
    echo "$(date) start $name (attempt $attempt)" >> $S
    python -u -m habitat_baselines.run --config-name=pointnav/$cfg \
      habitat.seed=$seed \
      habitat_baselines.checkpoint_folder=data/runs/$name \
      habitat_baselines.eval_ckpt_path_dir=data/runs/$name \
      habitat_baselines.tensorboard_dir=tb/runs/$name > logs/runs/${name}_a${attempt}.log 2>&1
    echo "$(date) $name attempt $attempt exited with $?" >> $S
    sleep 20
  done
  [ -f data/runs/$name/ckpt.9.pth ] || echo "$(date) $name FAILED after 3 attempts" >> $S
}

# Alternate PPO/ORION so a partial night still gives matched pairs.
for seed in 6 7 8 9 10; do
  run ppo   $seed orion_ppo_baseline.yaml
  run orion $seed orion_success.yaml
done
echo "$(date) ALL DONE" >> $S
