#!/bin/bash
# Evaluate ckpt.9 of all 6 runs on the val split, one at a time.
cd ~/habitat-lab
for name in ppo_s1 orion_s1 ppo_s2 orion_s2 ppo_s3 orion_s3 ppo_s4 orion_s4 ppo_s5 orion_s5; do
  case $name in ppo_*) cfg=orion_ppo_baseline.yaml;; orion_*) cfg=orion_success.yaml;; esac
  if grep -q "Average episode success" logs/eval/$name.log 2>/dev/null; then
    echo "$(date) $name already evaluated"; continue
  fi
  echo "$(date) evaluating $name"
  python -u -m habitat_baselines.run --config-name=pointnav/$cfg \
    habitat_baselines.evaluate=True \
    habitat_baselines.eval.split=val \
    habitat_baselines.test_episode_count=-1 \
    habitat_baselines.eval_ckpt_path_dir=data/runs/$name/ckpt.9.pth \
    habitat_baselines.checkpoint_folder=data/runs/$name \
    habitat_baselines.tensorboard_dir=tb/eval/$name > logs/eval/$name.log 2>&1
  echo "$(date) $name exit=$?"
  grep "Average episode" logs/eval/$name.log
done
echo "$(date) EVAL DONE"
