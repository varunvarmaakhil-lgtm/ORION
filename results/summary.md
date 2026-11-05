# ORION results (held-out val split)

Seeds: PPO = 5, ORION = 5. Mean +/- std across seeds.

| Method | Success (higher better) | SPL (higher better) | Distance to goal (m) (lower better) |
|---|---|---|---|
| PPO (baseline) | 0.574 +/- 0.307 | 0.495 +/- 0.260 | 1.216 +/- 0.614 |
| PPO + ORION | 0.646 +/- 0.152 | 0.548 +/- 0.138 | 0.629 +/- 0.125 |

## Welch t-test (ORION vs PPO)

| Metric | Diff in mean | t | df | p |
|---|---|---|---|---|
| Success | +0.072 | 0.47 | 5.8 | 0.656 |
| SPL | +0.053 | 0.40 | 6.1 | 0.699 |
| Distance to goal (m) | -0.587 | -2.10 | 4.3 | 0.099 |

## Per-run values

| Run | Success | SPL | Distance to goal (m) |
|---|---|---|---|
| ppo_s1 | 0.7500 | 0.6221 | 1.2789 |
| ppo_s2 | 0.1600 | 0.1374 | 1.7869 |
| ppo_s3 | 0.3300 | 0.3029 | 1.8143 |
| ppo_s4 | 0.8200 | 0.6956 | 0.4305 |
| ppo_s5 | 0.8100 | 0.7158 | 0.7677 |
| orion_s1 | 0.5800 | 0.4556 | 0.5720 |
| orion_s2 | 0.4700 | 0.4293 | 0.6655 |
| orion_s3 | 0.7900 | 0.6831 | 0.4752 |
| orion_s4 | 0.5700 | 0.4586 | 0.8141 |
| orion_s5 | 0.8200 | 0.7136 | 0.6167 |
