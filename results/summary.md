# ORION results (held-out val split)

Seeds: PPO = 10, ORION = 10. Mean +/- std across seeds.

| Method | Success (higher better) | SPL (higher better) | Distance to goal (m) (lower better) |
|---|---|---|---|
| PPO (baseline) | 0.619 +/- 0.240 | 0.519 +/- 0.198 | 1.056 +/- 0.457 |
| PPO + ORION | 0.635 +/- 0.171 | 0.526 +/- 0.150 | 0.643 +/- 0.163 |

## Welch t-test (ORION vs PPO)

| Metric | Diff in mean | t | df | p |
|---|---|---|---|---|
| Success | +0.016 | 0.17 | 16.2 | 0.866 |
| SPL | +0.007 | 0.09 | 16.8 | 0.932 |
| Distance to goal (m) | -0.413 | -2.69 | 11.2 | 0.021 |

## Per-run values

| Run | Success | SPL | Distance to goal (m) |
|---|---|---|---|
| ppo_s1 | 0.7500 | 0.6221 | 1.2789 |
| ppo_s10 | 0.6900 | 0.5911 | 0.7630 |
| ppo_s2 | 0.1600 | 0.1374 | 1.7869 |
| ppo_s3 | 0.3300 | 0.3029 | 1.8143 |
| ppo_s4 | 0.8200 | 0.6956 | 0.4305 |
| ppo_s5 | 0.8100 | 0.7158 | 0.7677 |
| ppo_s6 | 0.7200 | 0.6271 | 1.0523 |
| ppo_s7 | 0.3600 | 0.2987 | 1.0428 |
| ppo_s8 | 0.7600 | 0.5588 | 0.6644 |
| ppo_s9 | 0.7900 | 0.6403 | 0.9637 |
| orion_s1 | 0.5800 | 0.4556 | 0.5720 |
| orion_s10 | 0.8300 | 0.6650 | 0.5602 |
| orion_s2 | 0.4700 | 0.4293 | 0.6655 |
| orion_s3 | 0.7900 | 0.6831 | 0.4752 |
| orion_s4 | 0.5700 | 0.4586 | 0.8141 |
| orion_s5 | 0.8200 | 0.7136 | 0.6167 |
| orion_s6 | 0.8600 | 0.7133 | 0.5299 |
| orion_s7 | 0.4400 | 0.3468 | 0.6975 |
| orion_s8 | 0.4500 | 0.3527 | 1.0042 |
| orion_s9 | 0.5400 | 0.4401 | 0.4975 |
