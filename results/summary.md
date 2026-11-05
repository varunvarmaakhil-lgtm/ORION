# ORION results (held-out val split)

Seeds: PPO = 3, ORION = 3. Mean +/- std across seeds.

| Method | Success (higher better) | SPL (higher better) | Distance to goal (m) (lower better) |
|---|---|---|---|
| PPO (baseline) | 0.413 +/- 0.304 | 0.354 +/- 0.246 | 1.627 +/- 0.302 |
| PPO + ORION | 0.613 +/- 0.163 | 0.523 +/- 0.140 | 0.571 +/- 0.095 |

## Welch t-test (ORION vs PPO)

| Metric | Diff in mean | t | df | p |
|---|---|---|---|---|
| Success | +0.200 | 1.01 | 3.1 | 0.387 |
| SPL | +0.169 | 1.03 | 3.2 | 0.375 |
| Distance to goal (m) | -1.056 | -5.78 | 2.4 | 0.019 |

## Per-run values

| Run | Success | SPL | Distance to goal (m) |
|---|---|---|---|
| ppo_s1 | 0.7500 | 0.6221 | 1.2789 |
| ppo_s2 | 0.1600 | 0.1374 | 1.7869 |
| ppo_s3 | 0.3300 | 0.3029 | 1.8143 |
| orion_s1 | 0.5800 | 0.4556 | 0.5720 |
| orion_s2 | 0.4700 | 0.4293 | 0.6655 |
| orion_s3 | 0.7900 | 0.6831 | 0.4752 |
