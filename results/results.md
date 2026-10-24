Evaluated on PushT, 50 episodes, seeds 100000-100049 (identical for every policy).

| Policy | Success rate | 95% CI (Wilson) | Avg max reward | Eval time (s) |
|---|---|---|---|---|
| Diffusion Policy (lerobot/diffusion_pusht) | 29/50 = 58% | 44-71% | 0.933 | 656 |
| ACT run 4 (run 2 settings, 100k steps) | 15/50 = 30% | 19-44% | 0.765 | 63 |
| ACT run 3 (kl_weight 1, 20k steps) | 12/50 = 24% | 14-37% | 0.672 | 62 |
| ACT run 2 (chunk 20, batch 32, 20k steps) | 7/50 = 14% | 7-26% | 0.763 | 89 |

Paired comparison on identical seeds (exact McNemar test):

| A | B | A ok / B fail | A fail / B ok | p-value |
|---|---|---|---|---|
| Diffusion Policy (lerobot/diffusion_pusht) | ACT run 4 (run 2 settings, 100k steps) | 22 | 8 | 0.016 |
| Diffusion Policy (lerobot/diffusion_pusht) | ACT run 3 (kl_weight 1, 20k steps) | 24 | 7 | 0.003 |
| Diffusion Policy (lerobot/diffusion_pusht) | ACT run 2 (chunk 20, batch 32, 20k steps) | 23 | 1 | <0.001 |
| ACT run 4 (run 2 settings, 100k steps) | ACT run 3 (kl_weight 1, 20k steps) | 10 | 7 | 0.629 |
| ACT run 4 (run 2 settings, 100k steps) | ACT run 2 (chunk 20, batch 32, 20k steps) | 13 | 5 | 0.096 |
| ACT run 3 (kl_weight 1, 20k steps) | ACT run 2 (chunk 20, batch 32, 20k steps) | 12 | 7 | 0.359 |
