Evaluated on PushT, 50 episodes, seeds 100000-100049 (identical for every policy).

| Policy | Success rate | 95% CI (Wilson) | Avg max reward | Eval time (s) |
|---|---|---|---|---|
| ACT run 2 (chunk 20, batch 32, 20k steps) | 7/50 = 14% | 7-26% | 0.763 | 89 |
| Diffusion Policy (lerobot/diffusion_pusht) | 29/50 = 58% | 44-71% | 0.933 | 656 |

Paired comparison on identical seeds (exact McNemar test):

| A | B | A ok / B fail | A fail / B ok | p-value |
|---|---|---|---|---|
| ACT run 2 (chunk 20, batch 32, 20k steps) | Diffusion Policy (lerobot/diffusion_pusht) | 1 | 23 | <0.001 |
