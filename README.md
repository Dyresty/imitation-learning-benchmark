# PushT imitation learning benchmark: ACT vs Diffusion Policy

I trained ACT policies on LeRobot's PushT dataset and compared them with the pretrained Diffusion Policy on 50 identical simulator seeds. The best ACT run reached **30%** success against **58%** for Diffusion Policy. Most of the repo is about why, and about how much a single evaluation number can be trusted.

**Stack:** LeRobot 0.6.1 · PyTorch · Weights & Biases · gym-pusht · Google Colab (T4) and a local RTX 5070 Ti Laptop GPU
**W&B:** [project](https://wandb.ai/rochans-org/lerobot-pusht-benchmark) · [ACT run 4](https://wandb.ai/rochans-org/lerobot-pusht-benchmark/runs/6qg4etve) · [ACT run 3](https://wandb.ai/rochans-org/lerobot-pusht-benchmark/runs/5nvkitu4) · [ACT run 2](https://wandb.ai/rochans-org/lerobot-pusht-benchmark/runs/2v7lax9i) · [ACT run 1](https://wandb.ai/rochans-org/lerobot-pusht-benchmark/runs/z72m3mq5)
**Notebook:** [`pusht_act_vs_diffusion.ipynb`](pusht_act_vs_diffusion.ipynb) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dyresty/imitation-learning-benchmark/blob/main/pusht_act_vs_diffusion.ipynb)

## Goal

Measure how a policy I trained myself (ACT) compares with a strong public baseline (Diffusion Policy) on the same task, using one fixed evaluation protocol rather than a single headline number, and understand the gap.

## Setup

| | |
|---|---|
| Task | PushT: push a T-shaped block onto a target pose in 2D (`gym-pusht`). Observation: 96x96 RGB image + 2D agent position. Max 300 steps per episode. |
| Data | `lerobot/pusht`: 206 human demonstrations, 25,650 frames |
| Baseline | `lerobot/diffusion_pusht`, pretrained by the LeRobot team (not retrained) |
| Evaluation | 50 episodes per policy, seeds 100000-100049, so every policy starts from identical initial states. Success = T covers at least 95% of the target. "Max reward" is the best coverage-based reward reached in an episode (1.0 = success). |
| Hardware | Runs 1 and 2: Google Colab, NVIDIA T4. Runs 3 and 4: local NVIDIA RTX 5070 Ti Laptop GPU |

ACT settings (everything else is the LeRobot default: ResNet-18 backbone, CVAE objective, learning rate 1e-5):

| | Run 1 | Run 2 | Run 3 | Run 4 |
|---|---|---|---|---|
| Actions predicted per step (chunk size) | 100 | 20 | 20 | 20 |
| Actions executed before re-predicting | 100 | 10 | 10 | 10 |
| Batch size | 8 | 32 | 32 | 32 |
| KL weight (`kl_weight`) | 10 (default) | 10 (default) | 1 | 10 (default) |
| Training steps | 25k (about 8 passes over the data) | 20k (about 25 passes) | 20k (about 25 passes) | 100k (about 125 passes) |
| Training speed | about 10 steps/s (T4) | about 6 steps/s (T4, 54 min total) | about 5 steps/s (RTX 5070 Ti, 65 min total) | about 3.4 steps/s on average (RTX 5070 Ti, 8 h 13 min total; data loading on the CPU was the bottleneck) |

Run 3 is run 2 with only the KL weight changed, to test whether the collapsed latent (see below) is what holds ACT back. Run 4 is run 2 with only the number of training steps changed (5x), to see how much of the gap to Diffusion is just training budget.

## Results

Seeds 100000-100049, 50 episodes per policy ([`results/results.md`](results/results.md), per-seed numbers in [`results/per_seed.csv`](results/per_seed.csv)):

| Policy | Success | 95% CI (Wilson) | Avg max reward |
|---|---|---|---|
| Diffusion Policy (pretrained) | 29/50 = **58%** | 44-71% | 0.933 |
| ACT run 4 (run 2 settings, 100k steps) | 15/50 = **30%** | 19-44% | 0.765 |
| ACT run 3 (run 2 with `kl_weight` 1) | 12/50 = **24%** | 14-37% | 0.672 |
| ACT run 2 (chunk 20, batch 32, 20k steps) | 7/50 = **14%** | 7-26% | 0.763 |
| ACT run 1 (defaults)* | 0/50 = 0% | 0-7% | 0.434 |
| ACT run 1 + temporal ensembling*, inference-time only | 1/50 = 2% | 0-10% | 0.339 |

\* Run 1's checkpoint and evaluation files were lost when the Colab session reset, so these two rows are copied from the evaluation logs of the same protocol and seeds. They have no per-episode files or videos.

**Paired comparisons on identical seeds (exact McNemar test):**

| | Both succeed | Neither | First only | Second only | p |
|---|---|---|---|---|---|
| ACT run 2 vs Diffusion | 6 | 20 | 1 | 23 | < 0.001 |
| ACT run 3 vs Diffusion | 5 | 14 | 7 | 24 | 0.003 |
| ACT run 4 vs Diffusion | 7 | 13 | 8 | 22 | 0.016 |
| ACT run 3 vs ACT run 2 | 0 | 31 | 12 | 7 | 0.36 |
| ACT run 4 vs ACT run 2 | 2 | 30 | 13 | 5 | 0.10 |
| ACT run 4 vs ACT run 3 | 5 | 28 | 10 | 7 | 0.63 |

Run 3 has the higher success rate, but the difference from run 2 is not statistically significant, and its average max reward is actually lower (0.672 vs 0.763). Oddly, the two runs never succeed on the same seed.

Run 4 is the best ACT run so far, and training 5x longer moved it from 14% to 30%. Against run 2 it wins on 13 seeds and loses on 5, which points to longer training helping, but with p = 0.10 I can't call it significant on 50 episodes. It is still clearly behind Diffusion (p = 0.016).

**Training progress** (10 evaluation episodes at each checkpoint, so noisy):

| Step | Run 2 successes | Run 2 avg max reward | Run 3 successes | Run 3 avg max reward |
|---|---|---|---|---|
| 5k | 0/10 | 0.45 | 0/10 | 0.58 |
| 10k | 1/10 | 0.58 | 1/10 | 0.64 |
| 15k | 3/10 | 0.77 | 1/10 | 0.61 |
| 20k | 2/10 | 0.83 | 1/10 | 0.49 |

For run 2, average max reward rose at every checkpoint while the success count moved around. Run 3 started faster but did not keep improving, and its last 10-episode check (0.49) was the lowest of the four, yet the 50-episode evaluation of that same checkpoint gave its best result (24%). Ten episodes is just not enough to rank checkpoints.

Run 4 (checks every 10k steps):

| Step | 10k | 20k | 30k | 40k | 50k | 60k | 70k | 80k | 90k | 100k |
|---|---|---|---|---|---|---|---|---|---|---|
| Successes | 2/10 | 3/10 | 1/10 | 1/10 | 2/10 | 2/10 | 1/10 | 3/10 | 3/10 | 3/10 |
| Avg max reward | 0.77 | 0.65 | 0.56 | 0.85 | 0.80 | 0.70 | 0.69 | 0.79 | 0.77 | 0.67 |

The training loss kept going down the whole time (`l1_loss` 0.118 at 10k, 0.045 at 100k), but the 10-episode checks just bounce around with no clear trend after 20k.

## Failure analysis

Side by side, ACT run 2 (left) and Diffusion Policy (right), same seed. The pusher is the blue dot with the red cross, the green T is the target. Each GIF shows the full episode; an episode that ends early (success) holds its last frame.

**Failure patterns**

| | Failures | Within 10% of success (max reward >= 0.9) | Partial (0.5-0.9) | Never got going (< 0.5) |
|---|---|---|---|---|
| ACT run 2 | 43 | 21 | 9 | 13 |
| ACT run 3 | 38 | 13 | 7 | 18 |
| ACT run 4 | 35 | 17 | 4 | 14 |
| Diffusion Policy | 21 | 13 | 6 | 2 |

Both policies fail mostly by stopping just short of the strict 95% threshold. ACT additionally has a group of episodes (13 of 43) where it never gets the T near the target, which Diffusion rarely does (2 of 21 failures). Run 3 shifts this further: fewer near-misses, but more episodes that never get going (18 of 38). Run 4 still has 14 episodes that never get going, so training 5x longer did not get rid of that group. The GIFs below use run 2.

**Seed 100006: near-miss for both** (ACT max reward 0.997, Diffusion 0.998)
![seed 100006](media/seed_100006_act_vs_diffusion.gif)
ACT puts the T almost exactly on the target by about 15 s and hovers there without the last small correction; in the final seconds it knocks the T out of alignment. Diffusion settles a few pixels off the target and stays there.

**Seed 100005: partial for both** (ACT 0.453, Diffusion 0.693)
![seed 100005](media/seed_100005_act_vs_diffusion.gif)
ACT gets the T near the target at the wrong angle, then pushes it against the top edge of the arena, where it stays stuck for the last ~12 s. Diffusion approaches the target and then drifts away to the lower left.

**Seed 100008: Diffusion stalls** (ACT 0.986, Diffusion 0.270)
![seed 100008](media/seed_100008_act_vs_diffusion.gif)
Diffusion parks the pusher on one end of the T for about 16 s without moving it to the target, then drifts away. ACT gets this one almost aligned (a near-miss).

**Successes**

**Seed 100002: ACT succeeds in about 13 s, Diffusion misses by a hair** (ACT 1.000, Diffusion 0.995)
![seed 100002](media/seed_100002_act_vs_diffusion.gif)
Diffusion has the T almost aligned early on, then keeps adjusting for the full 30 s without crossing the threshold.

**Seed 100004: both succeed** (ACT in 289 of 300 steps, Diffusion in 176)
![seed 100004](media/seed_100004_act_vs_diffusion.gif)
ACT first pushes the T far to the lower left and then recovers; it finishes just before the step limit. Diffusion takes a more direct path.

The descriptions are from watching the rollouts (full-quality videos in [`media/videos/`](media/videos/)); the failure lists for every episode are in [`results/failures.md`](results/failures.md).

## Why the gap is so large

First, the comparison isn't even. The Diffusion checkpoint was trained by the LeRobot team; I only evaluated it. These are the reasons I think explain most of the gap between ACT (14% to 30% across runs 2-4) and Diffusion (58%), roughly in the order I first suspected them.

1. **Training budget.** The Diffusion checkpoint's `train_config.json` says 200k steps at batch size 64, so about 12.8M samples. ACT run 2 saw 20k x 32 = 640k samples, about 20 times less, and it was still improving when training ended (average max reward 0.45, 0.58, 0.77, 0.83 across the four checkpoints). I tested this with run 4: the same settings for 100k steps (3.2M samples, still 4 times less than Diffusion). Success went from 14% to 30%, but the gap to Diffusion is still significant (p = 0.016). So the budget explains part of the gap, not all of it.
2. **More than one way to solve the task.** There is often more than one sensible way to push the T into place, for example going around one side of it or the other. Diffusion Policy is built to model that kind of multimodal action distribution. ACT can only do it through its CVAE latent, and in run 2 that latent collapsed (`kld_loss` around 0.001). With the latent unused, ACT is essentially doing L1 regression, which tends to average two strategies into a hesitant path somewhere in between. That would fit the ACT failures that never got going (13 for run 2, 18 for run 3, 14 for run 4), against 2 for Diffusion. I tried to keep the latent alive in run 3 by lowering the KL weight from 10 to 1, but `kld_loss` still collapsed (0.0003 at 20k steps), and in run 4 it went to essentially zero from 30k steps on. So this stays a hypothesis, but it is now my main suspect for the part of the gap that more training did not close.
3. **Augmentation and observation history.** The Diffusion config trains on random 84x84 crops and looks at the last 2 frames (`n_obs_steps=2`). ACT with LeRobot defaults gets a single frame and no augmentation. With only 206 demonstrations, augmentation probably makes a real difference.
4. **ACT wasn't designed for this kind of task.** ACT was built for bimanual manipulation from 50 Hz teleoperation data, predicting long action chunks. PushT runs at 10 Hz, and success depends on small, precise corrections right at the end. That is where most of the failures are: 21 of run 2's 43 failures reached a max reward of 0.9 or more.

Points 2 to 4 are harder to separate. The next things I'd try are adding random-crop augmentation to ACT (point 3) and an ACT variant without the CVAE (`use_vae=false`): if that one does just as well, the latent was never doing anything on this task anyway.

## What I learned

- **A single number hides a lot.** Success rate on 50 episodes has a wide confidence interval (14% is 7-26%), and the 10-episode checks during training swung between 0% and 30% while the underlying reward steadily rose. Average max reward and the near-miss counts were more informative than success rate alone.
- **Evaluation was reproducible.** Re-running the Diffusion evaluation in a fresh session gave identical numbers (29/50, 0.933), so differences between policies come from the policies, not the simulator.
- **Training configuration mattered more than inference tricks.** Temporal ensembling on the run-1 checkpoint did not help (2% vs 0%). Changing the chunking and training budget in run 2 took ACT from 0% to 14%. Several settings changed together, so I cannot say which one did it.
- **ACT's latent variable collapsed** in run 2 (`kld_loss` fell to about 0.001), meaning the model barely uses it. PushT has many valid ways to push the T, so I suspect this hurts. Lowering the KL weight to 1 in run 3 did not prevent the collapse, and the success rate went to 24% with no significant difference from run 2 (p = 0.36). So a lower KL weight alone is not the fix.
- **More training helps, but only so far.** Training 5x longer (run 4) took ACT from 14% to 30%, the best ACT result, while the training loss kept dropping. The 10-episode checks stayed flat after 20k, and the 50-seed result is still well below Diffusion. A lower loss did not turn into a matching gain in success.
- **Both policies lack a clean "finish".** Many failures hover within a few percent of the threshold, and on seed 100002 Diffusion keeps adjusting for the full episode without crossing it.

## Next steps

- Train ACT with random-crop augmentation, and an ACT variant without the CVAE (`use_vae=false`), to separate points 2 and 3 above.
- Evaluate on more seeds (for example 200) to tighten the confidence intervals.
- Fine-tune a vision-language-action model (SmolVLA with LoRA) on a LeRobot simulation task and evaluate it with this same protocol.

## Reproduce

1. Open the notebook in Colab with a GPU runtime, add your `WANDB_API_KEY` to Colab Secrets.
2. Run the sections in order. Checkpoints and evaluation outputs are copied to Google Drive.
3. The comparison script works on any `lerobot-eval` outputs:
   `python scripts/compare_results.py --run "ACT=outputs/eval/act" --run "Diffusion=outputs/eval/diffusion" --start-seed 100000`

## Repo layout

```
pusht_act_vs_diffusion.ipynb   training and evaluation notebook
scripts/compare_results.py     success rates, Wilson CIs, McNemar test, per-seed CSV, failure list
results/                       results.md, per_seed.csv, failures.md
media/                         side-by-side GIFs; media/videos has the full-quality rollouts
```

## Acknowledgements

Built on [LeRobot](https://github.com/huggingface/lerobot) (Apache-2.0). Baseline checkpoint `lerobot/diffusion_pusht` by the LeRobot team. ACT: Zhao et al., 2023. Diffusion Policy: Chi et al., 2023.
