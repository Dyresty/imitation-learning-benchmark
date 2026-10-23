"""Compare LeRobot eval runs that were evaluated on the same seeds.

Reads the eval_info.json written by `lerobot-eval` for each run and produces:
  results/results.md   - summary table (success rate with 95% CI, avg max reward) + paired test
  results/per_seed.csv - one row per seed, success/max_reward for every run
  results/failures.md  - failed seeds per run, with the rendered video path when one exists

Usage:
  python compare_results.py \
      --run "ACT (ours, 25k steps)=outputs/eval/act" \
      --run "Diffusion (lerobot/diffusion_pusht)=outputs/eval/diffusion" \
      --start-seed 100000 --out results [--wandb-project lerobot-pusht-benchmark]
"""

import argparse
import csv
import json
import math
from pathlib import Path


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs (b, c)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def load_run(eval_dir: Path, start_seed: int) -> dict:
    info = json.loads((eval_dir / "eval_info.json").read_text())
    tasks = info["per_task"]
    if len(tasks) != 1:
        raise ValueError(f"{eval_dir}: expected a single task (PushT), found {len(tasks)}")
    metrics = tasks[0]["metrics"]
    successes = [bool(s) for s in metrics["successes"]]
    max_rewards = [float(r) for r in metrics["max_rewards"]]
    # lerobot-eval seeds episode i with start_seed + i, and renders the first episodes in order.
    videos = dict(enumerate(metrics.get("video_paths", [])))
    return {
        "seeds": [start_seed + i for i in range(len(successes))],
        "successes": successes,
        "max_rewards": max_rewards,
        "videos": videos,
        "eval_s": info.get("overall", {}).get("eval_s"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help='"Label=path/to/eval_dir"')
    parser.add_argument("--start-seed", type=int, required=True, help="The --seed passed to lerobot-eval")
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--wandb-project", default=None, help="Also log the tables (and failure videos) to W&B")
    args = parser.parse_args()

    runs = {}
    for spec in args.run:
        label, _, path = spec.partition("=")
        runs[label] = load_run(Path(path), args.start_seed)

    seed_lists = {tuple(r["seeds"]) for r in runs.values()}
    if len(seed_lists) != 1:
        raise ValueError("Runs were not evaluated on the same number of episodes / seeds")
    seeds = runs[next(iter(runs))]["seeds"]
    n = len(seeds)
    args.out.mkdir(parents=True, exist_ok=True)

    # Summary table
    lines = [
        f"Evaluated on PushT, {n} episodes, seeds {seeds[0]}-{seeds[-1]} (identical for every policy).",
        "",
        "| Policy | Success rate | 95% CI (Wilson) | Avg max reward | Eval time (s) |",
        "|---|---|---|---|---|",
    ]
    for label, r in runs.items():
        k = sum(r["successes"])
        lo, hi = wilson_ci(k, n)
        avg_max = sum(r["max_rewards"]) / n
        eval_s = f"{r['eval_s']:.0f}" if r["eval_s"] is not None else "-"
        lines.append(f"| {label} | {k}/{n} = {100 * k / n:.0f}% | {100 * lo:.0f}-{100 * hi:.0f}% | {avg_max:.3f} | {eval_s} |")

    # Paired comparison between every pair of runs (same seeds -> McNemar)
    labels = list(runs)
    if len(labels) > 1:
        lines += ["", "Paired comparison on identical seeds (exact McNemar test):", ""]
        lines += ["| A | B | A ok / B fail | A fail / B ok | p-value |", "|---|---|---|---|---|"]
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                a, b = runs[labels[i]]["successes"], runs[labels[j]]["successes"]
                only_a = sum(x and not y for x, y in zip(a, b))
                only_b = sum(y and not x for x, y in zip(a, b))
                lines.append(f"| {labels[i]} | {labels[j]} | {only_a} | {only_b} | {mcnemar_exact_p(only_a, only_b):.3f} |")
    (args.out / "results.md").write_text("\n".join(lines) + "\n")

    # Per-seed CSV
    with open(args.out / "per_seed.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["seed"] + [f"{l} {m}" for l in labels for m in ("success", "max_reward")])
        for i, seed in enumerate(seeds):
            writer.writerow([seed] + [v for l in labels for v in (int(runs[l]["successes"][i]), f"{runs[l]['max_rewards'][i]:.4f}")])

    # Failures
    fail_lines = ["# Failed episodes", ""]
    for label, r in runs.items():
        failed = [i for i, s in enumerate(r["successes"]) if not s]
        fail_lines.append(f"## {label}: {len(failed)} failures")
        fail_lines.append("")
        for i in failed:
            video = r["videos"].get(i)
            note = f" - video: `{video}`" if video else " - no video (only the first episodes are rendered)"
            fail_lines.append(f"- seed {seeds[i]} (episode {i}), max reward {r['max_rewards'][i]:.3f}{note}")
        fail_lines.append("")
    (args.out / "failures.md").write_text("\n".join(fail_lines))

    print((args.out / "results.md").read_text())
    print(f"Wrote {args.out}/results.md, per_seed.csv, failures.md")

    if args.wandb_project:
        import wandb

        run = wandb.init(project=args.wandb_project, name="pusht-benchmark-comparison", job_type="eval")
        summary = wandb.Table(columns=["policy", "success_rate", "ci_low", "ci_high", "avg_max_reward"])
        for label, r in runs.items():
            k = sum(r["successes"])
            lo, hi = wilson_ci(k, n)
            summary.add_data(label, k / n, lo, hi, sum(r["max_rewards"]) / n)
        per_seed = wandb.Table(columns=["policy", "seed", "success", "max_reward", "video"])
        for label, r in runs.items():
            for i, seed in enumerate(seeds):
                video = r["videos"].get(i)
                per_seed.add_data(label, seed, r["successes"][i], r["max_rewards"][i],
                                  wandb.Video(video, format="mp4") if video and Path(video).exists() else None)
        run.log({"summary": summary, "per_seed": per_seed})
        run.finish()


if __name__ == "__main__":
    main()
