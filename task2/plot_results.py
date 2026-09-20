"""Create training/validation bpb plots from a recorded Task 2 run."""

import argparse
import csv
import json

from task2 import ROOT, read_json, run_environment


def read_evaluations(path):
    """Select the periodic evaluation records used for comparable learning curves."""
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = [record for record in records if record["event"] == "evaluation"]
    if not rows or any(row["train_bpb"] is None for row in rows):
        raise ValueError("Run task2.py evaluate first to record both training and validation bpb.")
    return rows


def save_plot(rows, destination, title):
    """Plot training and validation bpb against the number of training tokens."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["tokens_seen"] for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, [row["train_bpb"] for row in rows], marker="o", label="Training sample")
    ax.plot(x, [row["validation_bpb"] for row in rows], marker="o", label="Validation sample")
    ax.set(xlabel="Training tokens processed", ylabel="Bits per byte (lower is better)", title=title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination.with_suffix(".png"), dpi=180)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def main():
    """Export the evaluation table and plots for the selected run."""
    import os
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run name under task2/results/.")
    args = parser.parse_args()
    os.environ.update(run_environment())
    results = ROOT / "results" / args.run
    rows = read_evaluations(results / "metrics.jsonl")
    with (results / "evaluation.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    mode = read_json(results / "run.json")["mode"]
    title = f"Task 2 depth-2 {mode}: {args.run}"
    destination = ROOT / "figures" / args.run
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_plot(rows, destination, title)
    print(f"Saved evaluation.csv and plots: {destination}.png / .pdf")


if __name__ == "__main__":
    main()
