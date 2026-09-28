"""Place your full benchmark scores beside sourced published-model scores.

This script does not run inference. It validates external_models.json and
produces CSV/Markdown tables with a direct source link for every external score.
"""

import argparse
import csv

import common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-file", required=True, help="Full-run benchmark.json.")
    parser.add_argument(
        "--external-file", default=str(common.HERE / "external_models.json"),
        help="Verified public-model scores and source URLs.",
    )
    parser.add_argument("--output-root", help="Where the new comparison folder is created.")
    parser.add_argument("--name", required=True, help="New, unique comparison folder name.")
    args = parser.parse_args()

    benchmark_path = common.path_from_repo(args.benchmark_file)
    external_path = common.path_from_repo(args.external_file)
    benchmark = common.read_json(benchmark_path)
    if not benchmark.get("complete") or benchmark["protocol"]["max_problems"] is not None:
        raise ValueError("Comparison requires a completed full-split benchmark, not a subset.")
    external = common.read_json(external_path)["models"]
    if not {"comparable", "larger"} <= {item.get("size_class") for item in external}:
        raise ValueError(
            "external_models.json needs at least one comparable and one larger "
            "model with sourced scores. See README.md."
        )

    own = benchmark["model"]
    rows = []
    for task in common.TASKS:
        score = benchmark["scores"][task]
        rows.append({
            "model": f"Nanochat {own['stage']} step {own['step']}",
            "size_class": "ours",
            "parameters": own.get("trainable_parameters", ""),
            "benchmark": task,
            "percent": score["percent"],
            "evaluated_examples": score["evaluated_examples"],
            "source_url": "",
            "protocol_notes": "Nanochat Task 3 evaluator; see benchmark.json.",
        })
    for item in external:
        for field in ("name", "size_class", "parameters", "scores"):
            if field not in item:
                raise ValueError(f"External model lacks {field}.")
        for task in common.TASKS:
            score = item["scores"].get(task)
            if not score or not score.get("source_url") or "percent" not in score:
                raise ValueError(f"{item['name']} lacks sourced {task} accuracy.")
            if not 0 <= score["percent"] <= 100:
                raise ValueError(f"{item['name']} {task} percent is outside 0-100.")
            rows.append({
                "model": item["name"],
                "size_class": item["size_class"],
                "parameters": item["parameters"],
                "benchmark": task,
                "percent": score["percent"],
                "evaluated_examples": score.get("evaluated_examples", ""),
                "source_url": score["source_url"],
                "protocol_notes": score.get("protocol_notes", ""),
            })

    directory = common.path_from_repo(
        args.output_root or common.HERE / "results"
    ) / common.run_name(args.name)
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    table = [
        "# Benchmark comparison",
        "",
        "Published scores may use different splits, prompts, few-shot settings, or answer rules.",
        "Read the linked sources and protocol notes in comparison.csv before drawing conclusions.",
        "",
        "| Model | Parameters | ARC-Easy | ARC-Challenge | GSM-8K |",
        "|---|---:|---:|---:|---:|",
    ]
    names = list(dict.fromkeys(row["model"] for row in rows))
    for name in names:
        values = {row["benchmark"]: row for row in rows if row["model"] == name}
        cells = []
        for task in common.TASKS:
            row = values[task]
            number = f"{row['percent']:.2f}%"
            cells.append(f"[{number}]({row['source_url']})" if row["source_url"] else number)
        table.append(
            f"| {name.replace('|', '/')} | {values[common.TASKS[0]]['parameters']} | "
            + " | ".join(cells) + " |"
        )
    (directory / "comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    common.write_json(directory / "sources.json", {
        "own_benchmark_file": str(benchmark_path),
        "own_checkpoint_sha256": own["model_sha256"],
        "external_file": str(external_path),
        "external_sha256": common.digest(external_path),
        "warning": "Check split, prompting, answer restriction, and scoring protocol "
                   "before interpreting cross-model differences.",
    })
    print(f"Saved sourced benchmark comparison: {directory}")


if __name__ == "__main__":
    main()
