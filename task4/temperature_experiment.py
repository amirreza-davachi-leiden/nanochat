"""Compare the same prompts at several sampling temperatures.

All responses use the same checkpoint, chat template, top-k, and output
budget. Outputs are saved as JSONL and a small CSV; nothing is retrained.
"""

import argparse
import csv
import json
import statistics

import common


def save_summary(directory, rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["prompt_id"], row["temperature"]), []).append(row)
    with (directory / "temperature.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "prompt_id", "temperature", "samples", "distinct_responses",
            "mean_generated_tokens", "mean_seconds",
        ])
        writer.writeheader()
        for (prompt_id, temperature), group in groups.items():
            writer.writerow({
                "prompt_id": prompt_id,
                "temperature": temperature,
                "samples": len(group),
                "distinct_responses": len({row["response"] for row in group}),
                "mean_generated_tokens": round(statistics.mean(
                    row["generated_tokens"] for row in group
                ), 2),
                "mean_seconds": round(statistics.mean(
                    row["seconds"] for row in group
                ), 3),
            })

    # These are comparison candidates, not automatically selected "best" answers.
    lines = [
        "# Candidate examples", "",
        "Review these outputs yourself before choosing five for the report.", "",
    ]
    for (prompt_id, temperature), group in groups.items():
        row = group[0]
        lines.extend([
            f"## {prompt_id} | temperature {temperature} | seed {row['seed']}",
            "", f"Prompt: {row['prompt']}", "", "Response:", "",
            *(f"    {line}" for line in row["response"].splitlines() or [""]),
            "", f"Stop: {row['stop_reason']}; output tokens: {row['generated_tokens']}", "",
        ])
    (directory / "candidate_examples.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_model_options(parser)
    parser.add_argument("--name", required=True, help="New, unique results folder name.")
    parser.add_argument("--prompts", help="JSON prompts file; defaults to task4/prompts.json.")
    parser.add_argument("--prompt-id", action="append", help="Select one prompt ID; repeat as needed.")
    parser.add_argument("--temperatures", nargs="+", type=float)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()

    _, spec, generation, _ = common.configuration(args)
    identity = common.verify_files(spec)
    max_new, top_k = common.generation_settings(args, generation)
    temperatures = args.temperatures or generation["temperatures"]
    seeds = args.seeds or generation["seeds"]
    if not temperatures or any(value < 0 for value in temperatures):
        raise ValueError("Provide at least one nonnegative temperature.")
    if not seeds or any(value < 0 for value in seeds):
        raise ValueError("Provide at least one nonnegative seed.")
    prompts_path = common.path_from_repo(args.prompts or common.HERE / "prompts.json")
    prompts = common.load_prompts(prompts_path, args.prompt_id)
    model, tokenizer, engine = common.load_nanochat(spec, identity)
    for item in prompts:
        common.context_check(
            model, common.chat_prefix(tokenizer, item["text"]), max_new
        )

    settings = {
        "prompts_file": str(prompts_path),
        "prompts_sha256": common.digest(prompts_path),
        "prompt_ids": [item["id"] for item in prompts],
        "temperatures": temperatures,
        "seeds": seeds,
        "top_k": top_k,
        "max_new_tokens": max_new,
    }
    directory, record = common.new_run(
        spec, args.name, "temperature_sweep", identity, settings
    )
    rows = []
    try:
        with (directory / "generations.jsonl").open("w", encoding="utf-8") as stream:
            for item in prompts:
                ids = common.chat_prefix(tokenizer, item["text"])
                for temperature in temperatures:
                    for seed in seeds:
                        result = common.generate_reply(
                            engine, tokenizer, ids, temperature=temperature,
                            top_k=top_k, max_new_tokens=max_new, seed=seed,
                        )
                        row = {
                            "prompt_id": item["id"],
                            "prompt": item["text"],
                            "prompt_tokens": len(ids),
                            "temperature": temperature,
                            "seed": seed,
                            "top_k": top_k,
                            "max_new_tokens": max_new,
                            **result,
                        }
                        rows.append(row)
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        stream.flush()
                        print(
                            f"{item['id']:12} temp={temperature:<3} seed={seed} "
                            f"tokens={result['generated_tokens']:<3} "
                            f"stop={result['stop_reason']}"
                        )
        save_summary(directory, rows)
        record["completed_generations"] = len(rows)
        common.finish_run(directory, record, "complete")
    except BaseException:
        record["completed_generations"] = len(rows)
        common.finish_run(directory, record, "failed")
        raise
    print(f"Saved temperature experiment: {directory}")


if __name__ == "__main__":
    main()
