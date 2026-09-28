"""Evaluate one exact checkpoint on ARC-Easy, ARC-Challenge, and GSM-8K.

This calls Nanochat's existing benchmark evaluator. A small --max-problems
run checks the setup; omit it for reportable full-split scores.
"""

import argparse
import json
import time

import common


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common.add_model_options(parser)
    parser.add_argument("--name", required=True, help="New, unique results folder name.")
    parser.add_argument("--max-problems", type=int, help="Subset setup check; omit for full test splits.")
    parser.add_argument("--dry-run", action="store_true", help="Show paths/protocol without downloading data.")
    args = parser.parse_args()

    _, spec, _, settings = common.configuration(args)
    identity = common.verify_files(spec)
    if args.max_problems is not None and args.max_problems < 1:
        raise ValueError("--max-problems must be positive.")
    protocol = {**settings, "max_problems": args.max_problems, "tasks": list(common.TASKS)}
    if protocol["num_samples"] != 1 or protocol["temperature"] != 0:
        raise ValueError("Keep one greedy answer for Task 3 benchmark comparability.")
    if args.dry_run:
        print(json.dumps({
            "model": identity,
            "protocol": protocol,
            "output": str(spec["output_root"] / common.run_name(args.name)),
            "note": "A full run may download test data and take substantial time.",
        }, indent=2))
        return

    model, tokenizer, engine = common.load_nanochat(spec, identity)
    from scripts.chat_eval import run_chat_eval
    from tasks.arc import ARC
    from tasks.gsm8k import GSM8K

    directory, record = common.new_run(spec, args.name, "benchmark", identity, protocol)
    scores = {}
    try:
        for task in common.TASKS:
            started = time.perf_counter()
            accuracy = run_chat_eval(
                task, model, tokenizer, engine, **settings,
                max_problems=args.max_problems,
            )
            dataset = (
                GSM8K(subset="main", split="test") if task == "GSM8K"
                else ARC(subset=task, split="test")
            )
            count = (
                len(dataset) if args.max_problems is None
                else min(len(dataset), args.max_problems)
            )
            scores[task] = {
                "accuracy": accuracy,
                "percent": round(accuracy * 100, 4),
                "evaluated_examples": count,
                "seconds": time.perf_counter() - started,
            }
            # Save each completed task immediately; a long run may be interrupted.
            common.write_json(directory / "benchmark.json", {
                "model": identity, "protocol": protocol,
                "scores": scores, "complete": False,
            })
            print(f"{task}: {scores[task]['percent']:.2f}% on {count} examples")
        common.write_json(directory / "benchmark.json", {
            "model": identity, "protocol": protocol,
            "scores": scores, "complete": True,
        })
        common.finish_run(directory, record, "complete")
    except BaseException:
        common.finish_run(directory, record, "failed")
        raise
    print(f"Saved benchmark results: {directory}")


if __name__ == "__main__":
    main()
