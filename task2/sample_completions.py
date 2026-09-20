"""Reload a Task 2 checkpoint and save five raw text completions."""

import argparse
import os

from task2 import ROOT, ARTIFACTS, read_json, write_json, sha256, use_nanochat


def load_run(name, device):
    """Load the final checkpoint belonging to the named run."""
    from nanochat.checkpoint_manager import build_model
    results = ROOT / "results" / name
    resolved = read_json(results / "resolved_config.json")
    step = resolved["num_iterations"]
    directory = ARTIFACTS / "base_checkpoints" / name
    model, tokenizer, metadata = build_model(str(directory), step, device, phase="eval")
    expected = read_json(results / "preparation.json")["tokenizer"]["tokenizer_sha256"]
    if sha256(ARTIFACTS / "tokenizer" / "tokenizer.pkl") != expected:
        raise ValueError("The prepared tokenizer differs from this run's tokenizer.")
    if sum(p.numel() for p in model.parameters()) != resolved["parameter_counts"]["total"]:
        raise ValueError("Reloaded parameter count does not match the recorded model.")
    return model, tokenizer, results, directory / f"model_{step:06d}.pt"


def check_reload(model, tokenizer):
    """Verify that the reloaded model produces finite next-token scores."""
    import torch
    ids = tokenizer.encode("A small test", prepend=tokenizer.get_bos_token_id())
    inputs = torch.tensor([ids], device=model.get_device())
    with torch.no_grad():
        if not torch.isfinite(model(inputs)).all().item():
            raise ValueError("Reloaded model produced non-finite scores.")


def generate_examples(model, tokenizer, settings):
    """Generate continuations for five fixed prompts without chat formatting."""
    examples = []
    for index, prompt in enumerate(settings["prompts"]):
        ids = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
        # GPT.generate gives raw continuations; the chat engine also handles tool calls.
        continuation_ids = list(model.generate(
            ids, max_tokens=settings["max_new_tokens"],
            temperature=settings["temperature"], top_k=settings["top_k"], seed=settings["seed"] + index,
        ))
        continuation = tokenizer.decode(continuation_ids)
        examples.append({
            "prompt": prompt, "completion": continuation, "full_text": prompt + continuation,
            "prompt_ids": ids, "completion_ids": continuation_ids, "seed": settings["seed"] + index,
        })
    return examples


def main():
    """Check a saved checkpoint, or record its five completions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Run name under task2/results/.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, help="Override generation length, e.g. for a smoke check.")
    args = parser.parse_args()
    use_nanochat()
    if args.device == "cpu":
        os.environ["NANOCHAT_DTYPE"] = "float32"
    import torch
    torch.set_num_threads(4)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable.")
    output = ROOT / "results" / args.run / "completions.json"
    if not args.verify_only and output.exists():
        parser.error("Completions already exist for this run; preserve the recorded examples.")
    model, tokenizer, results, checkpoint = load_run(args.run, torch.device(args.device))
    check_reload(model, tokenizer)
    write_json(results / "reload_check.json", {
        "passed": True, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "device": args.device, "parameters": sum(p.numel() for p in model.parameters()),
    })
    print("Checkpoint reload and finite-output check passed.")
    if args.verify_only:
        return
    settings = read_json(results / "config.json")["sampling"]
    if args.max_new_tokens is not None:
        if args.max_new_tokens <= 0:
            parser.error("--max-new-tokens must be positive.")
        settings["max_new_tokens"] = args.max_new_tokens
    examples = generate_examples(model, tokenizer, settings)
    write_json(output, {
        "run": args.run, "mode": read_json(results / "run.json")["mode"],
        "checkpoint_sha256": sha256(checkpoint), "settings": settings, "device": args.device,
        "note": "Raw prompts with a document BOS token; no system prompt or chat template.",
        "examples": examples,
    })
    print(f"Saved {len(examples)} completions: {output}")


if __name__ == "__main__":
    main()
