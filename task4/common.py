"""Shared Task 4 plumbing: exact model selection, safe paths, and Nanochat inference.

The four runnable scripts import this module. Keeping one loader here prevents
chat, temperature tests, and benchmarks from accidentally using different
tokenizers or checkpoints. This file has no command-line entry point.
"""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
TASKS = ("ARC-Easy", "ARC-Challenge", "GSM8K")
sys.dont_write_bytecode = True


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def digest(path):
    hash_value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hash_value.update(chunk)
    return hash_value.hexdigest()


def now_utc():
    return datetime.now(timezone.utc).isoformat()


def path_from_repo(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO / path).resolve()


def run_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", value):
        raise ValueError("Run name must use only letters, digits, _ or -, and start with a letter or digit.")
    return value


def add_model_options(parser):
    parser.add_argument("--config", default=str(HERE / "config.json"), help="JSON settings file.")
    parser.add_argument("--checkpoint-dir", help="Directory containing model_XXXXXX.pt and meta_XXXXXX.json.")
    parser.add_argument("--tokenizer-dir", help="Directory containing the matching tokenizer.pkl.")
    parser.add_argument("--step", type=int, help="Exact checkpoint step; never selects latest implicitly.")
    parser.add_argument("--stage", choices=["base", "mid", "sft"], help="Expected checkpoint stage.")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--cache-dir", help="Ignored cache directory, including benchmark downloads.")
    parser.add_argument("--output-root", help="Where new results folders are created.")


def configuration(args):
    config = read_json(args.config)
    checkpoint = config["model"]
    generation = config["generation"]
    benchmark = config["benchmark"]
    spec = {
        "checkpoint_dir": path_from_repo(args.checkpoint_dir or checkpoint["checkpoint_dir"]),
        "tokenizer_dir": path_from_repo(args.tokenizer_dir or checkpoint["tokenizer_dir"]),
        "step": checkpoint["step"] if args.step is None else args.step,
        "stage": args.stage or checkpoint["stage"],
        "device": args.device or generation["device"],
        "dtype": args.dtype or generation["dtype"],
        "cache_dir": path_from_repo(args.cache_dir or HERE / "cache"),
        "output_root": path_from_repo(args.output_root or HERE / "results"),
    }
    if not isinstance(spec["step"], int) or spec["step"] < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer.")
    if spec["stage"] not in ("base", "mid", "sft"):
        raise ValueError("Stage must be base, mid, or sft.")
    if spec["device"] not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError("Device must be auto, cpu, mps, or cuda.")
    if spec["dtype"] not in ("auto", "float32", "float16", "bfloat16"):
        raise ValueError("Dtype must be auto, float32, float16, or bfloat16.")
    return config, spec, generation, benchmark


def verify_files(spec):
    """Read metadata and hashes before any model is loaded or output is written."""
    directory = spec["checkpoint_dir"]
    step = spec["step"]
    model_path = directory / f"model_{step:06d}.pt"
    meta_path = directory / f"meta_{step:06d}.json"
    tokenizer_path = spec["tokenizer_dir"] / "tokenizer.pkl"
    for path in (model_path, meta_path, tokenizer_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required file is missing: {path}")
    metadata = read_json(meta_path)
    if metadata.get("step") is not None and metadata["step"] != step:
        raise ValueError(f"Checkpoint metadata says step {metadata['step']}, not {step}.")
    if metadata.get("stage") is not None and metadata["stage"] != spec["stage"]:
        raise ValueError(
            f"Checkpoint metadata says stage {metadata['stage']}, not {spec['stage']}."
        )
    model_config = metadata["model_config"]
    if model_config["sequence_len"] < 2 or model_config["vocab_size"] < 256:
        raise ValueError("Checkpoint model configuration is invalid.")
    return {
        "stage": spec["stage"],
        "step": step,
        "checkpoint_dir": str(directory),
        "model_file": str(model_path),
        "model_sha256": digest(model_path),
        "metadata_file": str(meta_path),
        "tokenizer_dir": str(spec["tokenizer_dir"]),
        "tokenizer_sha256": digest(tokenizer_path),
        "model_config": model_config,
    }


def load_nanochat(spec, identity):
    """Use Nanochat's own checkpoint loader and Engine without editing either."""
    import torch

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Task 4 is a single-device runner; do not launch it with torchrun.")
    device_name = spec["device"]
    if device_name == "auto":
        device_name = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot use CUDA here.")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but PyTorch cannot use MPS here. Try --device cpu.")
    dtype = spec["dtype"]
    if dtype == "auto":
        dtype = (
            "bfloat16" if device_name == "cuda" and torch.cuda.is_bf16_supported()
            else "float32"
        )
    if device_name != "cuda" and dtype == "float16":
        raise ValueError("Use float32 for CPU/MPS inference, or explicitly select a supported dtype.")

    # Nanochat's get_tokenizer() expects BASE_DIR/tokenizer/tokenizer.pkl.
    # Link to the selected tokenizer in an ignored cache; do not copy/edit it.
    base_dir = spec["cache_dir"] / "tokenizers" / identity["tokenizer_sha256"][:16]
    base_dir.mkdir(parents=True, exist_ok=True)
    link = base_dir / "tokenizer"
    if link.is_symlink():
        if link.resolve() != spec["tokenizer_dir"]:
            raise ValueError(f"Existing tokenizer link points elsewhere: {link}")
    elif link.exists():
        if not link.is_dir() or digest(link / "tokenizer.pkl") != identity["tokenizer_sha256"]:
            raise ValueError(f"Existing tokenizer directory does not match: {link}")
    else:
        link.symlink_to(spec["tokenizer_dir"], target_is_directory=True)
    os.environ["NANOCHAT_BASE_DIR"] = str(base_dir)
    os.environ["NANOCHAT_DTYPE"] = dtype
    sys.path.insert(0, str(REPO))

    from nanochat.checkpoint_manager import build_model
    from nanochat.engine import Engine

    model, tokenizer, _ = build_model(
        str(spec["checkpoint_dir"]), spec["step"], torch.device(device_name), phase="eval"
    )
    if tokenizer.get_vocab_size() != identity["model_config"]["vocab_size"]:
        raise ValueError("Tokenizer vocabulary does not match the checkpoint.")
    identity["device"] = device_name
    identity["dtype"] = dtype
    identity["trainable_parameters"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    identity["python_version"] = sys.version.split()[0]
    identity["torch_version"] = torch.__version__
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=False
    )
    identity["nanochat_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
    return model, tokenizer, Engine(model, tokenizer)


def chat_prefix(tokenizer, message, history=None):
    """Render one user turn and the assistant-start marker using Nanochat IDs."""
    ids = list(history) if history is not None else [tokenizer.get_bos_token_id()]
    ids.extend((
        tokenizer.encode_special("<|user_start|>"),
        *tokenizer.encode(message),
        tokenizer.encode_special("<|user_end|>"),
        tokenizer.encode_special("<|assistant_start|>"),
    ))
    return ids


def context_check(model, ids, max_new_tokens):
    # Stay within the sequence length this checkpoint was trained on.
    if len(ids) + max_new_tokens > model.config.sequence_len:
        raise ValueError(
            f"Prompt has {len(ids)} tokens and reserves {max_new_tokens} reply tokens; "
            f"the trained context is {model.config.sequence_len}. Shorten the prompt, "
            "reduce --max-new-tokens, or use 'clear' in chat."
        )


def generate_reply(engine, tokenizer, prompt_ids, *, temperature, top_k, max_new_tokens, seed):
    """Use Nanochat Engine, including its KV cache and optional tool handling."""
    produced = []
    masks = []
    assistant_end = tokenizer.encode_special("<|assistant_end|>")
    bos = tokenizer.get_bos_token_id()
    started = time.perf_counter()
    for token_column, token_masks in engine.generate(
        prompt_ids, num_samples=1, max_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k, seed=seed,
    ):
        produced.append(int(token_column[0]))
        masks.append(int(token_masks[0]))
    elapsed = time.perf_counter() - started
    ended = bool(produced and produced[-1] in (assistant_end, bos))
    visible_ids = produced[:-1] if ended else produced
    return {
        "response": tokenizer.decode(visible_ids),
        "generated_ids": produced,
        "generated_tokens": len(produced),
        "sampled_tokens": sum(masks),
        "forced_tokens": len(masks) - sum(masks),
        "stop_reason": "end_token" if ended else "max_new_tokens",
        "seconds": elapsed,
    }


def generation_settings(args, generation):
    max_new = generation["max_new_tokens"] if args.max_new_tokens is None else args.max_new_tokens
    top_k = generation["top_k"] if args.top_k is None else args.top_k
    if max_new < 1 or top_k < 0:
        raise ValueError("max-new-tokens must be positive; top-k must be nonnegative.")
    return max_new, top_k


def load_prompts(path, selected_ids=None):
    items = read_json(path)["prompts"]
    if not items or len({item["id"] for item in items}) != len(items):
        raise ValueError("prompts.json needs at least one prompt and unique IDs.")
    for item in items:
        if not item["id"] or not isinstance(item["text"], str) or not item["text"].strip():
            raise ValueError("Each prompt needs an ID and nonempty text.")
    if selected_ids:
        unknown = set(selected_ids) - {item["id"] for item in items}
        if unknown:
            raise ValueError(f"Unknown prompt IDs: {', '.join(sorted(unknown))}")
        items = [item for item in items if item["id"] in selected_ids]
    return items


def new_run(spec, name, kind, identity, settings):
    directory = spec["output_root"] / run_name(name)
    directory.mkdir(parents=True, exist_ok=False)
    record = {
        "kind": kind, "name": name, "status": "running",
        "started_utc": now_utc(), "model": identity, "settings": settings,
    }
    write_json(directory / "run.json", record)
    return directory, record


def finish_run(directory, record, status):
    record["status"] = status
    record["finished_utc"] = now_utc()
    write_json(directory / "run.json", record)
