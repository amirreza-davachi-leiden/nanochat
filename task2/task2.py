"""Task 2: prepare data, run nanochat, and evaluate checkpoints.

Start with config.json, then prepare(), run(), and evaluate() below.
Sampling and plotting have their own small scripts.
Every function has a short explanation.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
NANOCHAT = ROOT.parent
if not (NANOCHAT / "nanochat" / "gpt.py").is_file():
    NANOCHAT = ROOT.parent / "nanochat"
ARTIFACTS = ROOT / "artifacts"
SCRIPT = Path(__file__).resolve()
VALIDATION_SHARD = "shard_06542.parquet"
DATA_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve"
sys.dont_write_bytecode = True



# Shared file and path helpers
# ----------------------------------------------------------------------


def read_json(path):
    """Load settings or recorded results from a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))



def write_json(path, value):
    """Save settings or results in a readable JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")



def sha256(path):
    """Identify a file's contents so runs can record exactly which inputs they used."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()



def run_environment():
    """Keep nanochat's generated files and caches inside task2."""
    env = os.environ.copy()
    env.update({
        "NANOCHAT_BASE_DIR": str(ARTIFACTS),
        "HF_HOME": str(ROOT / "cache" / "huggingface"),
        "TORCHINDUCTOR_CACHE_DIR": str(ROOT / "cache" / "inductor"),
        "MPLCONFIGDIR": str(ROOT / "cache" / "matplotlib"),
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
    })
    return env



def use_nanochat():
    """Make the local nanochat checkout and Task 2 artifact paths available."""
    if not (NANOCHAT / "nanochat" / "gpt.py").is_file():
        raise FileNotFoundError(f"Cannot find nanochat at {NANOCHAT}")
    os.environ.update(run_environment())
    sys.path.insert(0, str(NANOCHAT))



# 1. Prepare the tokenizer and dataset
# ----------------------------------------------------------------------


def copy_matching(source, destination):
    """Copy an input once, rejecting an existing copy with different contents."""
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise ValueError(f"Existing input differs: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)



def prepare_tokenizer(task1, vocab_size):
    """Reuse our trained tokenizer and create nanochat's token-byte tensor."""
    import torch
    from nanochat.tokenizer import RustBPETokenizer

    source = task1 / f"tokenizer_{vocab_size}"
    tokenizer = RustBPETokenizer.from_directory(str(source))
    if tokenizer.get_vocab_size() != vocab_size:
        raise ValueError("The saved tokenizer has the wrong vocabulary size.")
    special_ids = {tokenizer.encode_special(s) for s in tokenizer.get_special_tokens()}
    lengths = [
        0 if i in special_ids else len(tokenizer.decode_single_token_bytes(i))
        for i in range(vocab_size)
    ]
    if lengths != read_json(source / "token_bytes.json"):
        raise ValueError("Task 1's byte lengths do not match its tokenizer.")
    destination = ARTIFACTS / "tokenizer"
    copy_matching(source / "tokenizer.pkl", destination / "tokenizer.pkl")
    copy_matching(source / "training.json", destination / "training.json")
    tensor = torch.tensor(lengths, dtype=torch.int32)
    tensor_path = destination / "token_bytes.pt"
    if tensor_path.exists():
        if not torch.equal(torch.load(tensor_path, weights_only=True), tensor):
            raise ValueError("Existing token_bytes.pt does not match the tokenizer.")
    else:
        torch.save(tensor, tensor_path)
    return {
        "vocab_size": vocab_size, "source": str(source),
        "tokenizer_sha256": sha256(destination / "tokenizer.pkl"),
        "token_bytes_sha256": sha256(tensor_path),
        "training": read_json(source / "training.json"),
    }



def prepare_training_data(task1, destination, revision):
    """Reuse verified training shards through links instead of copying large files."""
    paths = sorted((task1 / "data" / "shards").glob("shard_*.parquet"))
    paths = [path for path in paths if path.name != VALIDATION_SHARD]
    if not paths:
        raise ValueError("No Task 1 training shards found; prepare Task 1 data first.")
    records = []
    for path in paths:
        provenance = read_json(path.with_suffix(".source.json"))
        expected_url = f"{DATA_URL}/{revision}/{path.name}"
        if provenance["url"] != expected_url or sha256(path) != provenance["sha256"]:
            raise ValueError(f"Dataset version/checksum mismatch: {path}")
        link = destination / path.name
        if link.is_symlink():
            if link.resolve() != path.resolve():
                raise ValueError(f"Existing dataset link points elsewhere: {link}")
        elif link.exists():
            raise ValueError(f"Expected a dataset link, found an existing file: {link}")
        else:
            link.symlink_to(path.resolve())
        records.append({"filename": path.name, **provenance})
    return records



def prepare_validation_data(destination, revision):
    """Download and record the separate shard used only for validation."""
    path = destination / VALIDATION_SHARD
    sidecar = path.with_suffix(".source.json")
    url = f"{DATA_URL}/{revision}/{path.name}"
    if path.exists():
        record = read_json(sidecar)
        if record["url"] != url or sha256(path) != record["sha256"]:
            raise ValueError("Existing validation data has a different source or checksum.")
        return {"filename": path.name, **record}
    partial = path.with_suffix(".partial")
    print(f"Downloading validation data: {url}", flush=True)
    with urlopen(url, timeout=60) as response, partial.open("wb") as stream:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            stream.write(chunk)
    record = {"url": url, "sha256": sha256(partial), "bytes": partial.stat().st_size}
    partial.replace(path)
    write_json(sidecar, record)
    return {"filename": path.name, **record}



def prepare(args):
    """Prepare the inputs and save a manifest describing what training will use."""
    use_nanochat()
    config = read_json(ROOT / "config.json")
    data_dir = ARTIFACTS / "base_data_climbmix"
    data_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = prepare_tokenizer(args.task1_dir.resolve(), config["vocab_size"])
    training = prepare_training_data(args.task1_dir.resolve(), data_dir, config["dataset_revision"])
    validation = prepare_validation_data(data_dir, config["dataset_revision"])
    expected = {record["filename"] for record in training} | {VALIDATION_SHARD}
    if {path.name for path in data_dir.glob("*.parquet")} != expected:
        raise ValueError("Unexpected Parquet files would change nanochat's train/validation split.")
    # Nanochat reserves the final sorted shard for validation.
    assert sorted(expected)[-1] == VALIDATION_SHARD
    write_json(ROOT / "results" / "preparation.json", {
        "dataset_revision": config["dataset_revision"], "tokenizer": tokenizer,
        "training_shards": training, "validation_shard": validation,
        "data_policy": "Model training reads original Parquet documents; Task 1's 500 MB limit and document cap do not apply.",
    })
    print(f"Prepared vocabulary {config['vocab_size']:,}, {len(training)} training shards, and one validation shard.")
    print(f"Saved input manifest: {ROOT / 'results' / 'preparation.json'}")



# 2. Run the existing nanochat trainer
# ----------------------------------------------------------------------


def build_command(settings, name):
    """Translate our configuration into nanochat's existing command-line options."""
    command = [sys.executable, "-u", "-B", str(SCRIPT), "_train"]
    for key, value in settings.items():
        command.append(f"--{key.replace('_', '-')}={value}")
    return command + [f"--model-tag={name}"]



def save_run_record(results, config, preparation, command, mode, compiled):
    """Record the inputs, software versions, and source code used for this run."""
    source_files = [
        "scripts/base_train.py", "nanochat/loss_eval.py", "nanochat/gpt.py",
        "nanochat/optim.py", "nanochat/dataloader.py", "nanochat/dataset.py",
        "nanochat/tokenizer.py", "nanochat/checkpoint_manager.py",
    ]
    versions = {name: importlib.metadata.version(name) for name in ("torch", "pyarrow", "rustbpe", "tiktoken", "matplotlib")}
    revision = subprocess.check_output(["git", "-C", str(NANOCHAT), "rev-parse", "HEAD"], text=True).strip()
    write_json(results / "config.json", config)
    write_json(results / "preparation.json", preparation)
    patch = subprocess.check_output(
        ["git", "-C", str(NANOCHAT), "diff", "HEAD", "--", *source_files], text=True
    )
    (results / "nanochat_changes.patch").write_text(patch)
    return {
        "status": "running", "mode": mode, "started_utc": datetime.now(timezone.utc).isoformat(),
        "command": command, "shell_command": shlex.join(command), "working_directory": str(NANOCHAT),
        "nanochat_revision": revision, "python": sys.version, "packages": versions,
        "source_sha256": {name: sha256(NANOCHAT / name) for name in source_files},
        "tokenizer_sha256": preparation["tokenizer"]["tokenizer_sha256"],
        "compile_enabled": compiled, "cpu_threads": 4,
        "task2_source_sha256": {path.name: sha256(path) for path in ROOT.glob("*.py")},
    }



def run_training(command, environment, log_path):
    """Run nanochat and save the same terminal output to a log file."""
    with log_path.open("x", encoding="utf-8") as log, log_path.with_name("training_progress.jsonl").open("x") as progress:
        with subprocess.Popen(
            command, cwd=NANOCHAT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        ) as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                    match = re.search(r"^step (\d+)/(\d+).*loss: ([\d.]+).*lrm: ([\d.]+).*dt: ([\d.]+)ms.*tok/sec: ([\d,]+)", line)
                    if match:
                        step, total, loss, multiplier, milliseconds, throughput = match.groups()
                        # Nanochat prints a smoothed loss from the final microbatch of each update.
                        progress.write(json.dumps({
                            "completed_updates": int(step) + 1, "total_updates": int(total),
                            "smoothed_last_microbatch_loss": float(loss),
                            "displayed_lr_multiplier": float(multiplier),
                            "step_seconds": float(milliseconds) / 1000,
                            "tokens_per_second": int(throughput.replace(",", "")),
                        }) + "\n")
                        progress.flush()
                return process.wait()
            except KeyboardInterrupt:
                process.terminate()
                process.wait()
                raise



def run(args):
    """Choose a run profile, train, and verify that the final checkpoint reloads."""
    config = read_json(ROOT / "config.json")
    settings = {**config["common"], **config["modes"][args.mode]}
    compiled = settings.pop("compile")
    if args.device:
        settings["device_type"] = args.device
    name = args.name or f"{args.mode}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("Use letters, numbers, underscores, and hyphens in the run name.")
    results = ROOT / "results" / name
    checkpoint = ARTIFACTS / "base_checkpoints" / name
    command = build_command(settings, name)
    if args.dry_run:
        print(f"Working directory: {NANOCHAT}")
        print(shlex.join(command))
        print(f"torch.compile enabled: {compiled}")
        return
    if results.exists() or checkpoint.exists():
        raise ValueError("That run name already exists. Choose a new name to preserve its outputs.")
    preparation = read_json(ROOT / "results" / "preparation.json")
    if preparation["tokenizer"]["vocab_size"] != config["vocab_size"]:
        raise ValueError("Configuration and prepared tokenizer differ; run preparation again.")
    for filename, key in (("tokenizer.pkl", "tokenizer_sha256"), ("token_bytes.pt", "token_bytes_sha256")):
        if sha256(ARTIFACTS / "tokenizer" / filename) != preparation["tokenizer"][key]:
            raise ValueError(f"Prepared tokenizer file changed: {filename}")

    environment = run_environment()
    environment["TASK2_RESULTS_DIR"] = str(results)
    # Disabling compilation makes a short CPU check practical.
    environment["TORCH_COMPILE_DISABLE"] = "0" if compiled else "1"
    os.environ.update(environment)
    import torch
    if settings["device_type"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Use the smoke profile here; install the GPU environment on a CUDA machine for the pilot/main run.")
    results.mkdir(parents=True)
    record = save_run_record(results, config, preparation, command, args.mode, compiled)
    record["device_name"] = torch.cuda.get_device_name(0) if settings["device_type"] == "cuda" else "CPU"
    write_json(results / "run.json", record)
    started = time.perf_counter()
    try:
        code = run_training(command, environment, results / "train.log")
        record["training_seconds"] = time.perf_counter() - started
        record["training_exit_code"] = code
        if code:
            raise RuntimeError(f"Training exited with code {code}; see {results / 'train.log'}")
        subprocess.run(
            [sys.executable, "-B", str(ROOT / "sample_completions.py"), "--run", name, "--verify-only"],
            cwd=ROOT, env=environment, check=True,
        )
        subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "evaluate", "--run", name,
             "--device", settings["device_type"]], cwd=ROOT, env=environment, check=True,
        )
        subprocess.run(
            [sys.executable, "-B", str(ROOT / "plot_results.py"), "--run", name],
            cwd=ROOT, env=environment, check=True,
        )
        record["status"] = "complete"
    except BaseException:
        record["status"] = "failed"
        raise
    finally:
        record["total_seconds_including_evaluation"] = time.perf_counter() - started
        record["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(results / "run.json", record)
    print(f"Saved run and verified checkpoint: {results}")



# Record the settings nanochat calculated
# ----------------------------------------------------------------------


def run_nanochat():
    """Let nanochat train normally, then save the settings it actually used."""
    use_nanochat()
    # run_module executes the existing script; it does not edit or replace its code.
    state = runpy.run_module("scripts.base_train", run_name="__main__")
    args = state["args"]
    optimizer = state["optimizer"]
    results = os.environ["TASK2_RESULTS_DIR"]
    write_json(os.path.join(results, "resolved_config.json"), {
        "model": state["model_config_kwargs"], "arguments": state["user_config"],
        "seed": 42, "parameter_counts": state["param_counts"],
        "scaling_parameters": state["num_scaling_params"],
        "target_tokens": state["target_tokens"], "total_tokens": state["total_tokens"],
        "num_iterations": state["num_iterations"], "total_batch_size": state["total_batch_size"],
        "gradient_accumulation_steps": state["grad_accum_steps"], "world_size": state["ddp_world_size"],
        "initial_learning_rates": [group["initial_lr"] for group in optimizer.param_groups],
        "optimizer_group_kinds": [group["kind"] for group in optimizer.param_groups],
        "batch_lr_scale": state["batch_lr_scale"], "scaled_weight_decay": state["weight_decay_scaled"],
        "schedule": {
            "warmup_steps": args.warmup_steps, "warmdown_ratio": args.warmdown_ratio,
            "final_lr_frac": args.final_lr_frac,
            "first_update_multiplier": state["get_lr_multiplier"](0),
            "last_update_multiplier": state["get_lr_multiplier"](state["num_iterations"] - 1),
        },
        "compute_dtype": str(state["COMPUTE_DTYPE"]), "device": str(state["device"]),
        "checkpoint_directory": state["checkpoint_dir"],
        "peak_gpu_memory_bytes": state["get_max_memory"]() if state["device_type"] == "cuda" else None,
    })



# 3. Evaluate saved checkpoints
# ----------------------------------------------------------------------


class LossRecorder:
    """Observe ordinary loss while nanochat calculates its original bpb metric."""

    def __init__(self, model, token_bytes):
        """Keep the model and counters for a single evaluation."""
        self.model = model
        self.token_bytes = token_bytes
        self.loss_sum = 0.0
        self.target_tokens = 0
        self.text_bytes = 0

    def get_device(self):
        """Tell nanochat's evaluator where the model runs."""
        return self.model.get_device()

    def __call__(self, inputs, targets, loss_reduction):
        """Return the model's unchanged losses and count valid targets."""
        losses = self.model(inputs, targets, loss_reduction=loss_reduction)
        valid = targets >= 0
        self.loss_sum += losses.reshape_as(targets)[valid].sum().item()
        self.target_tokens += valid.sum().item()
        self.text_bytes += self.token_bytes[targets[valid]].sum().item()
        return losses



def measure(model, batches, token_bytes):
    """Reuse nanochat's bpb evaluator and also return mean cross-entropy loss."""
    from nanochat.loss_eval import evaluate_bpb
    recorder = LossRecorder(model, token_bytes)
    device_batches = ((x.to(model.get_device()), y.to(model.get_device())) for x, y in batches)
    bpb = evaluate_bpb(recorder, device_batches, len(batches), token_bytes)
    return {
        "loss": recorder.loss_sum / recorder.target_tokens if recorder.target_tokens else float("inf"),
        "bpb": bpb, "target_tokens": recorder.target_tokens, "text_bytes": recorder.text_bytes,
    }



def fixed_batches(tokenizer, settings, split):
    """Keep one deterministic data prefix for every checkpoint evaluation."""
    from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
    batch_size, length = settings["device_batch_size"], settings["max_seq_len"]
    count = settings["eval_tokens"] // (batch_size * length)
    if count < 1:
        raise ValueError("eval_tokens must cover at least one complete batch.")
    loader = tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, batch_size, length, split=split, device="cpu"
    )
    batches, digest = [], hashlib.sha256()
    for _ in range(count):
        x, y = next(loader)
        # The loader reuses its buffers, so copies preserve the fixed sample.
        batches.append((x.clone(), y.clone()))
        digest.update(x.numpy().tobytes())
        digest.update(y.numpy().tobytes())
    return batches, {
        "batches": count, "target_tokens": count * batch_size * length,
        "sha256": digest.hexdigest(), "hash_format": "Each batch's x then y, contiguous int64 bytes.",
    }



def evaluate(args):
    """Evaluate saved checkpoints and write the table used by plot()."""
    use_nanochat()
    if args.device == "cpu":
        os.environ["NANOCHAT_DTYPE"] = "float32"
    import torch
    from nanochat.checkpoint_manager import build_model
    from nanochat.tokenizer import get_tokenizer, get_token_bytes

    torch.set_num_threads(4)
    device = torch.device(args.device)
    results = ROOT / "results" / args.run
    resolved = read_json(results / "resolved_config.json")
    preparation = read_json(results / "preparation.json")
    for filename, key in (("tokenizer.pkl", "tokenizer_sha256"), ("token_bytes.pt", "token_bytes_sha256")):
        if sha256(ARTIFACTS / "tokenizer" / filename) != preparation["tokenizer"][key]:
            raise ValueError(f"The prepared tokenizer file changed: {filename}")
    tokenizer = get_tokenizer()
    token_bytes = get_token_bytes(device=device)
    samples, sample_records = {}, {}
    for split in ("train", "val"):
        samples[split], sample_records[split] = fixed_batches(tokenizer, resolved["arguments"], split)
    write_json(results / "evaluation_samples.json", {
        "device": args.device, "samples": sample_records,
        "policy": "Same fixed prefix of each split at every checkpoint; separate from the training iterator.",
    })
    directory = ARTIFACTS / "base_checkpoints" / args.run
    checkpoints = sorted(directory.glob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError("No saved checkpoints found.")
    with (results / "metrics.jsonl").open("w", encoding="utf-8") as output:
        for checkpoint in checkpoints:
            started = time.perf_counter()
            step = int(checkpoint.stem.split("_")[-1])
            model, _, metadata = build_model(str(directory), step, device, phase="eval")
            row = {
                "event": "evaluation", "step": step,
                "tokens_seen": step * metadata["total_batch_size"],
                "checkpoint_sha256": sha256(checkpoint),
            }
            for split, label in (("train", "train"), ("val", "validation")):
                metrics = measure(model, samples[split], token_bytes)
                row.update({f"{label}_{key}": value for key, value in metrics.items()})
            row["evaluation_seconds"] = time.perf_counter() - started
            output.write(json.dumps(row, allow_nan=False) + "\n")
            output.flush()
            print(f"Step {step}: train bpb={row['train_bpb']:.4f}, validation bpb={row['validation_bpb']:.4f}")
            del model
    print(f"Saved checkpoint evaluations: {results / 'metrics.jsonl'}")


# Optional checks for the loss calculations

def check_metrics(args):
    """Run the three existing arithmetic checks without a separate test file."""
    import math
    import pytest
    import torch
    use_nanochat()
    from nanochat.loss_eval import evaluate_bpb

    class FixedLossModel:
        """Supply known losses so the metric can be checked by hand."""

        def get_device(self):
            """Keep this small arithmetic check on CPU."""
            return torch.device("cpu")

        def __call__(self, inputs, targets, loss_reduction):
            """Return losses supplied by the test instead of running a neural network."""
            assert loss_reduction == "none"
            return inputs



    def test_loss_and_bpb_treat_special_and_ignored_targets_correctly():
        """Bpb excludes zero-byte tokens; ordinary loss includes valid special tokens."""
        # Token 0 is special; tokens 1 and 2 represent two and three bytes.
        byte_lengths = torch.tensor([0, 2, 3])
        batches = [(torch.tensor([[2., 7., 99., 3.]]), torch.tensor([[1, 0, -1, 2]]))]
        metrics = measure(FixedLossModel(), batches, byte_lengths)
        assert metrics["loss"] == pytest.approx((2 + 7 + 3) / 3)
        assert metrics["bpb"] == pytest.approx((2 + 3) / (math.log(2) * 5))
        assert metrics["target_tokens"] == 3
        assert metrics["text_bytes"] == 5
        assert evaluate_bpb(FixedLossModel(), batches, 1, byte_lengths) == pytest.approx(metrics["bpb"])



    def test_bpb_uses_total_bytes_instead_of_averaging_batch_ratios():
        """Unequal byte counts must be combined before dividing the total loss."""
        byte_lengths = torch.tensor([0, 1, 9])
        batches = [
            (torch.tensor([[2.]]), torch.tensor([[1]])),
            (torch.tensor([[6.]]), torch.tensor([[2]])),
        ]
        metrics = measure(FixedLossModel(), batches, byte_lengths)
        assert metrics["loss"] == pytest.approx(4.)
        assert metrics["bpb"] == pytest.approx(8 / (math.log(2) * 10))
        assert metrics["target_tokens"] == 2
        assert metrics["text_bytes"] == 10



    def test_no_text_bytes_is_not_reported_as_zero_loss():
        """Special-only targets have ordinary loss but no defined text bpb."""
        batches = [(torch.tensor([[2.]]), torch.tensor([[0]]))]
        metrics = measure(FixedLossModel(), batches, torch.tensor([0]))
        assert metrics["loss"] == 2.
        assert math.isinf(metrics["bpb"])


    test_loss_and_bpb_treat_special_and_ignored_targets_correctly()
    test_bpb_uses_total_bytes_instead_of_averaging_batch_ratios()
    test_no_text_bytes_is_not_reported_as_zero_loss()
    print("All three metric checks passed.")


# Command-line entry point
# ----------------------------------------------------------------------

def main():
    """Choose which step to perform: prepare, run, evaluate, or check."""
    # A child process gives nanochat its own arguments and Torch settings.
    if sys.argv[1:2] == ["_train"]:
        sys.argv.pop(1)
        run_nanochat()
        return

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("prepare", help="Prepare the Task 1 tokenizer and CLIMBMix shards.")
    command.add_argument("--task1-dir", type=Path, default=ROOT.parent / "task1")
    command.set_defaults(action=prepare)

    command = commands.add_parser("run", help="Train with nanochat, then evaluate and plot.")
    command.add_argument("--mode", choices=["smoke", "pilot", "train"], default="pilot")
    command.add_argument("--name", help="New run name; defaults to mode plus a timestamp.")
    command.add_argument("--device", choices=["cpu", "cuda"], help="Override the profile's device.")
    command.add_argument("--dry-run", action="store_true", help="Print the command without training.")
    command.set_defaults(action=run)

    command = commands.add_parser("evaluate", help="Measure train/validation loss and bpb at checkpoints.")
    command.add_argument("--run", required=True)
    command.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    command.set_defaults(action=evaluate)

    command = commands.add_parser("check", help="Check the loss and bpb calculations.")
    command.set_defaults(action=check_metrics)

    args = parser.parse_args()
    args.action(args)


if __name__ == "__main__":
    main()
