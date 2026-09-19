"""Task 1: download text, prepare a sample, train two tokenizers, and compare them.

Run --help to see the available commands.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
NANOCHAT = ROOT.parent / "nanochat"
VOCAB_SIZES = [8192, 32768]
SHARD_URL = "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve"
VALIDATION_SHARD = "shard_06542.parquet"

sys.dont_write_bytecode = True
sys.path.insert(0, str(NANOCHAT))


def sha256(path):
    """Create a file identifier so we can check whether its contents changed."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    """Save information to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def read_json(path):
    """Read information from a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def require_new(*paths):
    """Stop if an output already exists, so earlier results are not overwritten."""
    for path in paths:
        if path.exists():
            raise ValueError(f"Already exists: {path}. Use a new output path to preserve this run.")


def positive_int(value):
    """Check that a command-line number is a whole number greater than zero."""
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def tokenizer_class():
    """Get nanochat's tokenizer so we can use it in this script."""
    if not (NANOCHAT / "nanochat" / "tokenizer.py").is_file():
        raise ValueError(f"Expected the nanochat checkout at {NANOCHAT}")
    from nanochat.tokenizer import RustBPETokenizer
    return RustBPETokenizer


def download(args):
    """Download the dataset files needed for training."""
    if args.num_shards > 6542:
        raise ValueError("There are 6542 training shards; shard 06542 is reserved for validation.")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for index in range(args.num_shards):
        path = args.data_dir / f"shard_{index:05d}.parquet"
        source_path = path.with_suffix(".source.json")
        url = f"{SHARD_URL}/{args.revision}/{path.name}"
        if path.exists():
            if not source_path.exists() or read_json(source_path)["url"] != url:
                raise ValueError(f"Cannot verify the source of {path}; choose another data directory.")
            if sha256(path) != read_json(source_path)["sha256"]:
                raise ValueError(f"Checksum mismatch: {path}")
            print(f"Already downloaded and verified: {path.name}", flush=True)
            continue
        temporary = path.with_suffix(".partial")
        print(f"Downloading {url}", flush=True)
        with urlopen(url, timeout=60) as response, temporary.open("wb") as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(chunk)
        metadata = {"url": url, "sha256": sha256(temporary), "bytes": temporary.stat().st_size}
        temporary.replace(path)
        write_json(source_path, metadata)
        print(f"Saved {path.name}: {metadata['bytes']:,} bytes", flush=True)


def prepare(args):
    """Create the shared text sample used to train both tokenizers."""
    import pyarrow.parquet as pq

    sample = args.sample
    manifest_path = sample.with_suffix(".manifest.json")
    require_new(sample, manifest_path)
    paths = sorted(p for p in args.data_dir.glob("shard_*.parquet") if p.name != VALIDATION_SHARD)
    if not paths:
        raise ValueError(f"No CLIMBMix training shards found in {args.data_dir}. Run download first.")
    sample.parent.mkdir(parents=True, exist_ok=True)
    temporary = sample.with_suffix(".partial")
    nbytes = nchars = ndocs = 0
    sources = []
    reached_limit = False
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for path in paths:
            source = {"filename": path.name, "sha256": sha256(path)}
            provenance = path.with_suffix(".source.json")
            if provenance.exists():
                source["download"] = read_json(provenance)
            sources.append(source)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=256, columns=["text"]):
                for text in batch.column(0).to_pylist():
                    if not isinstance(text, str) or not text:
                        continue
                    text = text[:args.doc_cap]
                    raw = text.encode("utf-8")
                    remaining = args.max_bytes - nbytes
                    if len(raw) >= remaining:
                        text = raw[:remaining].decode("utf-8", errors="ignore")
                        reached_limit = True
                    if text:
                        stream.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        nbytes += len(text.encode("utf-8"))
                        nchars += len(text)
                        ndocs += 1
                    if reached_limit:
                        break
                if reached_limit:
                    break
            if reached_limit:
                break
    if not reached_limit:
        raise ValueError(
            f"Only {nbytes:,} text bytes available; need {args.max_bytes:,}. "
            "Download more shards and run prepare again. The partial sample is not used for training."
        )
    manifest = {
        "requested_text_bytes": args.max_bytes, "text_bytes": nbytes,
        "characters": nchars, "documents": ndocs, "doc_cap_characters": args.doc_cap,
        "sample_sha256": sha256(temporary), "sources": sources,
        "sampling": "Sorted training shards, row order, capped documents, UTF-8-safe final prefix.",
        "byte_definition": "UTF-8 bytes of text fields only; 1 MB = 1,000,000 bytes.",
    }
    temporary.replace(sample)
    write_json(manifest_path, manifest)
    print(f"Sample: {nbytes:,} text bytes, {ndocs:,} documents; SHA256 {manifest['sample_sha256']}")


def sample_texts(path):
    """Read the saved training texts one at a time."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)["text"]


def train(args):
    """Train and save a tokenizer for each vocabulary size."""
    cls = tokenizer_class()
    manifest = read_json(args.sample.with_suffix(".manifest.json"))
    if sha256(args.sample) != manifest["sample_sha256"]:
        raise ValueError("Sample checksum differs from its manifest; do not train on a changed sample.")
    if len(set(args.vocab_sizes)) != len(args.vocab_sizes) or min(args.vocab_sizes) < 265:
        raise ValueError("Use distinct vocabulary sizes of at least 265 (256 bytes + 9 special tokens).")
    outputs = [args.output_dir / f"tokenizer_{size}" for size in args.vocab_sizes]
    require_new(*outputs)
    revision = subprocess.run(
        ["git", "-C", str(NANOCHAT), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip() or "unknown"
    versions = {name: importlib.metadata.version(name) for name in ("rustbpe", "tiktoken", "pyarrow")}
    for size, output in zip(args.vocab_sizes, outputs):
        print(f"Training vocabulary {size:,} on the recorded sample...", flush=True)
        started = time.perf_counter()
        tokenizer = cls.train_from_iterator(sample_texts(args.sample), size)
        elapsed = time.perf_counter() - started
        if tokenizer.get_vocab_size() != size:
            raise ValueError(f"Trainer produced {tokenizer.get_vocab_size()} tokens, expected {size}.")
        for text in ["Hello, world! 12345", "café 中文 🌍", "def f(x):\n    return x + 1\n"]:
            if tokenizer.decode(tokenizer.encode(text)) != text:
                raise ValueError("Tokenizer round-trip failed.")
        tokenizer.save(str(output))
        special_ids = {tokenizer.encode_special(s) for s in tokenizer.get_special_tokens()}
        byte_lengths = [
            0 if i in special_ids else len(tokenizer.decode_single_token_bytes(i))
            for i in range(size)
        ]
        write_json(output / "token_bytes.json", byte_lengths)
        write_json(output / "training.json", {
            "vocab_size": size, "special_tokens": len(special_ids), "training_seconds": elapsed,
            "sample": manifest, "nanochat_revision": revision,
            "tokenizer_source_sha256": sha256(NANOCHAT / "nanochat" / "tokenizer.py"),
            "python": sys.version, "packages": versions,
        })
        print(f"Saved {output}; training took {elapsed:.2f}s", flush=True)


def evaluate(args):
    """Compare how both tokenizers split the example texts and save the results."""
    cls = tokenizer_class()
    samples = read_json(args.examples)
    if not isinstance(samples, dict) or not samples or any(not isinstance(t, str) or not t for t in samples.values()):
        raise ValueError("Examples must be a nonempty JSON object mapping names to nonempty strings.")
    metrics_path = args.results_dir / "metrics.csv"
    examples_path = args.results_dir / "tokenizations.json"
    require_new(metrics_path, examples_path)
    rows, examples = [], []
    sample_hash = None
    for size in args.vocab_sizes:
        directory = args.output_dir / f"tokenizer_{size}"
        metadata = read_json(directory / "training.json")
        current_hash = metadata["sample"]["sample_sha256"]
        if sample_hash is not None and current_hash != sample_hash:
            raise ValueError("Tokenizers were trained on different samples.")
        sample_hash = current_hash
        tokenizer = cls.from_directory(str(directory))
        if tokenizer.get_vocab_size() != size or metadata["vocab_size"] != size:
            raise ValueError(f"Vocabulary does not match its directory/metadata: {directory}")
        for name, text in samples.items():
            ids = tokenizer.encode(text)
            if tokenizer.decode(ids) != text:
                raise ValueError(f"Round-trip failed for {name}, vocabulary {size}.")
            nbytes, nchars, ntokens = len(text.encode("utf-8")), len(text), len(ids)
            rows.append({
                "vocab_size": size, "sample": name, "characters": nchars, "utf8_bytes": nbytes,
                "tokens": ntokens, "tokens_per_character": ntokens / nchars,
                "bytes_per_token": nbytes / ntokens, "round_trip": True,
            })
            examples.append({
                "vocab_size": size, "sample": name, "text": text, "token_ids": ids,
                "token_bytes_hex": [tokenizer.decode_single_token_bytes(i).hex() for i in ids],
                "token_pieces_display": [tokenizer.decode([i]) for i in ids],
            })
    args.results_dir.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(examples_path, {
        "training_sample_sha256": sample_hash, "evaluation_samples_sha256": sha256(args.examples),
        "note": "Individual token displays may contain replacement characters; raw hex preserves exact bytes.",
        "examples": examples,
    })
    print(f"Saved {metrics_path} and {examples_path}")


def main():
    """Read the command you entered and run the selected step."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("download", help="Download CLIMBMix training shards (network and disk usage).")
    p.add_argument("--num-shards", type=positive_int, default=3)
    p.add_argument("--revision", default="main", help="Hugging Face dataset revision; use a commit for exact replay.")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "shards")
    p.set_defaults(func=download)
    p = sub.add_parser("prepare", help="Create the shared 500 MB UTF-8 text sample and manifest.")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "shards")
    p.add_argument("--sample", type=Path, default=ROOT / "data" / "sample.jsonl")
    p.add_argument("--max-bytes", type=positive_int, default=500_000_000)
    p.add_argument("--doc-cap", type=positive_int, default=10_000)
    p.set_defaults(func=prepare)
    p = sub.add_parser("train", help="Train separate 8,192- and 32,768-token BPEs on the shared sample.")
    p.add_argument("--sample", type=Path, default=ROOT / "data" / "sample.jsonl")
    p.add_argument("--output-dir", type=Path, default=ROOT)
    p.add_argument("--vocab-sizes", type=positive_int, nargs="+", default=VOCAB_SIZES)
    p.set_defaults(func=train)
    p = sub.add_parser("evaluate", help="Compare token counts and save inspectable tokenization examples.")
    p.add_argument("--output-dir", type=Path, default=ROOT)
    p.add_argument("--vocab-sizes", type=positive_int, nargs="+", default=VOCAB_SIZES)
    p.add_argument("--examples", type=Path, default=ROOT / "eval_samples.json")
    p.add_argument("--results-dir", type=Path, default=ROOT / "results")
    p.set_defaults(func=evaluate)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\nSee task1/README.md for setup and usage.\n")


if __name__ == "__main__":
    main()
