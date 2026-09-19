# Task 1: tokenization

This folder imports `RustBPETokenizer` from nanochat. It supports both the
published layout (`task1/` inside the nanochat repository) and the original
workspace layout (`task1/` beside a separate `nanochat/` checkout). All
experiment files belong here. Tokenizer training runs on the CPU.

## What we have done so far

Both tokenizers have been trained on the same 500,000,000-byte text sample,
and all seven evaluation examples have been run through both. All 14 full-text
encode/decode checks passed. See [the recorded results](results/summary.md) for
measurements, training times, and links to the saved logs and examples.

## Function overview

Start with these four steps: **download -> prepare -> train -> evaluate**.
Both tokenizers use the same training text so we can compare vocabulary sizes.

| Function | What it does |
| --- | --- |
| `download()` | Downloads the training data. |
| `prepare()` | Creates the text sample both tokenizers will learn from. |
| `train()` | Trains and saves the two tokenizers. |
| `evaluate()` | Compares their tokenizations and saves the results. |
| `main()` | Runs whichever step you choose in the terminal. |
| `sample_texts()` | Reads the saved training texts one at a time. |
| `tokenizer_class()` | Gets the tokenizer implementation from nanochat. |
| `write_json()` | Saves information to a JSON file. |
| `read_json()` | Reads information from a JSON file. |
| `sha256()` | Creates a file identifier to help detect changes. |
| `require_new()` | Prevents overwriting existing outputs. |
| `positive_int()` | Checks that a number you enter is a positive whole number. |

The helper functions support reading files, saving results, and checking inputs.

## Files

- `task1_tokenizers.py`: download, prepare, train, and evaluate commands.
- `requirements.txt`: the three direct dependencies, with versions taken from
  nanochat's `uv.lock`. Installation also resolves their transitive dependencies.
- `eval_samples.json`: fixed, AI-assisted starter examples for English, numbers,
  code, Dutch, Chinese, and Unicode. Review or expand these before your recorded
  evaluation, and keep the same examples for both tokenizers.
- `.gitignore`: excludes the environment, caches, data, and tokenizer binaries.
  Small files in `results/` remain suitable for version control.

Run the commands below from the directory containing `task1/`: the repository
root in the published layout, or the assignment root in the original workspace.
They create generated files only inside `task1/`.

## Environment

```bash
uv venv --python /usr/bin/python3 --cache-dir task1/cache/uv task1/.venv
uv pip install --python task1/.venv/bin/python --cache-dir task1/cache/uv -r task1/requirements.txt
task1/.venv/bin/python task1/task1_tokenizers.py --help
```

This is a tokenizer-only environment. Model training will need the additional
dependencies specified by nanochat. No model training is part of these commands.

## Run the experiments

The completed run used these commands. To repeat it, choose new output paths
as described below; existing completed outputs are protected. Run each command
separately and check its output before continuing:

```bash
task1/.venv/bin/python task1/task1_tokenizers.py download --num-shards 3 --revision 915333b4f8b8684f39aeaafea600fea6f43fb703
task1/.venv/bin/python task1/task1_tokenizers.py prepare
task1/.venv/bin/python task1/task1_tokenizers.py train
task1/.venv/bin/python task1/task1_tokenizers.py evaluate
```

1. **Download:** fetch the first three training shards from the CLIMBMix mirror
   named in nanochat's `nanochat/dataset.py`. Each shard gets a source URL and
   SHA-256 sidecar. The reserved validation shard is excluded. Downloads use
   network bandwidth and disk space. Use `--revision <dataset-commit>` to pin
   the upstream version; the default `main` can change. If the sample command
   reports insufficient text, increase `--num-shards`; existing verified shards
   are reused.
2. **Prepare:** create `data/sample.jsonl` and `data/sample.manifest.json`.
   Here **500 MB means 500,000,000 UTF-8 text bytes**, excluding JSON formatting
   and compression. As in the supplied trainer, documents are capped at 10,000
   characters. The sample uses sorted training shards and original row order.
   Its last document is truncated at a valid UTF-8 boundary, so the actual total
   can be up to three bytes below the target. Record the manifest's exact count
   and this sampling convention in your report. This is a deterministic prefix
   of the shuffled corpus, not a fresh random sample.
3. **Train:** use that exact saved sample for vocabularies 8,192 and 32,768,
   including nanochat's nine special tokens. Separate directories prevent one
   experiment replacing the other. Each includes the tokenizer, training time,
   dependency versions, nanochat revision/source hash, and sample manifest.
4. **Evaluate:** write `results/metrics.csv` and `results/tokenizations.json`.
   The script checks that both tokenizers used the same training sample and that
   every evaluation text survives an encode/decode round trip. It reports
   characters (Python Unicode code points), UTF-8 bytes, tokens, tokens per
   character, and bytes per token. Lower tokens per character means fewer tokens
   for that text. Individual token displays can contain replacement characters
   when the token is only part of a UTF-8 character; consult the stored raw bytes.

Completed samples, tokenizer directories, and result files are protected against
overwriting. To make another run, use new `--sample`, `--output-dir`, and/or
`--results-dir` paths as appropriate. Use the same `--output-dir` for training
and evaluation. Partial preparation/download files can be replaced by retries.

```text
task1/
├── .venv/                  # generated Python environment
├── cache/                  # generated package cache
├── data/
│   ├── shards/
│   ├── sample.jsonl
│   └── sample.manifest.json
├── tokenizer_8192/
│   ├── tokenizer.pkl
│   ├── token_bytes.json
│   └── training.json
├── tokenizer_32768/         # same output files
└── results/
    ├── metrics.csv
    └── tokenizations.json
```

`tokenizer.pkl` uses nanochat's existing save format. `token_bytes.json` records
raw byte lengths with zero for special tokens. For downstream model bpb
evaluation, convert this list to nanochat's `token_bytes.pt` (a CPU `torch.int32`
tensor) once PyTorch is installed; this helper does not claim its JSON is a
drop-in replacement for that tensor file.

## Evidence still needed for the report

- [x] Train both tokenizers on the documented 500 MB sample.
- [x] Record English token counts and tokens per character for comparison.
- [ ] Explain BPE versus word- and character-level tokenization.
- [ ] Construct an original merge-tree example and record pair counts at each
  merge. The training metadata does not contain learned merge frequencies.
- [ ] Explain embedding size, the softmax denominator, and rare-token trade-offs.
- [ ] Inspect numbers, code, and non-English examples and discuss observed cases.
- [ ] Write your own analysis in at most one page for Task 1, and disclose AI
  assistance as required by the assignment.

The published branch includes this folder at `task1/`. The dataset, virtual
environment, package caches, and tokenizer binaries remain local and are
excluded by `.gitignore`. Copies of the small sample manifest and training
records are saved in `results/` so the experiment evidence is available in Git.

The archived script and logs under `results/run_20260919T100207Z/` preserve
the original run, including its original workspace paths. Use the current
`task1_tokenizers.py` and the commands above when reproducing the experiment.
