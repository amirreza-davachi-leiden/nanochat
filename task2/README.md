# Task 2 — depth-2 pretraining

We use nanochat's existing model, trainer, optimizer, data loader, checkpoint
loader, and bpb evaluator. **No nanochat source files are changed.**
This folder prepares inputs, launches training, and records evidence for Task 2.

## What the assignment asks

See section 5.6, printed page 88 of the course assignment (Ass1Build.pdf):

1. **Depth:** explain what depth controls and how other settings follow from it.
   State layers, heads, embedding width, total trainable parameters, and training
   tokens. Discuss the benefits and limitations of the single-dial design.
2. **Scaling laws:** calculate a Chinchilla training-token budget, compare it with
   nanochat's horizon, and explain under-training and over-training.
3. **Loss analysis:** show training and validation **bits-per-byte** curves;
   discuss their shape, flattening, and the gap between them.
4. **Qualitative inspection:** generate five raw completions with no system
   prompt and discuss what the pretrained model can do.

Deliverables are the curves, a saved checkpoint, and about two report pages.
The main run is complete. Evidence and explanations are recorded in results/README.md.

## Simple layout

There are **three Python files**, with short explanations on every function:

| File | Purpose |
|---|---|
| [task2.py](task2.py) | Preparation, training, checkpoint evaluation, and metric checks. |
| [sample_completions.py](sample_completions.py) | Load a checkpoint and save five raw text completions. |
| [plot_results.py](plot_results.py) | Read the measurements and draw training/validation curves. |

[config.json](config.json) keeps our settings in one place, including depth 2.
The main functions to read first are:

| Function | What it does |
|---|---|
| **prepare()** in task2.py | Reuses the Task 1 tokenizer and prepares training/validation data. |
| **run()** in task2.py | Runs nanochat, records its settings, then evaluates and plots checkpoints. |
| **evaluate()** in task2.py | Measures training/validation loss and bpb on fixed samples. |
| **generate_examples()** in sample_completions.py | Produces five continuations from fixed prompts. |
| **save_plot()** in plot_results.py | Draws the two bpb curves. |

The remaining functions are helpers for these steps, with short docstrings.
The optional **check** command runs our three metric checks. Nanochat still
supplies the model and training loop.

## Our chosen settings

We keep Task 1's **32,768-token tokenizer** fixed for model training.

| Setting | Main run | Where it comes from |
|---|---:|---|
| Transformer layers | 2 | Our depth choice |
| Embedding width | 128 | Nanochat derives it from depth, aspect ratio, and head dimension |
| Attention heads per layer | 1 | Width divided by head dimension |
| Total trainable parameters | 12,976,170 | Counted from the actual model |
| Context length | 512 tokens | Our explicit override |
| Sequences per device batch | 2 | Our explicit override; verified on the RTX 4060 |
| Attention pattern | Full attention (L) | Our explicit choice for the SDPA fallback |
| Token budget | Automatic, ratio 12 | Nanochat's existing horizon rule |
| Global batch size | Automatic | Nanochat's existing scaling rule |
| Checkpoint interval | 25 updates, plus final | Our recording choice |
| Evaluation size | 16,384 target tokens per split | Our recording choice |

Depth does **not** independently determine every setting. Some settings are
derived from model size; others have defaults or explicit overrides.

Nanochat's horizon uses **4,587,532 scaling parameters** (transformer matrices
plus output matrix), which differs from the total trainable count. Its target
is therefore 55,050,384 tokens. With this configuration, its auto batch rule
chose 131,072 tokens/update, giving 420 updates and
55,050,240 actual training tokens in the completed depth2-vocab32768 run. Its
resolved_config.json records the actual values.

For the scaling-law question, distinguish this parameter definition from
the total trainable count used in your Chinchilla calculation. The code records
both. Learning-rate groups, warmup/warmdown settings, scaled weight decay,
gradient accumulation, and dtype are recorded as well.

## Data and measurement choices

Preparation reuses the three verified CLIMBMix shards from Task 1 and reserves
shard_06542.parquet for validation, all from the same pinned dataset revision.
Model training reads original Parquet documents. Task 1's 500 MB tokenizer
sample limit does not limit the model-training token budget.

results/preparation.json records data sources, checksums, and tokenizer identity.
Large input files, environments, caches, and checkpoints stay out of Git.

Nanochat evaluates validation bpb while training. After training, our evaluate command
reloads each saved checkpoint and evaluates **both** training and
validation on fixed samples. The samples use separate iterators and their
token-array hashes are saved. The curves start at the first saved checkpoint;
nanochat does not save a step-zero checkpoint.

- **Raw loss:** mean cross-entropy in nats over valid target tokens, including BOS.
- **Bpb:** nanochat's original byte-normalized loss; zero-byte special tokens and
  ignored targets do not contribute. Lower is better.
- **Terminal training loss:** nanochat prints a smoothed loss from the final
  microbatch in an update. It is labelled accordingly in training_progress.jsonl.
  The comparable, unsmoothed train/validation losses are in metrics.jsonl.

Evaluation samples are small estimates, not exhaustive evaluations of either split.
Keep evaluation size and packing settings fixed when comparing checkpoints.

## Commands

Run these from the GitHub repository root, which contains pyproject.toml,
task1/, and task2/. On a fresh clone, first follow task1/README.md to recreate
the 32,768-token tokenizer and training shards; those large artifacts are not
stored in Git. The scripts also support the original workspace layout with
sibling nanochat/, task1/, and task2/ folders; in that layout use --project
nanochat instead of --project . in the environment command below.

To create a CPU environment using nanochat's existing lockfile:

~~~bash
UV_PROJECT_ENVIRONMENT="$PWD/task2/.venv" uv sync --project . --frozen --extra cpu --group dev --no-install-project --cache-dir task2/cache/uv
task2/.venv/bin/python -B task2/task2.py prepare
~~~

On a CUDA machine, select **--extra gpu** instead of --extra cpu in the environment
command, and check that nvidia-smi works on that machine before starting a GPU run.

**Small setup check:** two CPU updates, to verify the workflow.

~~~bash
task2/.venv/bin/python -B task2/task2.py run --mode smoke --name smoke-02
~~~

**GPU pilot:** 20 updates to check speed, memory, and loss before the main run.

~~~bash
task2/.venv/bin/python -B task2/task2.py run --mode pilot --name pilot-d2
~~~

Inspect training_progress.jsonl, wall time in run.json, and peak GPU memory
in resolved_config.json. Use measured throughput to estimate the roughly
55-million-token run, allowing extra time for compilation, checkpoint saving,
and evaluation. Adjust the configuration after the pilot if needed.

**Main experiment:** preview first, then run after checking the pilot.

~~~bash
task2/.venv/bin/python -B task2/task2.py run --mode train --dry-run
task2/.venv/bin/python -B task2/task2.py run --mode train --name base-d2
task2/.venv/bin/python -B task2/sample_completions.py --run base-d2
~~~

Each run name must be new. The runner automatically checks checkpoint reload,
evaluates saved checkpoints, and plots results. The completion script saves five
prompts with 64 generated tokens each, temperature 0.8, top-k 50, and recorded
seeds. It uses a document BOS token, with no system prompt or chat template.

To regenerate derived evaluation files or run metric checks:

~~~bash
task2/.venv/bin/python -B task2/task2.py evaluate --run base-d2 --device cuda
task2/.venv/bin/python -B task2/plot_results.py --run base-d2
task2/.venv/bin/python -B task2/task2.py check
~~~

## Where results go

- **results/<run>/**: configuration, provenance, terminal log, structured metrics,
  checkpoint checks, evaluation CSV, and completions.
- **figures/<run>.png** and **.pdf**: training/validation bpb curves.
- **artifacts/base_checkpoints/<run>/**: model weights, optimizer state, and metadata.
- **artifacts/tokenizer/**: the selected tokenizer and byte-length tensor.
- **artifacts/base_data_climbmix/**: training links and the validation shard.

Current status is in [results/README.md](results/README.md). The main training run and five raw completions are finished. The report analysis
and later base-model benchmark comparison for Task 3 remain pending.
