# Task 3 — mid-training and supervised fine-tuning

**Ready to run? Start with [Run the experiment](#run-the-experiment).**

**Recorded run status:** Stage 1 (`mid-d2`) and Stage 2 (`sft-d2`) each completed 1,000 optimizer updates. Full ARC-Easy, ARC-Challenge, and GSM-8K evaluations are saved for the base, mid-trained, and SFT models; see the [saved benchmark table](results/benchmarks/da1c0d753dd4/scores.md). Model checkpoints remain local and are excluded from Git. The report analysis is separate from these recorded experiments.

Open [task3_experiments.ipynb](task3_experiments.ipynb). It starts with the
assignment's data sanity check in section 5.7.1: inspect MMLU and SmolTalk,
count their training examples, and understand one example. It also inspects
GSM-8K, the other Stage 1 dataset.

## Run it in VS Code

1. Open the notebook and click **Select Kernel**.
2. Choose **Select Another Kernel → Jupyter Kernels → Python (Task 2)**,
   using the kernel we registered. Its interpreter should be
   `<repository-root>/task2/.venv/bin/python`.
   Alternatively, select that interpreter under **Python Environments**.
3. Run cells from top to bottom, or choose **Run All**.

The existing Task 2 environment contains the data-loading packages and
`ipykernel`. No GPU or model checkpoint is needed for this notebook.
The first run downloads the three training splits to `task3/cache/`;
later runs reuse them. Nanochat's source files are unchanged.

## Experiments to try

- Change `EXAMPLE_INDEX` to see another raw row and its training conversation.
- Change `N_PREVIEW` to inspect more rows (some conversations are long).
- Change the repetition counts to see how the Stage 1 mixture changes.
- Compare how many assistant replies the selected examples contain.
- Add your own observations in the final Markdown cell.

Each section explains its purpose. The two small helper functions have
short docstrings: `print_rows()` displays original rows, and `file_sha256()`
identifies downloaded files for the saved record.

## Recorded results

The final code cell writes:

- [results/dataset_counts.csv](results/dataset_counts.csv): actual split sizes
  and calculated contributions for the selected repetition counts.
- [results/examples.json](results/examples.json): complete raw previews and
  selected raw/formatted examples.
- [results/inspection.json](results/inspection.json): source datasets, splits,
  shuffle seed, nanochat commit, package versions, file sizes and SHA-256 checksums.

Save the notebook to retain its displayed outputs. Rerunning the save cell
replaces these three result files with the latest inspection.

A dataset row is one example before repetition, not necessarily distinct text.
A SmolTalk row can contain several replies. Mixture entries, messages, tokens,
and optimizer steps are different quantities. Repetition settings here are
counting experiments; no training is launched.

The data inspection, both training stages, the masking example and the three-stage
benchmark comparison are recorded. The remaining report analysis should discuss
these results and the conceptual questions in the assignment.


## Training code: start here

There is one training/evaluation Python file: [task3.py](task3.py).
[config.json](config.json) holds the starting settings. This is a Task 3
adaptation of `scripts/chat_sft.py`; the original nanochat files are
unchanged. The model, tokenizer, dataset classes, optimizer, checkpoint format,
and benchmark evaluator all come from nanochat.

The main functions are:

| Function | What it does |
|---|---|
| `make_dataset()` | Select MMLU + GSM-8K for Stage 1, or SmolTalk for Stage 2. |
| `training_batches()` | Tokenize conversations, make input/target pairs, and mask targets. |
| `train()` | Load the previous model, update its weights, record losses, and save checkpoints. |
| `evaluate()` | Call `scripts.chat_eval.run_chat_eval()` for the three required benchmarks. |
| `mask_example()` | Save a concrete token-level example of assistant-only loss masking. |
| `benchmark_table()` | Combine comparable evaluation results into Markdown and CSV tables. |

Every function has a short explanation. Read these functions first; the other
functions handle settings, paths, and recording.

### How this meets the assignment

Checked against **section 5.7 of Ass1Build.pdf**:

| Requirement | Where it is handled | Current status |
|---|---|---|
| Inspect first rows and count training examples | Notebook and `results/dataset_counts.csv` | Inspection results saved |
| Stage 1: MMLU + GSM-8K, initialized from the base model | `train --stage mid` | Completed: 1,000 optimizer updates |
| Stage 2: only SmolTalk, initialized from Stage 1 | `train --stage sft --from-run mid-d2` | Completed: 1,000 optimizer updates |
| Save a separate checkpoint for each stage | `artifacts/checkpoints/<run>/` | Saved locally for both stages; excluded from Git |
| Evaluate ARC-Easy, ARC-Challenge, GSM-8K | `evaluate` for base, mid, and sft | Full test scores saved for all three model stages |
| Record implementation choices and hyperparameters | This README, config, and each run's JSON | Starting choices documented |
| Explain assistant-only loss with a token example | `mask` command and explanation below | Token example saved; report explanation remains |
| Discuss dataset purpose, size, and possible biases | Notebook observations and saved examples | Your report analysis remains |
| Compare all three model stages | Generated benchmark tables | Comparison table saved |
| Explain LoRA/QLoRA trade-offs | Conceptual report question | No PEFT implementation is required |

The assignment asks for approximately three report pages. The recorded experiments
provide evidence; the written analysis still needs to explain and interpret it.

### Our starting training choices

These are editable starting settings, not settings selected after experiments:

| Setting | Value |
|---|---|
| Starting base checkpoint | Task 2 `depth2-vocab32768`, step 420 |
| Model | Depth 2, width 128, 1 attention head, 32,768-token vocabulary |
| Context length | 512 tokens, matching Task 2 |
| Device batch size | 2 windows |
| Gradient accumulation | 4 microbatches per update |
| Batch capacity | 4,096 token positions per update, including padding |
| Horizon | 1,000 optimizer-step attempts per stage |
| MMLU / GSM-8K repetitions | 3 / 4 in the Stage 1 mixture |
| SmolTalk repetitions | 1 in the Stage 2 mixture |
| Input learning rates: embeddings / output / matrices | 0.3 / 0.008 / 0.02 |
| Initial LR fraction | 0.1 of nanochat's constructed optimizer-group rates |
| Schedule | 20 warmup steps, constant middle, linear decay over the last half |
| Final LR fraction | 0.1 of the initial rates |
| Optimizer | Nanochat Muon/AdamW; fresh optimizer moments at each stage |
| Parameters updated | All model parameters: full fine-tuning |
| Checkpoint interval | Every 250 attempts and at the end |
| Seed | 42 |

Nanochat scales some optimizer-group rates by model width. The run record saves
**actual group learning rates and weight decay**, rather than treating the input
rates as the final values. `weight_decay=0.0` is passed for matrix groups;
nanochat retains its built-in AdamW group decay settings.

A full configured run processes 4,096,000 token positions, including padding.
The number of supervised assistant targets is lower and is recorded separately.
The horizon counts optimizer-step attempts, not dataset epochs. In float16 mode,
GradScaler can skip an update; the log records whether it was applied. Short runs
reduce warmup to at most one tenth of their steps.

The inspected splits contain 99,842 MMLU rows, 7,473 GSM-8K rows, and 460,341
smol-smoltalk rows. The configured mixtures contain 329,418 Stage 1 entries and
460,341 Stage 2 entries before tokenization. A 1,000-step run does **not** promise
one complete pass over either mixture. If more training is needed, choose a new
run name and an explicit step budget after discussing the first observations.

### Differences from the original training script

The training loop follows `scripts/chat_sft.py`, with the stage selection and
checkpoint paths made explicit. To keep this one-device version understandable:

- Stage 1 loads the specified Task 2 checkpoint. Stage 2 requires a completed
  **mid** run, so it cannot silently start from the base model again.
- We use fixed update budgets and fresh optimizer moments for each stage.
  The original trainer can warm-start optimizer moments and stop by dataset pass.
- We divide long conversations into consecutive windows instead of using the
  original best-fit conversation packing. Windows overlap by one context token;
  each next-token target is included once. Short final windows are padded.
  A continuation window only sees its local preceding context, which is a
  trade-off to discuss when interpreting results.
- Windows containing no assistant targets are skipped. No answer is dropped just
  because its conversation is longer than 512 tokens. Counters record long
  conversations, skipped context-only windows, padding, and prepared windows.
- Loss is normalized by the total number of supervised targets across all
  microbatches in an update. This avoids giving a heavily padded microbatch the
  same weight as one with many assistant targets.
- Evaluation is an explicit separate command. There is no automatic benchmark
  sweep, validation-based stopping, or multi-GPU support in this small runner.

The batching counters count conversations **opened** by the iterator, which can
include a partly consumed final conversation. Windows and supervised-token
counts provide the more precise measure of actual training work.

### Loss masking: what to look for

For tokens `t0, t1, ...`, inputs are `ids[:-1]` and targets are `ids[1:]`.
We must therefore also shift the mask: `mask[1:]`. A masked target becomes `-1`,
which nanochat's cross-entropy ignores. Padding targets also become `-1`.

For each target, let `loss_i = -log P(t_i | previous tokens)`:

- Without masking: average `loss_i` over all next-token targets in the conversation.
- With masking: average only the `loss_i` whose target mask is 1.

User text supplies context but is not an answer we want the assistant to produce.
Assistant text, its end marker, and generated tool-call tokens are supervised.
Tool **output** tokens are not supervised because the tool supplies them.
Do not describe this as masking every special token: some special tokens are
part of the assistant's output and have mask 1.

The `mask` command produces actual token IDs, token pieces, masks, and resulting
training targets for a short conversation with a calculator call. It does not
load model weights or perform training. The displayed loss description is a
formula, not a measured loss value.

## Run the experiment

Run the following steps **one at a time**, waiting for each command to finish
successfully before continuing. They use the existing depth-2, 32,768-token
model. Stage 1 trains on **MMLU + GSM-8K**; Stage 2 trains on **SmolTalk**.
The main commands below request **1,000 optimizer-step attempts per stage**.

These instructions reproduce the recorded experiment. The existing `mid-d2` and
`sft-d2` runs are already complete in this workspace. For a new experiment, use
new run names consistently, as explained under **Running another experiment**.

### 1. Open a terminal and check the environment

Use VS Code's terminal on the GPU computer, then enter your cloned repository
folder (replace `/path/to/nanochat` with its actual location):

```bash
cd /path/to/nanochat
```

Use **Task 2's Python** explicitly in every command. You do not need to activate
an environment, and the notebook's selected kernel does not affect these commands.

```bash
task2/.venv/bin/python - <<'PY'
import sys
import numpy
import torch

print("Python:", sys.executable)
print("NumPy:", numpy.__version__)
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

The Python path should contain **`task2/.venv`**, and CUDA should print **`True`**
for the GPU commands below. If it prints `False`, use the GPU computer/environment
that worked for Task 2 before continuing.

On a fresh clone, recreate the Task 2 Python environment using the instructions
in [task2/README.md](../task2/README.md), then restore the tokenizer and base
checkpoint from your local training artifacts. The recorded result files do
not include model weights. The assignment PDF is supplied separately.

These inputs must also be available locally:

- `task2/artifacts/base_checkpoints/depth2-vocab32768/model_000420.pt`
- `task2/artifacts/base_checkpoints/depth2-vocab32768/meta_000420.json`
- `task2/artifacts/tokenizer/tokenizer.pkl`
- `task3/results/inspection.json` from the notebook's save cell

The notebook inspection is already recorded in this workspace. On a new machine,
make sure the local checkpoints and tokenizer are present too: they are excluded
from Git. Dataset files are reused from `task3/cache/` or downloaded when needed.

### 2. Preview the plan and save the masking example

This preview prints the input checkpoint, datasets, settings and output directory.
It does not load model weights or start training:

```bash
task2/.venv/bin/python -B task3/task3.py train --stage mid --name mid-d2 --steps 1000 --device cuda --dry-run
```

Save the concrete token-level example needed for the report's loss-masking question:

```bash
task2/.venv/bin/python -B task3/task3.py mask
```

Its output is `task3/results/loss_mask_example.csv`. This command only uses the
tokenizer; it does not train a model. An example is already saved in this workspace;
you can rerun this command to inspect it in the terminal.

### 3. Run a short GPU pilot

Try 20 steps before committing to the main experiment:

```bash
task2/.venv/bin/python -B task3/task3.py train --stage mid --name mid-pilot --steps 20 --device cuda
```

A successful run ends with `Saved checkpoint and training records`, and
`task3/results/mid-pilot/run.json` has `"status": "complete"`. The first update
may take longer because PyTorch compiles the model. If a command raises an error,
resolve it before proceeding to the next step.

The pilot has its own output folder. The main Stage 1 run below starts again
from the Task 2 base model; it does not continue from the pilot.

### 4. Evaluate the base model

Record the starting scores on **ARC-Easy, ARC-Challenge and GSM-8K**:

```bash
task2/.venv/bin/python -B task3/task3.py evaluate --stage base --device cuda
```

This evaluates all three complete test splits. GSM-8K generates answers, so it
can take longer than the multiple-choice tasks. Let the command finish all three.
It prints the folder where it saved the benchmark evidence.

### 5. Train and evaluate Stage 1: mid-training

Train on MMLU + GSM-8K:

```bash
task2/.venv/bin/python -B task3/task3.py train --stage mid --name mid-d2 --steps 1000 --device cuda
```

After it finishes successfully, evaluate its checkpoint:

```bash
task2/.venv/bin/python -B task3/task3.py evaluate --stage mid --run mid-d2 --device cuda
```

The final model is saved as
`task3/artifacts/checkpoints/mid-d2/model_001000.pt`.
Training settings and losses are in `task3/results/mid-d2/`.

### 6. Train and evaluate Stage 2: SFT

Load the completed **mid-d2** checkpoint and train only on SmolTalk:

```bash
task2/.venv/bin/python -B task3/task3.py train --stage sft --name sft-d2 --from-run mid-d2 --steps 1000 --device cuda
```

Here, **`--from-run mid-d2`** selects the Stage 1 checkpoint. After training
finishes successfully, evaluate the SFT model:

```bash
task2/.venv/bin/python -B task3/task3.py evaluate --stage sft --run sft-d2 --device cuda
```

The final model is saved as
`task3/artifacts/checkpoints/sft-d2/model_001000.pt`.
Training settings and losses are in `task3/results/sft-d2/`.

### 7. Open the recorded results

| What to inspect | File or folder |
|---|---|
| Stage 1 training loss | `task3/results/mid-d2/metrics.csv` |
| Stage 2 training loss | `task3/results/sft-d2/metrics.csv` |
| Settings, actual token counts, timing and completion status | Each run's `run.json` |
| Base / mid / SFT benchmark comparison | `task3/results/benchmarks/<protocol>/scores.md` |
| The same scores for analysis | `task3/results/benchmarks/<protocol>/scores.csv` |
| Token-level masking example | `task3/results/loss_mask_example.csv` |

`<protocol>` is a generated folder name, not text you need to type. Each
evaluation prints its exact path. With the same evaluation settings, dtype,
tokenizer and test files, the three models appear together in one comparison
table. Keep `config.json` unchanged between these benchmark runs for a consistent
comparison. Use the same GPU environment for all three evaluations.

### Optional: a quick evaluation check

To check evaluation with at most 10 problems per task before the full runs:

```bash
task2/.venv/bin/python -B task3/task3.py evaluate --stage base --max-problems 10 --device cuda
```

This is a **subset check** and is saved in a separate table. It does not replace
the full benchmark commands above.

### Running another experiment

Training run names cannot be reused. For another experiment, choose a new name
such as `mid-d2-v2`, use that same name with `evaluate --run`, and pass it as
`--from-run mid-d2-v2` when starting the matching SFT run. Give that SFT run a new
name too. A failed or interrupted run is not accepted as a completed Stage 1
input; the runner does not resume interrupted optimizer state.

The current defaults live in [config.json](config.json). `--steps` overrides the
training budget for a particular command. If you change the step count, the
number in the final checkpoint filename changes accordingly. You may explicitly
replace `--device cuda` with `--device cpu`, but full training and generated-answer
evaluation can be substantially slower.

## Benchmark protocol and saved evidence

Evaluation reuses the functions in **`scripts/chat_eval.py`**:

- **ARC-Easy / ARC-Challenge:** test splits, constrained choice-letter prediction.
- **GSM-8K:** `main/test`, one generated answer, greedy decoding (`temperature=0`),
  at most 512 new tokens; native tool handling and numeric-answer scoring.
- Default: **complete test splits**, with batch size 1 for categorical tasks.
- For a quick check, add `--max-problems 10`. Those scores are explicitly labelled
  as subset checks and go into a separate table from full evaluation results.

All stages use the same tokenizer and native chat prompt formatting, including
the base model. That model has not yet been trained to follow the chat format;
this is part of what the comparison measures. The native evaluator's handling
of prompt lengths is retained. It may evaluate contexts longer than the
512-token training windows, within the model's rotary cache limit.

Each evaluation records its checkpoint checksum, settings, exact test-file
checksums, number of evaluated examples, and scores. Protocol-specific folders
prevent mixing different generation settings, subset sizes, tokenizer files,
dtypes, or test snapshots into one table. ARC accuracy and GSM-8K success rates
are stored as fractions in CSV and shown as percentages in Markdown.

Files are created only by their corresponding command:

| Output | Contents |
|---|---|
| `results/<run>/run.json` | Settings, data counts/checksums, model lineage, actual token counters, timing and status |
| `results/<run>/metrics.csv` | Assistant-target cross-entropy per update, target count and LR multiplier |
| `artifacts/checkpoints/<run>/` | Model, optimizer and metadata checkpoints; excluded from Git |
| `results/benchmarks/<protocol>/` | Individual evaluation JSON files and combined `scores.csv` / `scores.md` |
| `results/loss_mask_example.csv` | The concrete token-level masking example |

The original dataset-inspection results and executed notebook are preserved.
Completed training and benchmark records are included in this folder.

## Code verification

The initial implementation passed ten non-training checks; the historical record
and checked source hash are in [results/code_checks.json](results/code_checks.json).
They covered stage selection, checkpoint routing, long-conversation windows,
loss masks, schedule boundaries, settings and dry-run behavior. That record
applies to its recorded source version. Subsequent completed training and
evaluation records are in `results/mid-d2/`, `results/sft-d2/`, and
`results/benchmarks/`; each records the relevant run or checkpoint provenance.
