# Task 4 - inference, deployment analysis, and reflection

This is a command-line wrapper around Nanochat, not another training stage.
It uses Nanochat's checkpoint loader, tokenizer, chat markers, generation
engine/KV cache, and benchmark evaluator. It never writes to a checkpoint.
The instructor explicitly replaced the PDF's web UI with CLI interaction.
The three production-hardening changes are proposals for the report, not
features you must implement; restoring the web UI cannot count as one.

Run commands from the repository root. The default paths in config.json point
to the Mac notebook's depth-2 SFT checkpoint at step 1000 and its 32,768-token
tokenizer. Task 4 uses exactly that model unless you supply overrides.
The Task 3 benchmark results committed in Git are from a different run;
do not present them as measurements of this Mac checkpoint.

## Files and outputs

| File | Purpose |
|---|---|
| chat.py | Check the selected model and chat interactively. |
| temperature_experiment.py | Run a controlled prompt-and-temperature sweep. |
| benchmark.py | Evaluate ARC-Easy, ARC-Challenge, and GSM-8K. |
| compare.py | Combine your full benchmark results with sourced public-model scores. |
| common.py | Shared model loading, chat formatting, context checks, and result bookkeeping; not a runnable script. |
| kv_cache_trace.ipynb | Educational, cell-by-cell view of prompt tokens, prefill, decoding, and KV-cache behavior. |
| config.json | Mac default model path and experimental settings. |
| prompts.json | Five fixed prompts, not benchmark test questions. |
| external_models.json | Published benchmark figures and URLs. Starts empty intentionally. |
| cache/ | Ignored tokenizer link and benchmark downloads. |
| results/RUN_NAME/ | Structured outputs. Each new run needs a unique name. |

Large checkpoint files, private chat transcripts, and downloaded data do not
belong in Git. The chat command never records a transcript. Review results
before committing them: prompt and response text may contain sensitive data.

## 1. Check your local model

Call the existing project's Python directly; no activation is required:

~~~bash
.venv/bin/python -B task4/chat.py --check-only
~~~

This verifies model and metadata, checks tokenizer vocabulary, loads the SFT
checkpoint, and generates one token. It reports the model hash, step, config,
and device. Auto device selects CUDA, then MPS, then CPU, depending on what
PyTorch reports available in that terminal. CPU works for this small model.

Only load checkpoint and tokenizer files you trust: PT and PKL files are
Python serialization formats, not safe untrusted downloads.

## 2. Chat through the CLI

~~~bash
.venv/bin/python -B task4/chat.py --temperature 0.7 --top-k 50 --max-new-tokens 96
~~~

The default is your Mac's SFT checkpoint. At the `You [sft]:` prompt, type
`:models` to list the three configured checkpoints, then `:model base`,
`:model mid`, or `:model sft` to switch **inside the CLI**. You can also use
`:model 1`, `:model 2`, or `:model 3`; `:current` shows the exact loaded file.
Switching loads the selected checkpoint and clears the previous model's chat
history. It does not change or overwrite checkpoint files. The paths and
steps for this menu live in `config.json` under `chat_models`. Base was not
trained as a chat assistant, so its replies may look like text continuation.

Type `:clear` for a new conversation and `:exit` to leave. Nanochat's Engine reuses
the KV cache within an answer; on the next user turn it processes the saved
conversation again. This wrapper refuses a prompt plus reserved answer budget
longer than the checkpoint's trained 512-token context. It does not silently
discard earlier turns. Shorten the prompt, lower the output budget, or clear
the conversation if it warns about length.

## 3. Temperature experiment

~~~bash
.venv/bin/python -B task4/temperature_experiment.py --name mac-sft-temperature-01
~~~

Defaults: five prompts x temperatures 0.1, 0.7, 1.5 x seeds 42, 43, 44 =
45 responses from the same checkpoint. Top-k 50, output budget, and prompt
format stay fixed. Temperature 0.1 is close to greedy, not exactly greedy.

Results include:

- run.json: checkpoint/tokenizer hashes and settings;
- generations.jsonl: every prompt, output, seed, token count, timing, stop
  reason, and sampled-versus-forced token counts;
- temperature.csv: descriptive length/variation measurements, not quality scores;
- candidate_examples.md: first-seed outputs. Review and select five examples
  yourself for the report.

A quick one-prompt test, using a new name:

~~~bash
.venv/bin/python -B task4/temperature_experiment.py --name sample-check-01 --prompt-id factual --temperatures 0.7 --seeds 42 --max-new-tokens 16
~~~

Existing run folders are never overwritten.

## 4. See prefill and decoding

Open `task4/kv_cache_trace.ipynb` in VS Code, select this repository's `.venv`
kernel, and run its cells in order. It wraps the model's forward call without
changing its calculations. The displayed calls show one prefill with the whole
prompt, followed by one-token decoding; cache-position fields show where keys
and values are reused. It also compares cached and uncached logits. This is an
explanatory exercise, not a speed benchmark or a training run, and it does not
save a trace file unless you explicitly save the notebook's outputs.

## 5. Evaluate this exact checkpoint

Preview without downloading data:

~~~bash
.venv/bin/python -B task4/benchmark.py --name mac-sft-benchmark-preview --dry-run
~~~

Small subset check, not a reportable full score:

~~~bash
.venv/bin/python -B task4/benchmark.py --name mac-sft-benchmark-check --max-problems 5
~~~

Full ARC-Easy, ARC-Challenge, and GSM-8K evaluation:

~~~bash
.venv/bin/python -B task4/benchmark.py --name mac-sft-benchmark-full
~~~

This uses Nanochat's run_chat_eval with one greedy answer, batch size 1,
and a 512-token response budget for GSM-8K. It may download test data to
task4/cache and take substantial time, especially GSM-8K on CPU/MPS.
Scores are saved as each task finishes. If interrupted, the run is marked
failed; the current runner does not automatically resume a named run.
Do not call a subset result a full-test score.

## 6. Published-model comparison

Research at least two public models: one as close as feasible to your
approximately 13-million-parameter model, one substantially larger. Finding
all three benchmark results for a truly similar-size model may be difficult;
report size and protocol differences honestly. For each published figure,
record a direct source URL and evaluation details in external_models.json.
Its shape is:

~~~json
{
  "models": [
    {
      "name": "Verified model name",
      "size_class": "comparable",
      "parameters": "Verified parameter count",
      "scores": {
        "ARC-Easy": {"percent": 0.0, "source_url": "https://source.example", "protocol_notes": "Split and scoring method"},
        "ARC-Challenge": {"percent": 0.0, "source_url": "https://source.example", "protocol_notes": "Split and scoring method"},
        "GSM8K": {"percent": 0.0, "source_url": "https://source.example", "protocol_notes": "Split and scoring method"}
      }
    }
  ]
}
~~~

The zeroes and example URLs are schema placeholders, NOT real results. Add
a second model with size_class "larger", then run:

~~~bash
.venv/bin/python -B task4/compare.py --name model-comparison-01 --benchmark-file task4/results/mac-sft-benchmark-full/benchmark.json
~~~

This produces comparison.csv, comparison.md, and sources.json. Compare carefully: Nanochat
restricts ARC answers to allowed letters; another source may use free text,
different prompting, or different few-shot examples. GSM-8K answer extraction
may also differ.

## Same code on a Linux lab computer

Copy/push Task 4 source files, not Mac checkpoints. Pass the exact lab paths:

~~~bash
LAB_PYTHON=task2/.venv/bin/python
"$LAB_PYTHON" -B task4/chat.py --check-only \
  --checkpoint-dir /absolute/lab/path/to/sft/checkpoints \
  --tokenizer-dir /absolute/lab/path/to/tokenizer \
  --step 1000 --stage sft --device cuda \
  --cache-dir /local/YOUR_USER/task4-cache
~~~

Use the same path flags with chat.py, temperature_experiment.py, and
benchmark.py. For the notebook, override the model paths in its first code
cell if you want to inspect a lab checkpoint. Add
--output-root /data/YOUR_USER/task4-results to keep results off crowded home
storage; replace placeholders with actual permitted directories. The lab
checkpoint step may differ, so inspect its metadata and select the right step.
The built-in `:models` menu currently points to Mac notebook paths; on the lab,
make a machine-specific copy of the config under ignored `task4/local/`, edit
its `model` and `chat_models` paths, and pass it with `--config` if you want
the same interactive menu there.
Never mix a lab checkpoint with Mac scores. A Mac virtual environment cannot
be copied to Linux; use the lab's own locked environment.

The report still needs your own explanation of sampling, KV cache, five
outputs, three hardening proposals, sourced benchmark comparison, one
reflection prompt, and AI-use self-disclosure. The code collects evidence;
it does not generate your written interpretation.
