# Task 2 results

The GPU pilot and full training run both completed successfully on 2026-09-20.
The full run is **depth2-vocab32768**, trained on an NVIDIA GeForce RTX 4060.

| Measurement | Full run |
|---|---:|
| Vocabulary size | 32,768 |
| Transformer layers | 2 |
| Embedding width | 128 |
| Attention heads per layer | 1 |
| Total trainable parameters | 12,976,170 |
| Parameters used by nanochat's horizon rule | 4,587,532 |
| Context length | 512 tokens |
| Optimizer updates | 420 |
| Tokens per optimizer update | 131,072 |
| Total training tokens processed | 55,050,240 |
| Training command wall time, including setup and saving | 315.81 seconds |
| Full workflow wall time, including reload/evaluation/plotting | 328.17 seconds |
| Final training-sample cross-entropy | 4.609796 nats/token |
| Final validation-sample cross-entropy | 4.487810 nats/token |
| Final training-sample bpb | 1.412736 |
| Final validation-sample bpb | 1.333264 |

## Checks performed

- The pilot completed 20 updates; the full run completed all 420 updates.
- All 17 main-run model checksums and both pilot model checksums match the saved evaluation records.
- Checkpoint step counts and token counts agree with the resolved configurations.
- All recorded training/validation losses and bpb values are finite.
- The selected tokenizer and nanochat source checksums match the run records.
- The final checkpoint reloaded successfully and produced finite scores.
- Five raw completions have now been generated and saved, with their seeds,
  token IDs, settings, and the final checkpoint checksum.

## Explanations for the report

These notes cover the four questions in Task 2. Use them to understand the
results and write the approximately two-page report in your own words, including
the bpb figure, source citations, and the assignment's AI-assistance disclosure.

### 1. What does depth control, and how are the other settings chosen?

**Depth is the number of transformer blocks stacked in the model.** Each block
contains attention and a feed-forward network. Our depth of 2 therefore means
two blocks; it does not mean two attention heads.

In this nanochat revision, depth also influences the model's width and head
count. With the default aspect ratio of 64 and head dimension of 128:

~~~text
width = ceil(depth × 64 / 128) × 128 = 128
heads per layer = width / 128 = 1
~~~

The rounding makes the width divisible by the head dimension. Our model thus
has **2 layers, width 128, and 1 attention head per layer**. The actual total
trainable parameter count is **12,976,170**. This includes the input embeddings,
value embeddings, output matrix, transformer weights, and small scalar/gating
parameters; it should be counted from the implementation rather than guessed
from depth alone.

The remaining configuration follows a chain of rules:

- **Training horizon:** nanochat multiplies its selected scaling-parameter count
  by the default data-to-parameter ratio of 12. The calculation is shown below.
- **Global batch size:** it compares that target token budget with a depth-12
  reference, scales the reference batch of 524,288 tokens by
  `(target_tokens / reference_tokens)^0.383`, and rounds to a power of two.
  Our result is **131,072 tokens per optimizer update**.
- **Gradient accumulation:** each device batch contains 2 sequences × 512 tokens
  = 1,024 tokens. On one GPU, accumulating **128 batches** gives the global batch.
- **Learning rates:** the batch-size multiplier is
  `sqrt(131072 / 524288) = 0.5`. Embedding/output groups also receive a width
  multiplier of `sqrt(768 / 128)`, approximately 2.449. Different parameter
  groups have different learning rates; there is not one learning rate for
  the entire model.

| Parameter group | Initial learning rate before the schedule multiplier |
|---|---:|
| Input embeddings | 0.367423 |
| Value embeddings | 0.183712 |
| Output matrix | 0.009798 |
| Transformer matrix groups (Muon) | 0.010000 |

The learning-rate schedule uses **40 warmup updates**, then a constant portion,
then a linear reduction over approximately the last **65% of the run**.
The configured endpoint is 5% of the initial rates. Because the training loop
uses indices 0–419, the last executed update uses approximately **5.35%**.
The initial Muon weight-decay coefficient is automatically scaled to **3.360004**
and then follows a cosine decay toward zero. Scalar groups and their individual
rates are recorded in the resolved configuration.

**Not every setting is calculated from depth.** Our 32,768-token tokenizer,
512-token context, 2 sequences per device batch, full-attention pattern, and
evaluation/checkpoint intervals are explicit choices. Warmup length and the
warmdown fraction are configurable defaults. Depth changes the architecture
and influences derived settings, while these other inputs still matter.

The single-dial design gives a consistent starting point and reduces the number
of settings a beginner must tune. Its limitation is that rules calibrated on
reference models need not be optimal for every model size, dataset, or GPU.
Independent tuning offers more flexibility, at the cost of more experiments.

Code evidence: [architecture and training rules](../../scripts/base_train.py)
(`build_model_meta`, `get_scaling_params`, and `get_lr_multiplier`) and
[parameter counting and optimizer groups](../../nanochat/gpt.py)
(`num_scaling_params` and `setup_optimizer`). Exact run values are in
[resolved_config.json](depth2-vocab32768/resolved_config.json).

### 2. How does our training budget compare with Chinchilla?

The Chinchilla framework balances model size and training data for a fixed
training-compute budget. A common approximate rule is **20 training tokens per
parameter**, consistent with the paper's 70-billion-parameter model trained on
1.4 trillion tokens. This is a heuristic, not a universal exact optimum.
See [Hoffmann et al. (2022), Table 1 and Section 3](https://arxiv.org/html/2203.15556v1).

For the assignment's calculation using our **total trainable parameter count**:

~~~text
N_total = 12,976,170
D_Chinchilla ≈ 20 × N_total = 259,523,400 tokens
~~~

Nanochat uses a different parameter definition for its horizon:

~~~text
N_scaling = transformer parameters + output matrix
          = 393,228 + 4,194,304
          = 4,587,532
D_nanochat_target = 12 × N_scaling = 55,050,384 tokens
updates = floor(55,050,384 / 131,072) = 420
D_actual = 420 × 131,072 = 55,050,240 tokens
~~~

Our actual run processes about **4.24 tokens per total parameter**, or **21.2%**
of the 259.52-million-token estimate. The difference comes from **both** the
12-versus-20 multiplier and the smaller parameter count used by nanochat.

To isolate the multiplier difference, apply both rules to nanochat's selected
parameter count: `20 × 4,587,532 = 91,750,640` tokens. Our actual budget is
approximately **60%** of that value. This second calculation keeps the parameter
definition constant; it is not the total-parameter calculation above.

**Under-trained**, in this comparison, means using fewer training tokens than
the compute-optimal allocation would suggest for a model of that size.
**Over-trained**, relative to that allocation, means using more tokens: some
training compute could instead have gone into a larger model. Over-training in
this sense is different from overfitting; additional data can still improve
the model and may be useful when a small model is cheaper to serve.

Our run is below the simple 20-tokens-per-parameter estimate, but this does
not demonstrate the true optimum for our small, embedding-heavy architecture.
Data, tokenizer, optimizer, and architecture differ from the original study.
We followed nanochat's default horizon and measured one full run, rather than
conducting a sweep to establish compute optimality.

### 3. What do the loss curves show?

**Bits per byte measures how well the model predicts text, normalized by the
number of bytes represented by the target tokens. Lower is better.** Nanochat
sums the negative log probabilities of ordinary text targets, converts nats
to bits, and divides by their total byte count. Zero-byte special tokens are
excluded. Dividing ordinary cross-entropy by `ln(2)` alone would instead give
bits per token.

Our graph uses the same fixed **16,384-token training sample** and the same
fixed **16,384-token held-out validation sample** at every saved checkpoint.
These evaluation iterators are separate from the iterator used for training.
The first plotted checkpoint is step 25; it is not the untrained model.

| Checkpoint | Tokens processed | Training bpb | Validation bpb |
|---|---:|---:|---:|
| Step 25 | 3,276,800 | 2.182938 | 2.034531 |
| Step 200 | 26,214,400 | 1.545830 | 1.448652 |
| Step 300 | 39,321,600 | 1.456028 | 1.372635 |
| Step 420 | 55,050,240 | 1.412736 | 1.333264 |

Both curves decrease at every recorded checkpoint. A reasonable interpretation
is that the model first learns common token patterns quickly, then makes
smaller refinements. The learning-rate reduction later in training can also
contribute to slower changes; this run does not separate these effects.

The curve becomes noticeably flatter around **steps 200–300**, or roughly
26–39 million tokens, but there is no exact stopping point where learning ends.
For equal 100-update intervals, validation bpb falls by about **0.495** from
steps 25–125, **0.076** from steps 200–300, and **0.036** from steps 300–400.
These measurements support a gradual slowdown rather than a complete plateau.

Validation bpb is lower than training bpb here. The samples contain different
text, and the validation sample may be easier. It also represents more text
bytes for the same number of target tokens: **79,476 versus 76,993**. Both
content and token-to-byte distributions can affect the comparison. The
validation cross-entropy is also lower, so the byte denominator alone does
not explain the gap.

A rising validation curve while training loss continues falling can indicate
overfitting. We do not observe that pattern in these samples. However, these
small fixed samples and a single run cannot establish generalization over the
whole dataset. The graph measures text prediction, not factual accuracy or
reasoning ability.

Use the [bpb plot](../figures/depth2-vocab32768.pdf) in the report and the
[evaluation CSV](depth2-vocab32768/evaluation.csv) for the numbers. The metric
definition is implemented in [nanochat's evaluator](../../nanochat/loss_eval.py).

### 4. What do the five raw completions reveal?

We generated one continuation for each of five fixed prompts, using temperature
0.8, top-k 50, 64 new tokens per prompt, and seeds 42–46. A document-start token
was included, with **no system prompt or chat template**. Sampling used the
final checkpoint on CPU in float32. Full outputs and token IDs are in
[completions.json](depth2-vocab32768/completions.json).

| Prompt/topic | Actual output excerpt | Observation |
|---|---|---|
| Garden | “is the largest garden with the largest garden space.” | Uses relevant vocabulary but immediately repeats itself. |
| Freezing water | “The water is placed in a water of the water” | Associates water with the prompt, but produces incoherent statements. |
| Solving a difficult problem | “to solve a problem.” | Gives a circular continuation rather than a useful explanation. |
| Student and notebook | “What is the teacher? What is the teacher?” | Moves toward classroom language, then repeats a question. |
| City and bicycles | “the city's population is always at home” | Produces a sentence-like claim while drifting from the bicycle premise. |

The examples show that the model has learned some word associations, sentence
patterns, and document formatting. They also show grammar errors, repetition,
weak continuity, and unsupported statements. The outputs stop at the fixed
64-token limit, so an unfinished final sentence alone is not evidence of a
failure to end a sentence.

A plausible explanation is the combination of limited model capacity, a modest
training-token budget, and next-token pretraining without instruction tuning.
Pretraining rewards predicting text; it does not directly teach the model to
answer a user's question or check a claim. These five samples illustrate
limitations, but they cannot isolate which factor caused each failure.

Overall, lower bpb is evidence of improved text prediction, while the generated
examples show that coherent, useful responses remain limited.

## Files to inspect

- [Evaluation table](depth2-vocab32768/evaluation.csv)
- [Learning curves](../figures/depth2-vocab32768.png)
- [Five completions](depth2-vocab32768/completions.json)
- [Actual model and training configuration](depth2-vocab32768/resolved_config.json)
- [Run provenance and timing](depth2-vocab32768/run.json)
- Final checkpoint (local, excluded from Git):
  task2/artifacts/base_checkpoints/depth2-vocab32768/model_000420.pt

The table and plot use evaluation of reloaded checkpoints. The in-training log
reports final validation bpb 1.332760, whereas checkpoint evaluation reports
1.333264. Training uses a compiled model and checkpoint evaluation an uncompiled
one; the small difference is consistent with numerical variation between those
paths. Keep the checkpoint-evaluation values together when discussing the curves.
The training log's MFU warning concerns an unavailable RTX 4060 peak-FLOPS
estimate; it is not a non-finite model loss.

The completions were generated on CPU in float32, with temperature 0.8, top-k 50,
64 generated tokens per prompt, seeds 42 through 46, and no system prompt or
chat formatting. They use the same final weights, loaded through nanochat.

The earlier setup-smoke and layout-smoke folders are pipeline checks only.
Explanatory notes for all four Task 2 report questions are now included above.
The final report still needs to be written and condensed, with the plot included
and an accessible checkpoint link. Task 3 comparison benchmarks remain pending.
