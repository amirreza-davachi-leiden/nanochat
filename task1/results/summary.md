# Task 1 experiment results

Completed: 2026-09-19T10:02:33.643685+00:00. Both runs and the evaluation completed successfully.

## Training setup

- Shared training sample: 500,000,000 UTF-8 text bytes (500 MB), 499,265,595 Unicode code points, and 189,827 documents.
- Sampling: first three training shards in sorted order, original row order, documents capped at 10,000 characters, final document cut to fit the byte budget.
- The sample is a prefix of the shuffled corpus; no new random sample was drawn.
- Dataset revision: `915333b4f8b8684f39aeaafea600fea6f43fb703`.
- Sample SHA-256: `510879be699550caa8eab8a800ab5d4616941364a5fe66ba8a65e43ebb6a47cc`.
- Nanochat revision: `92d63d4e8bb4df75c3b71618f31ddde2378b2bcd`.
- CPU training with RAYON_NUM_THREADS=8. Both vocabulary sizes include nine special tokens.
- Dependencies: pyarrow 21.0.0, rustbpe 0.1.0, tiktoken 0.11.0.

| Vocabulary size | Training time (seconds) |
| --- | ---: |
| 8,192 | 6.93 |
| 32,768 | 7.40 |

Training times cover the trainer call, excluding file saving and evaluation. They are single-run timings, not a speed benchmark.

## Measurements

Both tokenizers received the same seven starter examples. Tokens per character means token count divided by Python Unicode code-point count. Lower means fewer tokens for the same text. No beginning-of-sequence or conversation tokens were added.

| Example | Characters | Tokens: 8,192 | Tokens: 32,768 | Tokens/character: 8,192 | Tokens/character: 32,768 |
| --- | ---: | ---: | ---: | ---: | ---: |
| English: everyday | 181 | 49 | 39 | 0.2707 | 0.2155 |
| English: science | 196 | 35 | 30 | 0.1786 | 0.1531 |
| Numbers | 81 | 55 | 55 | 0.6790 | 0.6790 |
| Python code | 124 | 55 | 50 | 0.4435 | 0.4032 |
| Dutch | 161 | 63 | 55 | 0.3913 | 0.3416 |
| Chinese | 34 | 102 | 87 | 3.0000 | 2.5588 |
| Accents and emoji | 41 | 38 | 30 | 0.9268 | 0.7317 |
| English combined | 377 | 84 | 69 | 0.2228 | 0.1830 |

The larger vocabulary used 17.86% fewer tokens across the two English examples. The combined row sums the separately encoded examples and divides total tokens by total characters; it is not an unweighted average of the two ratios.

These are measurements on short, fixed starter examples, not an estimate for all English text or all languages.

## Checks and evidence for discussion

- All 14 full-text encode/decode checks passed.
- Both training records contain the same sample manifest and sample hash.
- Saved raw token bytes reconstruct every original example exactly.
- Individual token displays can show replacement characters when a token contains only part of a UTF-8 character. This does not indicate loss in full-text decoding.
- Question 2: the English rows provide token counts and tokens per character.
- Question 4: the detailed examples preserve how numbers, code, and non-English text split.
- Question 3 still needs a discussion of embedding size, softmax, and rare tokens; these tokenizer runs do not measure language-model quality.
- A separate synthetic teaching example uses zab × 4, zac × 2, and zut × 1. Its selected merges are z+a (6 occurrences), za+b (4), and za+c (2). All pair frequencies at each step are recorded separately from the CLIMBMix runs.

## Saved files

- [Training sample manifest](sample.manifest.json)
- [Full metrics](metrics.csv)
- [Exact token IDs, token pieces, and raw bytes](tokenizations.json)
- [Commands, timestamps, and exit codes](run_20260919T100207Z/run.json)
- [Training log](run_20260919T100207Z/train.log)
- [Script snapshot](run_20260919T100207Z/task1_tokenizers.py)
- [Evaluation examples snapshot](run_20260919T100207Z/eval_samples.json)
- [8,192-token training record](training_8192.json)
- [32,768-token training record](training_32768.json)

- [Decoded outputs and all token IDs/pieces](round_trip_outputs.json)
- [Separate teaching corpus, merge trees, and pair frequencies](bpe_teaching_example.json)
