Full test splits. Missing stages have not been evaluated.

| Stage | Run | ARC-Easy | ARC-Challenge | GSM-8K |
|---|---|---:|---:|---:|
| base | depth2-vocab32768 | 25.88% | 23.38% | 0.00% |
| mid | mid-d2 | 26.77% | 26.02% | 0.83% |
| mid | mid-d2-bestfit | 26.73% | 25.94% | 0.61% |
| sft | sft-d2 | 25.08% | 22.70% | 0.00% |
| sft | sft-d2-bestfit | 25.08% | 22.70% | 0.00% |

The runs named mid-d2-bestfit and sft-d2-bestfit used the historical window-based loader, as confirmed by their saved runner hashes and data counters. Their names do not identify a measured packing-method change. See [the Task 3 provenance notes](../../../README.md).
