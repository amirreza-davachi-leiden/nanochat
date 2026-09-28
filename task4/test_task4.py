"""Small CPU-free checks for the Task 4 runner's bookkeeping."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import common
import compare
import chat


class FakeTokenizer:
    def get_bos_token_id(self):
        return 100

    def encode_special(self, name):
        return {
            "<|user_start|>": 101,
            "<|user_end|>": 102,
            "<|assistant_start|>": 103,
        }[name]

    def encode(self, text):
        return [104, 105]


class Task4Tests(unittest.TestCase):
    def test_chat_model_profile_changes_only_checkpoint_selection(self):
        current = {
            "checkpoint_dir": Path("/tmp/old-checkpoint"),
            "tokenizer_dir": Path("/tmp/old-tokenizer"),
            "step": 1000, "stage": "sft", "device": "cpu",
            "cache_dir": Path("/tmp/cache"),
        }
        selected = chat.profile_spec(current, {
            "checkpoint_dir": "task4",
            "tokenizer_dir": "task4",
            "step": 420, "stage": "base",
        })
        self.assertEqual(selected["stage"], "base")
        self.assertEqual(selected["step"], 420)
        self.assertEqual(selected["device"], "cpu")
        self.assertEqual(selected["cache_dir"], current["cache_dir"])
        self.assertFalse(chat.same_checkpoint(current, selected))
        with self.assertRaises(ValueError):
            chat.profile_spec(current, {"stage": "mid"})

    def test_chat_prefix_keeps_history_and_adds_markers(self):
        ids = common.chat_prefix(FakeTokenizer(), "Hi", [100, 200])
        self.assertEqual(ids, [100, 200, 101, 104, 105, 102, 103])

    def test_context_budget_counts_prompt_and_reply(self):
        model = SimpleNamespace(config=SimpleNamespace(sequence_len=8))
        common.context_check(model, [1, 2, 3], 5)
        with self.assertRaises(ValueError):
            common.context_check(model, [1, 2, 3, 4], 5)

    def test_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            spec = {"output_root": Path(temporary)}
            common.new_run(spec, "run-1", "test", {}, {})
            with self.assertRaises(FileExistsError):
                common.new_run(spec, "run-1", "test", {}, {})

    def test_comparison_requires_full_benchmark_and_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            benchmark = folder / "benchmark.json"
            external = folder / "external.json"
            own = {
                "complete": True,
                "protocol": {"max_problems": None},
                "model": {
                    "stage": "sft", "step": 1000,
                    "model_sha256": "example", "trainable_parameters": 12976170,
                },
                "scores": {
                    task: {"percent": 20.0, "evaluated_examples": 10}
                    for task in common.TASKS
                },
            }
            benchmark.write_text(json.dumps(own), encoding="utf-8")
            entry = lambda size: {
                "name": f"Model {size}",
                "size_class": size,
                "parameters": "example",
                "scores": {
                    task: {
                        "percent": 30.0,
                        "source_url": "https://example.org/source",
                        "protocol_notes": "test fixture",
                    }
                    for task in common.TASKS
                },
            }
            external.write_text(json.dumps({
                "models": [entry("comparable"), entry("larger")]
            }), encoding="utf-8")
            with patch("sys.argv", [
                "compare.py", "--benchmark-file", str(benchmark),
                "--external-file", str(external), "--output-root", str(folder),
                "--name", "comparison",
            ]):
                compare.main()
            self.assertTrue((folder / "comparison" / "comparison.csv").is_file())
            self.assertTrue((folder / "comparison" / "comparison.md").is_file())
            self.assertEqual(
                len((folder / "comparison" / "comparison.csv").read_text().splitlines()),
                10,
            )


if __name__ == "__main__":
    unittest.main()
