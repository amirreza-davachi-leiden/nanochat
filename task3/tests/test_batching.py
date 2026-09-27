"""Small, data-free checks for the Task 3 SFT batch construction."""

import unittest

from task3.task3 import training_batches


class FakeTokenizer:
    def get_bos_token_id(self):
        return 0

    def render_conversation(self, conversation, max_tokens=None):
        return conversation['ids'], conversation['mask']


def stats():
    return dict(conversations_opened=0, oversized_skipped=0,
                unsupervised_skipped=0, conversations_packed=0,
                rows_prepared=0, padding_positions=0)


class TrainingBatchesTest(unittest.TestCase):
    def test_short_conversations_share_a_row_and_mask_the_second_bos(self):
        data = [
            dict(ids=[0, 11, 12, 13], mask=[0, 0, 1, 1]),
            dict(ids=[0, 21, 22, 23, 24], mask=[0, 0, 1, 1, 1]),
        ]
        counters = stats()
        batches = training_batches(data, FakeTokenizer(), length=8, batch_size=1,
                                   stats=counters, buffer_size=2)
        x, y = next(batches)
        self.assertEqual(x.shape, (1, 8))
        self.assertEqual(y.shape, (1, 8))
        self.assertEqual(x[0, 0].item(), 0)
        self.assertEqual([token for token in y[0].tolist() if token >= 0],
                         [22, 23, 24, 12, 13])
        self.assertEqual(counters['conversations_packed'], 2)
        self.assertEqual(counters['padding_positions'], 0)
        batches.close()

    def test_oversized_conversation_is_counted_but_never_buffered(self):
        data = [
            dict(ids=list(range(15)), mask=[0, 0] + [1] * 13),
            dict(ids=[0, 21, 22, 23], mask=[0, 0, 1, 1]),
        ]
        counters = stats()
        batches = training_batches(data, FakeTokenizer(), length=4, batch_size=1,
                                   stats=counters, buffer_size=1)
        x, y = next(batches)
        self.assertEqual(x[0].tolist(), [0, 21, 22, 23])
        self.assertEqual([token for token in y[0].tolist() if token >= 0], [22, 23])
        # The cursor wraps during refill, so the long item is skipped again.
        self.assertEqual(counters['oversized_skipped'], 2)
        self.assertEqual(counters['conversations_packed'], 1)
        self.assertEqual(counters['padding_positions'], 1)
        batches.close()

    def test_all_oversized_conversations_fail_after_one_mixture_pass(self):
        data = [
            dict(ids=list(range(8)), mask=[0, 0] + [1] * 6),
            dict(ids=list(range(9)), mask=[0, 0] + [1] * 7),
        ]
        counters = stats()
        batches = training_batches(data, FakeTokenizer(), length=4, batch_size=1,
                                   stats=counters, buffer_size=2)
        with self.assertRaisesRegex(ValueError, 'fits in one training row'):
            next(batches)
        self.assertEqual(counters['oversized_skipped'], 2)
        batches.close()

    def test_dataset_without_assistant_targets_fails(self):
        data = [dict(ids=[0, 11, 12], mask=[0, 0, 0])]
        batches = training_batches(data, FakeTokenizer(), length=4, batch_size=1,
                                   stats=stats(), buffer_size=1)
        with self.assertRaisesRegex(ValueError, 'assistant targets'):
            next(batches)
        batches.close()


if __name__ == '__main__':
    unittest.main()
