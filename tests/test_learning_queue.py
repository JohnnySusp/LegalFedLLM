from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from client.learning_queue import LearningQueue
from client.training import load_private_examples


class LearningQueueTests(unittest.TestCase):
    def test_consumed_batch_is_deleted_without_deleting_new_examples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private" / "train.jsonl"
            queue = LearningQueue(path)
            first = queue.append("first prompt", "first answer")
            second = queue.append("second prompt", "second answer")
            batch = queue.begin()
            self.assertIsNotNone(batch)
            assert batch is not None
            self.assertFalse(path.exists())
            self.assertEqual([item.example_id for item in batch.examples], [first.example_id, second.example_id])

            third = queue.append("third prompt", "third answer")
            receipt = queue.complete(
                batch,
                adapter_version=2,
                checkpoint_hash="a" * 64,
                round_id="round-2",
                training_record_hash="b" * 64,
            )

            remaining = load_private_examples(path)
            self.assertEqual([item.example_id for item in remaining], [third.example_id])
            self.assertEqual(receipt["consumed_example_count"], 2)
            receipt_text = json.dumps(receipt)
            self.assertNotIn("first prompt", receipt_text)
            self.assertNotIn("first answer", receipt_text)
            self.assertFalse(batch.path.exists())

    def test_failed_batch_restores_old_examples_before_new_examples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            queue = LearningQueue(path)
            first = queue.append("first", "answer one")
            batch = queue.begin()
            assert batch is not None
            second = queue.append("second", "answer two")

            queue.restore(batch)

            restored = load_private_examples(path)
            self.assertEqual(
                [item.example_id for item in restored],
                [first.example_id, second.example_id],
            )
            self.assertFalse(batch.path.exists())

    def test_startup_recovers_unfinished_inflight_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            queue = LearningQueue(path)
            first = queue.append("recover me", "answer")
            batch = queue.begin()
            assert batch is not None
            self.assertFalse(path.exists())

            restarted = LearningQueue(path)

            examples = load_private_examples(path)
            self.assertEqual([item.example_id for item in examples], [first.example_id])
            self.assertEqual(restarted.status()["queued_example_count"], 1)
            self.assertFalse(batch.path.exists())

    def test_missing_queue_is_a_valid_empty_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private" / "train.jsonl"
            queue = LearningQueue(path)
            self.assertEqual(
                queue.status(),
                {"queued_example_count": 0, "queued_dataset_hash": None},
            )
            self.assertIsNone(queue.begin())
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
