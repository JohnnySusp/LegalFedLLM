from __future__ import annotations

import asyncio
import unittest

from coordinator.inference_queue import HostInferenceQueue, HostInferenceQueueFull, HostInferenceQueueTimeout


class HostInferenceQueueTests(unittest.IsolatedAsyncioTestCase):
    async def test_queue_is_fifo_and_bounds_waiting_requests(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        order: list[str] = []

        async def runner(prompt: str, max_new_tokens: int):
            order.append(prompt)
            if prompt == "first":
                started.set()
                await release.wait()
            return {"text": prompt, "max_new_tokens": max_new_tokens}

        queue = HostInferenceQueue(runner, maximum_queued=1, timeout_seconds=2)
        first = asyncio.create_task(queue.submit("first", 10))
        await started.wait()
        second = asyncio.create_task(queue.submit("second", 20))
        await asyncio.sleep(0)
        with self.assertRaises(HostInferenceQueueFull):
            await queue.submit("third", 30)
        release.set()
        self.assertEqual((await first)["text"], "first")
        self.assertEqual((await second)["text"], "second")
        self.assertEqual(order, ["first", "second"])
        await queue.close()

    async def test_timeout_cancels_request_without_returning_late_result(self) -> None:
        release = asyncio.Event()

        async def runner(prompt: str, max_new_tokens: int):
            await release.wait()
            return {"text": prompt}

        queue = HostInferenceQueue(runner, maximum_queued=1, timeout_seconds=0.02)
        with self.assertRaises(HostInferenceQueueTimeout):
            await queue.submit("slow", 10)
        release.set()
        await asyncio.sleep(0.01)
        self.assertEqual(queue.status()["queued"], 0)
        await queue.close()


if __name__ == "__main__":
    unittest.main()
