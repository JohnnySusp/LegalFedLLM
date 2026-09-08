from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Awaitable, Callable


class HostInferenceQueueError(RuntimeError):
    pass


class HostInferenceQueueFull(HostInferenceQueueError):
    pass


class HostInferenceQueueTimeout(HostInferenceQueueError):
    pass


@dataclass(slots=True)
class _QueuedInference:
    prompt: str
    max_new_tokens: int
    future: asyncio.Future[dict[str, Any]]


class HostInferenceQueue:
    def __init__(
        self,
        runner: Callable[[str, int], Awaitable[dict[str, Any]]],
        *,
        maximum_queued: int = 4,
        timeout_seconds: float = 60.0,
    ):
        if maximum_queued < 1:
            raise ValueError("Host inference queue size must be positive")
        if timeout_seconds <= 0:
            raise ValueError("Host inference timeout must be positive")
        self.runner = runner
        self.maximum_queued = maximum_queued
        self.timeout_seconds = timeout_seconds
        self._queue: asyncio.Queue[_QueuedInference] = asyncio.Queue(
            maxsize=maximum_queued
        )
        self._worker: asyncio.Task[None] | None = None
        self._closed = False
        self._running = False

    def status(self) -> dict[str, Any]:
        return {
            "maximum_queued": self.maximum_queued,
            "queued": self._queue.qsize(),
            "running": self._running,
            "closed": self._closed,
        }

    def _ensure_worker(self) -> None:
        if self._closed:
            raise HostInferenceQueueError("Host inference queue is closed")
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run())

    async def submit(
        self,
        prompt: str,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        if self._closed:
            raise HostInferenceQueueError("Host inference queue is closed")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        item = _QueuedInference(prompt, max_new_tokens, future)
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull as exc:
            raise HostInferenceQueueFull("Host inference queue is full") from exc
        self._ensure_worker()
        try:
            return await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            future.cancel()
            raise HostInferenceQueueTimeout(
                "Host inference request expired while queued or running"
            ) from exc

    async def _run(self) -> None:
        while not self._closed:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                if item.future.cancelled():
                    continue
                self._running = True
                result = await self.runner(item.prompt, item.max_new_tokens)
                if not item.future.done():
                    item.future.set_result(result)
            except Exception as exc:
                if not item.future.done():
                    item.future.set_exception(exc)
            finally:
                self._running = False
                self._queue.task_done()

    async def close(self) -> None:
        self._closed = True
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not item.future.done():
                item.future.cancel()
            self._queue.task_done()
        worker = self._worker
        if worker is not None and not worker.done():
            worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                pass
