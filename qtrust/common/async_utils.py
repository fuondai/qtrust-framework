"""
Asynchronous utilities for QTrust framework.

This module provides asynchronous processing utilities for the QTrust blockchain
sharding framework, including event handling, result processing, and caching.
"""

import asyncio
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


class AsyncEvent:
    """
    Asynchronous event for inter-component communication.

    This class provides a mechanism for components to signal events to each other
    in an asynchronous manner, with support for event data and callbacks.
    """

    def __init__(self, event_type: str, data: Any = None, source: str = None):
        """
        Initialize an asynchronous event.

        Args:
            event_type: Type identifier for the event
            data: Optional data associated with the event
            source: Optional identifier for the event source
        """
        self.event_id = str(uuid.uuid4())
        self.event_type = event_type
        self.data = data
        self.source = source
        self.timestamp = time.time()
        self.processed = False
        self.callbacks: List[Callable] = []

    def add_callback(self, callback: Callable) -> None:
        """
        Add a callback function to be executed when the event is processed.

        Args:
            callback: Function to call when event is processed
        """
        self.callbacks.append(callback)

    def mark_processed(self) -> None:
        """Mark the event as processed and execute all registered callbacks."""
        self.processed = True
        for callback in self.callbacks:
            try:
                callback(self)
            except Exception as e:
                print(f"Error in event callback: {e}")


class AsyncResult:
    """
    Container for asynchronous operation results.

    This class provides a mechanism for tracking the status and result of
    asynchronous operations, with support for callbacks and timeouts.
    """

    def __init__(self, operation_id: str = None):
        """
        Initialize an asynchronous result container.

        Args:
            operation_id: Optional identifier for the operation
        """
        self.operation_id = operation_id or str(uuid.uuid4())
        self.completed = False
        self.success = False
        self.result = None
        self.error = None
        self.timestamp = time.time()
        self.completion_time = None
        self._callbacks: List[Callable] = []
        self._event = threading.Event()

    def set_result(self, result: Any, success: bool = True) -> None:
        """
        Set the operation result and mark as completed.

        Args:
            result: Result data
            success: Whether the operation was successful
        """
        self.result = result
        self.success = success
        self.completed = True
        self.completion_time = time.time()
        self._event.set()
        self._execute_callbacks()

    def set_error(self, error: Exception) -> None:
        """
        Set an error and mark the operation as failed.

        Args:
            error: Exception or error information
        """
        self.error = error
        self.success = False
        self.completed = True
        self.completion_time = time.time()
        self._event.set()
        self._execute_callbacks()

    def add_callback(self, callback: Callable) -> None:
        """
        Add a callback function to be executed when the result is available.

        Args:
            callback: Function to call when result is available
        """
        self._callbacks.append(callback)
        if self.completed:
            try:
                callback(self)
            except Exception as e:
                print(f"Error in result callback: {e}")

    def _execute_callbacks(self) -> None:
        """Execute all registered callbacks."""
        for callback in self._callbacks:
            try:
                callback(self)
            except Exception as e:
                print(f"Error in result callback: {e}")

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the operation to complete.

        Args:
            timeout: Maximum time to wait in seconds, or None to wait indefinitely

        Returns:
            True if the operation completed, False if timed out
        """
        return self._event.wait(timeout)


class AsyncCache:
    """
    Cache for asynchronous operation results.

    This class provides a time-based caching mechanism for results of
    asynchronous operations, with support for expiration and size limits.
    """

    def __init__(self, max_size: int = 1000, default_ttl: float = 60.0):
        """
        Initialize an asynchronous cache.

        Args:
            max_size: Maximum number of items to store in the cache
            default_ttl: Default time-to-live for cache entries in seconds
        """
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache if it exists and hasn't expired.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if expiry > time.time():
                    return value
                else:
                    del self._cache[key]
        return None

    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        Store a value in the cache with the specified time-to-live.

        Args:
            key: Cache key
            value: Value to store
            ttl: Time-to-live in seconds, or None to use the default
        """
        expiry = time.time() + (ttl if ttl is not None else self._default_ttl)
        with self._lock:
            if len(self._cache) >= self._max_size and key not in self._cache:
                self._evict_oldest()
            self._cache[key] = (value, expiry)

    def _evict_oldest(self) -> None:
        """Remove the oldest entry from the cache."""
        oldest_key = None
        oldest_time = float("inf")

        for key, (_, expiry) in self._cache.items():
            if expiry < oldest_time:
                oldest_key = key
                oldest_time = expiry

        if oldest_key:
            del self._cache[oldest_key]

    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()

    def remove(self, key: str) -> bool:
        """
        Remove a specific key from the cache.

        Args:
            key: Cache key to remove

        Returns:
            True if the key was found and removed, False otherwise
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
        return False

    def size(self) -> int:
        """
        Get the current number of entries in the cache.

        Returns:
            Number of cache entries
        """
        with self._lock:
            return len(self._cache)


class AsyncProcessor:
    """
    Processor for asynchronous tasks.

    This class provides a mechanism for executing tasks asynchronously,
    with support for task prioritization, rate limiting, and result tracking.
    """

    def __init__(self, max_workers: int = 10, queue_size: int = 100):
        """
        Initialize an asynchronous processor.

        Args:
            max_workers: Maximum number of worker threads
            queue_size: Maximum size of the task queue
        """
        self._max_workers = max_workers
        self._queue_size = queue_size
        self._tasks: Dict[str, Tuple[Callable, Any, int, AsyncResult]] = {}
        self._active_tasks: Set[str] = set()
        self._lock = threading.RLock()
        self._workers: List[threading.Thread] = []
        self._running = False
        self._task_event = threading.Event()

    def start(self) -> None:
        """Start the processor and worker threads."""
        with self._lock:
            if self._running:
                return
            self._running = True

        for _ in range(self._max_workers):
            worker = threading.Thread(target=self._worker_loop)
            worker.daemon = True
            worker.start()
            self._workers.append(worker)

    def stop(self) -> None:
        """Stop the processor and worker threads."""
        with self._lock:
            self._running = False
        self._task_event.set()

        for worker in self._workers:
            worker.join(timeout=1.0)
        self._workers.clear()

    def submit(
        self, task: Callable, args: Any = None, priority: int = 0
    ) -> AsyncResult:
        """
        Submit a task for asynchronous execution.

        Args:
            task: Function to execute
            args: Arguments to pass to the function
            priority: Task priority (higher values = higher priority)

        Returns:
            AsyncResult object for tracking the task result
        """
        task_id = str(uuid.uuid4())
        result = AsyncResult(task_id)

        with self._lock:
            if len(self._tasks) >= self._queue_size:
                result.set_error(Exception("Task queue full"))
                return result

            self._tasks[task_id] = (task, args, priority, result)

        self._task_event.set()
        return result

    def _worker_loop(self) -> None:
        """Main worker thread loop for processing tasks."""
        while self._running:
            task_id = self._get_next_task()

            if not task_id or not self._running:
                self._task_event.wait(timeout=0.1)
                self._task_event.clear()
                continue

            task, args, _, result = self._tasks[task_id]

            try:
                task_result = task(args) if args is not None else task()
                result.set_result(task_result)
            except Exception as e:
                result.set_error(e)
            finally:
                with self._lock:
                    self._active_tasks.remove(task_id)
                    del self._tasks[task_id]

    def _get_next_task(self) -> Optional[str]:
        """
        Get the next task to process based on priority.

        Returns:
            Task ID of the next task to process, or None if no tasks are available
        """
        with self._lock:
            if not self._tasks:
                return None

            available_tasks = {
                tid: (task, args, priority, result)
                for tid, (task, args, priority, result) in self._tasks.items()
                if tid not in self._active_tasks
            }

            if not available_tasks:
                return None

            next_task_id = max(
                available_tasks.keys(), key=lambda tid: available_tasks[tid][2]
            )

            self._active_tasks.add(next_task_id)
            return next_task_id

    def active_count(self) -> int:
        """
        Get the number of currently active tasks.

        Returns:
            Number of active tasks
        """
        with self._lock:
            return len(self._active_tasks)

    def queued_count(self) -> int:
        """
        Get the number of queued tasks.

        Returns:
            Number of queued tasks
        """
        with self._lock:
            return len(self._tasks) - len(self._active_tasks)
