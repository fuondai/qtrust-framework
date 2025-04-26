#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QTrust Blockchain Sharding Framework - Async Utilities
This module provides asynchronous processing utilities to reduce synchronization barriers.
"""

import asyncio
import threading
import queue
import time
import functools
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, Coroutine


class AsyncProcessor:
    """
    Provides asynchronous processing capabilities to reduce synchronization barriers.
    Implements a thread pool and task queue for non-blocking operations.
    """

    def __init__(self, num_workers: int = 4, queue_size: int = 1000):
        """
        Initialize the async processor.

        Args:
            num_workers: Number of worker threads
            queue_size: Maximum size of the task queue
        """
        self.num_workers = num_workers
        self.task_queue = queue.Queue(maxsize=queue_size)
        self.workers = []
        self.running = False
        self.results = {}
        self.result_lock = threading.Lock()
        self.result_event = threading.Event()
        self.task_counter = 0
        self.task_counter_lock = threading.Lock()

    def start(self):
        """
        Start the async processor.
        """
        if self.running:
            return

        self.running = True
        self.workers = []

        for i in range(self.num_workers):
            worker = threading.Thread(target=self._worker_loop, args=(i,))
            worker.daemon = True
            worker.start()
            self.workers.append(worker)

    def stop(self):
        """
        Stop the async processor.
        """
        self.running = False

        # Add termination tasks to ensure workers exit
        for _ in range(self.num_workers):
            try:
                self.task_queue.put(None, block=False)
            except queue.Full:
                pass

        # Wait for workers to terminate
        for worker in self.workers:
            worker.join(timeout=2.0)

        self.workers = []

    def _worker_loop(self, worker_id: int):
        """
        Main loop for worker threads.

        Args:
            worker_id: Worker thread ID
        """
        while self.running:
            try:
                task = self.task_queue.get(timeout=1.0)
                if task is None:
                    # Termination signal
                    self.task_queue.task_done()
                    break

                task_id, func, args, kwargs, callback = task

                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    result = e
                    success = False

                # Store the result
                with self.result_lock:
                    self.results[task_id] = (success, result)

                # Notify waiting threads
                self.result_event.set()

                # Call the callback if provided
                if callback:
                    try:
                        callback(success, result)
                    except Exception as e:
                        print(f"Error in callback: {e}")

                self.task_queue.task_done()

            except queue.Empty:
                # No tasks available, continue
                continue
            except Exception as e:
                print(f"Error in worker {worker_id}: {e}")
                time.sleep(1.0)

    def submit_task(
        self, func: Callable, *args, callback: Optional[Callable] = None, **kwargs
    ) -> int:
        """
        Submit a task for asynchronous execution.

        Args:
            func: Function to execute
            *args: Arguments for the function
            callback: Optional callback function to call with the result
            **kwargs: Keyword arguments for the function

        Returns:
            Task ID
        """
        if not self.running:
            self.start()

        with self.task_counter_lock:
            task_id = self.task_counter
            self.task_counter += 1

        task = (task_id, func, args, kwargs, callback)
        self.task_queue.put(task)

        return task_id

    def get_result(
        self, task_id: int, timeout: Optional[float] = None
    ) -> Tuple[bool, Any]:
        """
        Get the result of a task.

        Args:
            task_id: Task ID
            timeout: Timeout in seconds

        Returns:
            Tuple of (success, result)
        """
        start_time = time.time()

        while timeout is None or time.time() - start_time < timeout:
            with self.result_lock:
                if task_id in self.results:
                    result = self.results.pop(task_id)
                    return result

            # Wait for new results
            self.result_event.wait(timeout=0.1)
            self.result_event.clear()

        # Timeout
        return (False, TimeoutError("Task result not available within timeout"))

    def wait_for_tasks(
        self, task_ids: List[int], timeout: Optional[float] = None
    ) -> Dict[int, Tuple[bool, Any]]:
        """
        Wait for multiple tasks to complete.

        Args:
            task_ids: List of task IDs
            timeout: Timeout in seconds

        Returns:
            Dictionary mapping task IDs to (success, result) tuples
        """
        results = {}
        remaining_ids = set(task_ids)
        start_time = time.time()

        while remaining_ids and (timeout is None or time.time() - start_time < timeout):
            with self.result_lock:
                for task_id in list(remaining_ids):
                    if task_id in self.results:
                        results[task_id] = self.results.pop(task_id)
                        remaining_ids.remove(task_id)

            if not remaining_ids:
                break

            # Wait for new results
            self.result_event.wait(timeout=0.1)
            self.result_event.clear()

        # Add timeouts for any remaining tasks
        for task_id in remaining_ids:
            results[task_id] = (
                False,
                TimeoutError("Task result not available within timeout"),
            )

        return results

    def map_async(
        self, func: Callable, items: List[Any], callback: Optional[Callable] = None
    ) -> List[int]:
        """
        Apply a function to each item in a list asynchronously.

        Args:
            func: Function to apply
            items: List of items
            callback: Optional callback function to call with each result

        Returns:
            List of task IDs
        """
        task_ids = []

        for item in items:
            task_id = self.submit_task(func, item, callback=callback)
            task_ids.append(task_id)

        return task_ids

    def map(
        self, func: Callable, items: List[Any], timeout: Optional[float] = None
    ) -> List[Any]:
        """
        Apply a function to each item in a list and wait for all results.

        Args:
            func: Function to apply
            items: List of items
            timeout: Timeout in seconds

        Returns:
            List of results
        """
        task_ids = self.map_async(func, items)
        results = self.wait_for_tasks(task_ids, timeout=timeout)

        # Sort results by task ID to maintain order
        sorted_results = [results[task_id][1] for task_id in task_ids]

        return sorted_results


class AsyncEvent:
    """
    Event for asynchronous coordination between components.
    Similar to threading.Event but with callback support.
    """

    def __init__(self):
        """
        Initialize the async event.
        """
        self.event = threading.Event()
        self.callbacks = []
        self.lock = threading.Lock()

    def set(self):
        """
        Set the event and trigger callbacks.
        """
        with self.lock:
            callbacks = self.callbacks.copy()

        self.event.set()

        for callback in callbacks:
            try:
                callback()
            except Exception as e:
                print(f"Error in event callback: {e}")

    def clear(self):
        """
        Clear the event.
        """
        self.event.clear()

    def is_set(self) -> bool:
        """
        Check if the event is set.

        Returns:
            True if the event is set, False otherwise
        """
        return self.event.is_set()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for the event to be set.

        Args:
            timeout: Timeout in seconds

        Returns:
            True if the event was set, False if timeout occurred
        """
        return self.event.wait(timeout=timeout)

    def add_callback(self, callback: Callable):
        """
        Add a callback to be called when the event is set.

        Args:
            callback: Callback function
        """
        with self.lock:
            self.callbacks.append(callback)

    def remove_callback(self, callback: Callable) -> bool:
        """
        Remove a callback.

        Args:
            callback: Callback function

        Returns:
            True if the callback was removed, False if not found
        """
        with self.lock:
            try:
                self.callbacks.remove(callback)
                return True
            except ValueError:
                return False


class AsyncResult:
    """
    Container for an asynchronous result.
    Similar to concurrent.futures.Future but with simpler interface.
    """

    def __init__(self):
        """
        Initialize the async result.
        """
        self.result = None
        self.exception = None
        self.done = AsyncEvent()
        self.callbacks = []

    def set_result(self, result: Any):
        """
        Set the result value.

        Args:
            result: Result value
        """
        self.result = result
        self.done.set()

        for callback in self.callbacks:
            try:
                callback(self)
            except Exception as e:
                print(f"Error in result callback: {e}")

    def set_exception(self, exception: Exception):
        """
        Set an exception.

        Args:
            exception: Exception object
        """
        self.exception = exception
        self.done.set()

        for callback in self.callbacks:
            try:
                callback(self)
            except Exception as e:
                print(f"Error in result callback: {e}")

    def get(self, timeout: Optional[float] = None):
        """
        Get the result, waiting if necessary.

        Args:
            timeout: Timeout in seconds

        Returns:
            Result value

        Raises:
            Exception: If an exception was set
            TimeoutError: If timeout occurs
        """
        if not self.done.wait(timeout=timeout):
            raise TimeoutError("Result not available within timeout")

        if self.exception:
            raise self.exception

        return self.result

    def add_done_callback(self, callback: Callable):
        """
        Add a callback to be called when the result is ready.

        Args:
            callback: Callback function
        """
        if self.done.is_set():
            # Result already available, call immediately
            try:
                callback(self)
            except Exception as e:
                print(f"Error in result callback: {e}")
        else:
            self.callbacks.append(callback)

    def is_done(self) -> bool:
        """
        Check if the result is ready.

        Returns:
            True if the result is ready, False otherwise
        """
        return self.done.is_set()


class AsyncBarrier:
    """
    Barrier for synchronizing multiple asynchronous operations.
    Similar to threading.Barrier but with callback support.
    """

    def __init__(self, parties: int, callback: Optional[Callable] = None):
        """
        Initialize the async barrier.

        Args:
            parties: Number of threads that must call wait()
            callback: Optional function to call when the barrier is crossed
        """
        self.parties = parties
        self.callback = callback
        self.count = 0
        self.generation = 0
        self.lock = threading.Lock()
        self.event = threading.Event()

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for all parties to reach the barrier.

        Args:
            timeout: Timeout in seconds

        Returns:
            True if the barrier was crossed, False if timeout occurred
        """
        with self.lock:
            generation = self.generation
            self.count += 1

            if self.count == self.parties:
                # Last thread to arrive
                self.event.set()
                self.count = 0
                self.generation += 1

                if self.callback:
                    try:
                        self.callback()
                    except Exception as e:
                        print(f"Error in barrier callback: {e}")

                return True

        # Wait for the last thread
        success = self.event.wait(timeout=timeout)

        if success and self.generation != generation:
            # Barrier was crossed
            return True

        # Timeout
        return False

    def reset(self):
        """
        Reset the barrier.
        """
        with self.lock:
            self.count = 0
            self.generation += 1
            self.event.clear()

    def abort(self):
        """
        Abort the barrier, causing all waiting threads to receive a BrokenBarrierError.
        """
        with self.lock:
            self.count = 0
            self.generation += 1
            self.event.set()


class AsyncCache:
    """
    Cache for expensive asynchronous operations.
    Implements LRU caching with TTL support.
    """

    def __init__(self, max_size: int = 1000, ttl: Optional[float] = None):
        """
        Initialize the async cache.

        Args:
            max_size: Maximum number of items in the cache
            ttl: Time-to-live in seconds (None for no expiration)
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.lock = threading.Lock()

    def get(self, key: Any) -> Optional[Any]:
        """
        Get a value from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        with self.lock:
            if key not in self.cache:
                return None

            value, timestamp = self.cache[key]

            # Check if expired
            if self.ttl is not None and time.time() - timestamp > self.ttl:
                del self.cache[key]
                del self.access_times[key]
                return None

            # Update access time
            self.access_times[key] = time.time()

            return value

    def put(self, key: Any, value: Any):
        """
        Put a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        with self.lock:
            # Check if we need to evict items
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_lru()

            timestamp = time.time()
            self.cache[key] = (value, timestamp)
            self.access_times[key] = timestamp

    def _evict_lru(self):
        """
        Evict the least recently used item from the cache.
        """
        if not self.access_times:
            return

        # Find the least recently used key
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]

        # Remove from cache
        del self.cache[lru_key]
        del self.access_times[lru_key]

    def clear(self):
        """
        Clear the cache.
        """
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

    def remove(self, key: Any) -> bool:
        """
        Remove a key from the cache.

        Args:
            key: Cache key

        Returns:
            True if the key was removed, False if not found
        """
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                del self.access_times[key]
                return True
            return False

    def size(self) -> int:
        """
        Get the current size of the cache.

        Returns:
            Number of items in the cache
        """
        with self.lock:
            return len(self.cache)


def async_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    exceptions: Tuple[Exception] = (Exception,),
):
    """
    Decorator for retrying a function on failure.

    Args:
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds
        exceptions: Tuple of exceptions to catch

    Returns:
        Decorated function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt >= max_retries:
                        raise
                    print(f"Retry {attempt + 1}/{max_retries} after error: {e}")
                    time.sleep(retry_delay * (2**attempt))  # Exponential backoff

        return wrapper

    return decorator


def async_timeout(timeout: float):
    """
    Decorator for adding a timeout to a function.

    Args:
        timeout: Timeout in seconds

    Returns:
        Decorated function
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            exception = [None]
            completed = [False]

            def target():
                try:
                    result[0] = func(*args, **kwargs)
                    completed[0] = True
                except Exception as e:
                    exception[0] = e
                    completed[0] = True

            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout)

            if not completed[0]:
                raise TimeoutError(f"Function timed out after {timeout} seconds")

            if exception[0]:
                raise exception[0]

            return result[0]

        return wrapper

    return decorator
