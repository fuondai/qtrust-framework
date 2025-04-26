"""
Unit tests for asynchronous utilities.

This module contains unit tests for the asynchronous utilities
in the QTrust blockchain sharding framework.
"""

import os
import sys
import unittest
import time
import threading
from typing import Dict, List, Any

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from qtrust.common.async_utils import AsyncEvent, AsyncResult, AsyncCache, AsyncProcessor


class TestAsyncEvent(unittest.TestCase):
    """Unit tests for the AsyncEvent class."""

    def test_event_creation(self):
        """Test creation of asynchronous events."""
        event = AsyncEvent("test_event", data={"key": "value"}, source="test_source")
        self.assertEqual(event.event_type, "test_event")
        self.assertEqual(event.data, {"key": "value"})
        self.assertEqual(event.source, "test_source")
        self.assertFalse(event.processed)
        self.assertIsNotNone(event.event_id)
        self.assertIsNotNone(event.timestamp)

    def test_callback_execution(self):
        """Test execution of event callbacks."""
        event = AsyncEvent("test_event")
        callback_executed = False
        
        def callback(evt):
            nonlocal callback_executed
            callback_executed = True
            self.assertEqual(evt, event)
        
        event.add_callback(callback)
        event.mark_processed()
        self.assertTrue(callback_executed)
        self.assertTrue(event.processed)

    def test_multiple_callbacks(self):
        """Test execution of multiple event callbacks."""
        event = AsyncEvent("test_event")
        callback_count = 0
        
        def callback1(evt):
            nonlocal callback_count
            callback_count += 1
            
        def callback2(evt):
            nonlocal callback_count
            callback_count += 2
            
        def callback3(evt):
            nonlocal callback_count
            callback_count += 3
        
        event.add_callback(callback1)
        event.add_callback(callback2)
        event.add_callback(callback3)
        event.mark_processed()
        self.assertEqual(callback_count, 6)


class TestAsyncResult(unittest.TestCase):
    """Unit tests for the AsyncResult class."""

    def test_result_creation(self):
        """Test creation of asynchronous results."""
        result = AsyncResult("test_operation")
        self.assertEqual(result.operation_id, "test_operation")
        self.assertFalse(result.completed)
        self.assertFalse(result.success)
        self.assertIsNone(result.result)
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.timestamp)
        self.assertIsNone(result.completion_time)

    def test_successful_result(self):
        """Test setting a successful result."""
        result = AsyncResult()
        result.set_result("test_result", success=True)
        self.assertTrue(result.completed)
        self.assertTrue(result.success)
        self.assertEqual(result.result, "test_result")
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.completion_time)

    def test_error_result(self):
        """Test setting an error result."""
        result = AsyncResult()
        error = Exception("Test error")
        result.set_error(error)
        self.assertTrue(result.completed)
        self.assertFalse(result.success)
        self.assertIsNone(result.result)
        self.assertEqual(result.error, error)
        self.assertIsNotNone(result.completion_time)

    def test_callback_execution(self):
        """Test execution of result callbacks."""
        result = AsyncResult()
        callback_executed = False
        
        def callback(res):
            nonlocal callback_executed
            callback_executed = True
            self.assertEqual(res, result)
        
        result.add_callback(callback)
        result.set_result("test_result")
        self.assertTrue(callback_executed)

    def test_wait_for_completion(self):
        """Test waiting for result completion."""
        result = AsyncResult()
        
        def complete_after_delay():
            time.sleep(0.1)
            result.set_result("test_result")
        
        thread = threading.Thread(target=complete_after_delay)
        thread.start()
        
        # Wait for completion
        completed = result.wait(timeout=1.0)
        self.assertTrue(completed)
        self.assertTrue(result.completed)
        self.assertEqual(result.result, "test_result")
        
        thread.join()


class TestAsyncCache(unittest.TestCase):
    """Unit tests for the AsyncCache class."""

    def test_cache_operations(self):
        """Test basic cache operations."""
        cache = AsyncCache(max_size=10, default_ttl=1.0)
        
        # Test put and get
        cache.put("key1", "value1")
        self.assertEqual(cache.get("key1"), "value1")
        
        # Test non-existent key
        self.assertIsNone(cache.get("key2"))
        
        # Test custom TTL
        cache.put("key3", "value3", ttl=0.1)
        self.assertEqual(cache.get("key3"), "value3")
        time.sleep(0.2)
        self.assertIsNone(cache.get("key3"))  # Should be expired
        
        # Test default TTL
        cache.put("key4", "value4")
        self.assertEqual(cache.get("key4"), "value4")
        time.sleep(1.1)
        self.assertIsNone(cache.get("key4"))  # Should be expired

    def test_cache_size_limit(self):
        """Test cache size limit enforcement."""
        cache = AsyncCache(max_size=3, default_ttl=10.0)
        
        # Fill cache
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.put("key3", "value3")
        
        # All keys should be present
        self.assertEqual(cache.get("key1"), "value1")
        self.assertEqual(cache.get("key2"), "value2")
        self.assertEqual(cache.get("key3"), "value3")
        
        # Add one more key, should evict oldest
        cache.put("key4", "value4")
        self.assertIsNone(cache.get("key1"))  # Should be evicted
        self.assertEqual(cache.get("key2"), "value2")
        self.assertEqual(cache.get("key3"), "value3")
        self.assertEqual(cache.get("key4"), "value4")

    def test_cache_management(self):
        """Test cache management operations."""
        cache = AsyncCache(max_size=10, default_ttl=10.0)
        
        # Test size
        self.assertEqual(cache.size(), 0)
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        self.assertEqual(cache.size(), 2)
        
        # Test remove
        self.assertTrue(cache.remove("key1"))
        self.assertEqual(cache.size(), 1)
        self.assertFalse(cache.remove("key1"))  # Already removed
        
        # Test clear
        cache.put("key3", "value3")
        self.assertEqual(cache.size(), 2)
        cache.clear()
        self.assertEqual(cache.size(), 0)


class TestAsyncProcessor(unittest.TestCase):
    """Unit tests for the AsyncProcessor class."""

    def test_processor_operations(self):
        """Test basic processor operations."""
        processor = AsyncProcessor(max_workers=2, queue_size=10)
        processor.start()
        
        try:
            # Define a simple task
            def task(arg):
                return arg * 2
            
            # Submit a task
            result = processor.submit(task, 5)
            self.assertTrue(result.wait(timeout=1.0))
            self.assertTrue(result.completed)
            self.assertTrue(result.success)
            self.assertEqual(result.result, 10)
            
            # Submit multiple tasks
            results = []
            for i in range(5):
                results.append(processor.submit(task, i))
                
            # Wait for all tasks to complete
            for result in results:
                self.assertTrue(result.wait(timeout=1.0))
                self.assertTrue(result.success)
                
            # Check results
            for i, result in enumerate(results):
                self.assertEqual(result.result, i * 2)
                
        finally:
            processor.stop()

    def test_task_prioritization(self):
        """Test task prioritization."""
        processor = AsyncProcessor(max_workers=1, queue_size=10)
        processor.start()
        
        try:
            # Define a task that records execution order
            execution_order = []
            
            def task(arg):
                execution_order.append(arg)
                time.sleep(0.1)  # Ensure tasks don't complete immediately
                return arg
            
            # Submit tasks with different priorities
            processor.submit(task, "low", priority=0)
            processor.submit(task, "medium", priority=5)
            processor.submit(task, "high", priority=10)
            
            # Wait for all tasks to complete
            time.sleep(0.5)
            
            # Check execution order (should be high, medium, low)
            self.assertEqual(execution_order, ["high", "medium", "low"])
                
        finally:
            processor.stop()

    def test_error_handling(self):
        """Test error handling in tasks."""
        processor = AsyncProcessor(max_workers=1, queue_size=10)
        processor.start()
        
        try:
            # Define a task that raises an exception
            def failing_task(arg):
                raise ValueError(f"Error in task: {arg}")
            
            # Submit the failing task
            result = processor.submit(failing_task, "test")
            self.assertTrue(result.wait(timeout=1.0))
            self.assertTrue(result.completed)
            self.assertFalse(result.success)
            self.assertIsInstance(result.error, ValueError)
            self.assertEqual(str(result.error), "Error in task: test")
                
        finally:
            processor.stop()


if __name__ == '__main__':
    unittest.main()
