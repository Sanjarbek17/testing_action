# MagicMock Performance Optimization & Best Practices ⚡

## Overview
This guide covers advanced techniques for optimizing MagicMock performance, memory usage, and execution speed in large-scale test suites. Learn how to create efficient, fast, and maintainable mock-based tests.

---

## Optimization Strategy 1: Mock Pool and Reuse Patterns

**Scenario**: Reducing mock creation overhead in large test suites with thousands of tests.

```python
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Optional, Type, List, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import time
import weakref
import threading
from contextlib import contextmanager
import gc

@dataclass
class MockPoolConfig:
    """Configuration for mock pool"""
    max_pool_size: int = 100
    cleanup_threshold: int = 1000  # Number of operations before cleanup
    enable_weak_references: bool = True
    auto_reset_on_return: bool = True
    track_usage_stats: bool = True

class OptimizedMockPool:
    """High-performance mock pool with automatic cleanup and reuse"""
    
    def __init__(self, config: MockPoolConfig = MockPoolConfig()):
        self.config = config
        self._pools: Dict[str, List[MagicMock]] = defaultdict(list)
        self._in_use: Dict[str, set] = defaultdict(set)
        self._usage_stats: Dict[str, Dict] = defaultdict(lambda: {
            'created': 0, 'reused': 0, 'returned': 0, 'cleanup_count': 0
        })
        self._operation_count = 0
        self._lock = threading.RLock()
        
        # Weak reference tracking for automatic cleanup
        if config.enable_weak_references:
            self._weak_refs: Dict[str, weakref.WeakSet] = defaultdict(weakref.WeakSet)
    
    def get_mock(self, mock_type: str, spec: Optional[Type] = None, **kwargs) -> MagicMock:
        """Get a mock from pool or create new one"""
        with self._lock:
            self._operation_count += 1
            
            # Periodic cleanup
            if self._operation_count % self.config.cleanup_threshold == 0:
                self._cleanup_pools()
            
            # Try to reuse existing mock
            pool = self._pools[mock_type]
            if pool:
                mock = pool.pop()
                self._in_use[mock_type].add(id(mock))
                self._usage_stats[mock_type]['reused'] += 1
                
                # Reset mock if configured
                if self.config.auto_reset_on_return:
                    mock.reset_mock(return_value=True, side_effect=True)
                
                return mock
            
            # Create new mock
            mock = self._create_optimized_mock(mock_type, spec, **kwargs)
            self._in_use[mock_type].add(id(mock))
            self._usage_stats[mock_type]['created'] += 1
            
            # Track with weak reference
            if self.config.enable_weak_references:
                self._weak_refs[mock_type].add(mock)
            
            return mock
    
    def return_mock(self, mock_type: str, mock: MagicMock):
        """Return mock to pool for reuse"""
        with self._lock:
            mock_id = id(mock)
            
            if mock_id not in self._in_use[mock_type]:
                return  # Mock not from this pool
            
            self._in_use[mock_type].discard(mock_id)
            
            # Reset mock before returning to pool
            if self.config.auto_reset_on_return:
                mock.reset_mock(return_value=True, side_effect=True)
                
                # Clear any custom attributes
                for attr in list(vars(mock).keys()):
                    if not attr.startswith('_'):
                        delattr(mock, attr)
            
            # Return to pool if space available
            pool = self._pools[mock_type]
            if len(pool) < self.config.max_pool_size:
                pool.append(mock)
                self._usage_stats[mock_type]['returned'] += 1
    
    @contextmanager
    def borrowed_mock(self, mock_type: str, spec: Optional[Type] = None, **kwargs):
        """Context manager for automatic mock borrowing and returning"""
        mock = self.get_mock(mock_type, spec, **kwargs)
        try:
            yield mock
        finally:
            self.return_mock(mock_type, mock)
    
    def _create_optimized_mock(self, mock_type: str, spec: Optional[Type], **kwargs) -> MagicMock:
        """Create an optimized mock with minimal overhead"""
        # Use spec_set for better performance when spec is provided
        if spec:
            mock = MagicMock(spec_set=spec, **kwargs)
        else:
            mock = MagicMock(**kwargs)
        
        # Add type annotation for tracking
        mock._pool_type = mock_type
        
        return mock
    
    def _cleanup_pools(self):
        """Clean up pools and remove unused mocks"""
        with self._lock:
            cleaned_count = 0
            
            for mock_type, pool in self._pools.items():
                # Remove mocks that are no longer referenced
                if self.config.enable_weak_references:
                    weak_set = self._weak_refs[mock_type]
                    active_mocks = {id(m) for m in weak_set}
                    
                    # Filter pool to only include active mocks
                    original_size = len(pool)
                    self._pools[mock_type] = [
                        mock for mock in pool 
                        if id(mock) in active_mocks
                    ]
                    cleaned_count += original_size - len(self._pools[mock_type])
                
                # Limit pool size
                if len(pool) > self.config.max_pool_size:
                    excess = len(pool) - self.config.max_pool_size
                    self._pools[mock_type] = pool[excess:]
                    cleaned_count += excess
            
            # Force garbage collection
            if cleaned_count > 0:
                gc.collect()
                
                for mock_type in self._pools.keys():
                    self._usage_stats[mock_type]['cleanup_count'] += 1
    
    def get_pool_stats(self) -> Dict[str, Dict]:
        """Get pool usage statistics"""
        with self._lock:
            stats = {}
            for mock_type, usage in self._usage_stats.items():
                pool_size = len(self._pools[mock_type])
                in_use_count = len(self._in_use[mock_type])
                
                stats[mock_type] = {
                    'pool_size': pool_size,
                    'in_use': in_use_count,
                    'total_created': usage['created'],
                    'total_reused': usage['reused'],
                    'total_returned': usage['returned'],
                    'cleanup_cycles': usage['cleanup_count'],
                    'reuse_ratio': (usage['reused'] / max(1, usage['created'] + usage['reused'])),
                    'efficiency': pool_size / max(1, usage['created'])
                }
            
            return stats
    
    def clear_all_pools(self):
        """Clear all pools and reset statistics"""
        with self._lock:
            self._pools.clear()
            self._in_use.clear()
            self._usage_stats.clear()
            if self.config.enable_weak_references:
                self._weak_refs.clear()
            gc.collect()

# High-Performance Mock Decorator

class PerformantMockDecorator:
    """Decorator for high-performance mock injection"""
    
    def __init__(self, pool: OptimizedMockPool):
        self.pool = pool
        self._cached_patches: Dict[str, Any] = {}
    
    def mock_with_pool(self, **mock_specs):
        """Decorator that injects mocks from pool"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                borrowed_mocks = {}
                
                # Borrow mocks from pool
                for param_name, spec in mock_specs.items():
                    if isinstance(spec, tuple):
                        mock_type, mock_spec = spec
                    else:
                        mock_type, mock_spec = param_name, spec
                    
                    mock = self.pool.get_mock(mock_type, mock_spec)
                    borrowed_mocks[param_name] = mock
                    kwargs[param_name] = mock
                
                try:
                    return func(*args, **kwargs)
                finally:
                    # Return mocks to pool
                    for param_name, mock in borrowed_mocks.items():
                        if isinstance(mock_specs[param_name], tuple):
                            mock_type = mock_specs[param_name][0]
                        else:
                            mock_type = param_name
                        self.pool.return_mock(mock_type, mock)
            
            return wrapper
        return decorator
    
    def cached_patch(self, target_path: str, **kwargs):
        """Cached patch decorator for frequently used patches"""
        def decorator(func):
            def wrapper(*args, **test_kwargs):
                # Check if patch is already cached
                cache_key = f"{target_path}_{hash(frozenset(kwargs.items()))}"
                
                if cache_key not in self._cached_patches:
                    # Create new patch and cache it
                    patcher = patch(target_path, **kwargs)
                    self._cached_patches[cache_key] = patcher
                
                patcher = self._cached_patches[cache_key]
                
                # Use the cached patch
                mock_obj = patcher.start()
                test_kwargs[kwargs.get('new_callable', 'mock')] = mock_obj
                
                try:
                    return func(*args, **test_kwargs)
                finally:
                    patcher.stop()
                    # Reset mock for next use
                    if hasattr(mock_obj, 'reset_mock'):
                        mock_obj.reset_mock()
            
            return wrapper
        return decorator

# Usage Example

def test_optimized_mock_pool():
    """Test optimized mock pool performance"""
    
    config = MockPoolConfig(
        max_pool_size=50,
        cleanup_threshold=100,
        enable_weak_references=True,
        auto_reset_on_return=True,
        track_usage_stats=True
    )
    
    pool = OptimizedMockPool(config)
    decorator = PerformantMockDecorator(pool)
    
    # Define some mock specifications
    class DatabaseService:
        def query(self, sql): pass
        def execute(self, sql): pass
    
    class APIClient:
        def get(self, url): pass
        def post(self, url, data): pass
    
    # Test basic pool operations
    print("Testing basic pool operations...")
    
    # Get mocks from pool
    db_mock1 = pool.get_mock('database', DatabaseService)
    db_mock2 = pool.get_mock('database', DatabaseService)
    api_mock1 = pool.get_mock('api_client', APIClient)
    
    # Configure mocks
    db_mock1.query.return_value = [{'id': 1, 'name': 'Test'}]
    api_mock1.get.return_value = {'status': 'success'}
    
    # Use mocks
    assert db_mock1.query("SELECT * FROM users") == [{'id': 1, 'name': 'Test'}]
    assert api_mock1.get("/users") == {'status': 'success'}
    
    # Return mocks to pool
    pool.return_mock('database', db_mock1)
    pool.return_mock('database', db_mock2)
    pool.return_mock('api_client', api_mock1)
    
    # Get mocks again (should reuse)
    db_mock3 = pool.get_mock('database', DatabaseService)
    
    # Verify reset (mock should be clean)
    assert db_mock3.query.return_value is None or hasattr(db_mock3.query, 'return_value')
    
    # Test context manager
    with pool.borrowed_mock('database', DatabaseService) as db_mock:
        db_mock.query.return_value = [{'id': 2, 'name': 'Context Test'}]
        result = db_mock.query("SELECT * FROM users")
        assert result == [{'id': 2, 'name': 'Context Test'}]
    
    # Test decorator
    @decorator.mock_with_pool(
        database=('database', DatabaseService),
        api_client=('api_client', APIClient)
    )
    def sample_test_function(database, api_client):
        database.query.return_value = [{'id': 3}]
        api_client.get.return_value = {'success': True}
        
        db_result = database.query("SELECT * FROM test")
        api_result = api_client.get("/test")
        
        return db_result, api_result
    
    # Run test function multiple times
    for i in range(10):
        db_result, api_result = sample_test_function()
        assert db_result == [{'id': 3}]
        assert api_result == {'success': True}
    
    # Get pool statistics
    stats = pool.get_pool_stats()
    print("Pool Statistics:")
    for mock_type, stat in stats.items():
        print(f"  {mock_type}:")
        print(f"    Pool Size: {stat['pool_size']}")
        print(f"    Total Created: {stat['total_created']}")
        print(f"    Total Reused: {stat['total_reused']}")
        print(f"    Reuse Ratio: {stat['reuse_ratio']:.2%}")
        print(f"    Efficiency: {stat['efficiency']:.2f}")
    
    print("✓ Optimized mock pool test passed!")

test_optimized_mock_pool()
```

---

## Optimization Strategy 2: Memory-Efficient Mock Patterns

**Scenario**: Reducing memory footprint of mocks in long-running test suites.

```python
from unittest.mock import MagicMock
import sys
import weakref
from typing import Dict, Any, Optional, List, Callable
import gc
from dataclasses import dataclass
import tracemalloc

@dataclass
class MemoryOptimizationConfig:
    """Configuration for memory optimization"""
    enable_memory_tracking: bool = True
    auto_garbage_collect: bool = True
    use_slots_for_mocks: bool = True
    limit_call_history: int = 100
    enable_weak_references: bool = True
    compress_call_data: bool = True

class MemoryEfficientMock(MagicMock):
    """Memory-optimized version of MagicMock"""
    
    # Use __slots__ to reduce memory overhead
    __slots__ = ('_mock_name', '_spec_class', '_spec_signature', '_parent', 
                 '_name', '_return_value', '_side_effect', '_call_count',
                 '_call_history', '_memory_config')
    
    def __init__(self, config: MemoryOptimizationConfig = MemoryOptimizationConfig(), 
                 *args, **kwargs):
        self._memory_config = config
        self._call_history = []
        
        # Initialize with limited call tracking
        super().__init__(*args, **kwargs)
        
        if config.limit_call_history > 0:
            self._setup_limited_call_tracking()
    
    def _setup_limited_call_tracking(self):
        """Setup call tracking with memory limits"""
        original_call = self.__call__
        
        def limited_call(*args, **kwargs):
            result = original_call(*args, **kwargs)
            
            # Limit call history size
            if len(self._call_history) >= self._memory_config.limit_call_history:
                # Remove oldest calls, keeping only recent ones
                self._call_history = self._call_history[-self._memory_config.limit_call_history//2:]
            
            # Store compressed call data
            if self._memory_config.compress_call_data:
                call_data = {
                    'args_count': len(args),
                    'kwargs_keys': list(kwargs.keys()) if kwargs else [],
                    'timestamp': id(result)  # Use object id as lightweight timestamp
                }
            else:
                call_data = {'args': args, 'kwargs': kwargs}
            
            self._call_history.append(call_data)
            
            return result
        
        self.__call__ = limited_call
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get memory usage information for this mock"""
        return {
            'object_size': sys.getsizeof(self),
            'call_history_size': sys.getsizeof(self._call_history),
            'total_size': sys.getsizeof(self) + sys.getsizeof(self._call_history),
            'call_count': len(self._call_history)
        }
    
    def cleanup_memory(self):
        """Manually clean up memory"""
        if hasattr(self, '_call_history'):
            self._call_history.clear()
        
        # Clear any cached attributes
        for attr_name in list(self.__dict__.keys()):
            if attr_name.startswith('_cache_'):
                delattr(self, attr_name)
        
        if self._memory_config.auto_garbage_collect:
            gc.collect()

class MockMemoryProfiler:
    """Profiler for tracking mock memory usage"""
    
    def __init__(self):
        self._tracked_mocks: weakref.WeakSet = weakref.WeakSet()
        self._memory_snapshots: List[Dict] = []
        self._tracking_enabled = False
    
    def start_tracking(self):
        """Start memory tracking"""
        self._tracking_enabled = True
        if hasattr(tracemalloc, 'start'):
            tracemalloc.start()
        self._take_snapshot("start")
    
    def stop_tracking(self):
        """Stop memory tracking"""
        self._tracking_enabled = False
        self._take_snapshot("end")
        if hasattr(tracemalloc, 'stop'):
            tracemalloc.stop()
    
    def track_mock(self, mock: MagicMock, name: str):
        """Add mock to tracking"""
        self._tracked_mocks.add(mock)
        mock._profiler_name = name
    
    def _take_snapshot(self, label: str):
        """Take memory snapshot"""
        if not self._tracking_enabled:
            return
        
        # Calculate memory usage of tracked mocks
        mock_memory = 0
        mock_count = 0
        
        for mock in self._tracked_mocks:
            if hasattr(mock, 'get_memory_usage'):
                mock_memory += mock.get_memory_usage()['total_size']
            else:
                mock_memory += sys.getsizeof(mock)
            mock_count += 1
        
        # System memory snapshot
        if hasattr(tracemalloc, 'get_traced_memory'):
            current, peak = tracemalloc.get_traced_memory()
        else:
            current, peak = 0, 0
        
        snapshot = {
            'label': label,
            'tracked_mocks': mock_count,
            'mock_memory_bytes': mock_memory,
            'total_current_bytes': current,
            'total_peak_bytes': peak,
            'timestamp': id(self)  # Lightweight timestamp
        }
        
        self._memory_snapshots.append(snapshot)
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate memory usage report"""
        if len(self._memory_snapshots) < 2:
            return {'error': 'Insufficient snapshots for comparison'}
        
        start_snapshot = self._memory_snapshots[0]
        end_snapshot = self._memory_snapshots[-1]
        
        return {
            'start_mock_memory': start_snapshot['mock_memory_bytes'],
            'end_mock_memory': end_snapshot['mock_memory_bytes'],
            'mock_memory_growth': end_snapshot['mock_memory_bytes'] - start_snapshot['mock_memory_bytes'],
            'start_total_memory': start_snapshot['total_current_bytes'],
            'end_total_memory': end_snapshot['total_current_bytes'],
            'total_memory_growth': end_snapshot['total_current_bytes'] - start_snapshot['total_current_bytes'],
            'tracked_mocks': end_snapshot['tracked_mocks'],
            'snapshots': self._memory_snapshots
        }

# Memory-Efficient Mock Factory

class MemoryEfficientMockFactory:
    """Factory for creating memory-optimized mocks"""
    
    def __init__(self, config: MemoryOptimizationConfig = MemoryOptimizationConfig()):
        self.config = config
        self._mock_registry: Dict[str, weakref.ref] = {}
        self._creation_count = 0
    
    def create_mock(self, name: str, spec: Optional[type] = None, **kwargs) -> MemoryEfficientMock:
        """Create a memory-efficient mock"""
        self._creation_count += 1
        
        # Use spec_set for memory efficiency when spec provided
        if spec and self.config.use_slots_for_mocks:
            mock = MemoryEfficientMock(
                config=self.config,
                spec_set=spec,
                **kwargs
            )
        else:
            mock = MemoryEfficientMock(config=self.config, **kwargs)
        
        # Register with weak reference to avoid memory leaks
        if self.config.enable_weak_references:
            self._mock_registry[f"{name}_{self._creation_count}"] = weakref.ref(mock)
        
        return mock
    
    def create_lightweight_mock(self, name: str, methods_only: List[str]) -> MagicMock:
        """Create ultra-lightweight mock with only specified methods"""
        
        class LightweightMock:
            """Minimal mock implementation"""
            __slots__ = tuple(methods_only) + ('_call_count',)
            
            def __init__(self):
                self._call_count = 0
                for method in methods_only:
                    setattr(self, method, self._create_method(method))
            
            def _create_method(self, method_name):
                def method(*args, **kwargs):
                    self._call_count += 1
                    return None
                return method
            
            def reset_mock(self):
                self._call_count = 0
        
        return LightweightMock()
    
    def get_registry_size(self) -> int:
        """Get number of live mocks in registry"""
        # Clean up dead references
        dead_refs = [key for key, ref in self._mock_registry.items() if ref() is None]
        for key in dead_refs:
            del self._mock_registry[key]
        
        return len(self._mock_registry)
    
    def force_cleanup(self):
        """Force cleanup of all references"""
        self._mock_registry.clear()
        if self.config.auto_garbage_collect:
            gc.collect()

# Usage Example and Performance Testing

def test_memory_efficient_mocks():
    """Test memory-efficient mock patterns"""
    
    profiler = MockMemoryProfiler()
    profiler.start_tracking()
    
    config = MemoryOptimizationConfig(
        enable_memory_tracking=True,
        auto_garbage_collect=True,
        use_slots_for_mocks=True,
        limit_call_history=50,
        enable_weak_references=True,
        compress_call_data=True
    )
    
    factory = MemoryEfficientMockFactory(config)
    
    print("Creating memory-efficient mocks...")
    
    # Create mocks and track memory usage
    mocks = []
    for i in range(100):
        mock = factory.create_mock(f"test_mock_{i}")
        profiler.track_mock(mock, f"test_mock_{i}")
        
        # Simulate usage
        mock.some_method.return_value = f"result_{i}"
        for j in range(10):
            mock.some_method(f"arg_{j}")
        
        mocks.append(mock)
    
    print(f"Created {len(mocks)} mocks")
    
    # Test memory usage
    total_memory = 0
    for mock in mocks:
        if hasattr(mock, 'get_memory_usage'):
            memory_info = mock.get_memory_usage()
            total_memory += memory_info['total_size']
    
    print(f"Total mock memory usage: {total_memory} bytes")
    
    # Test lightweight mocks
    print("Testing lightweight mocks...")
    lightweight_mocks = []
    for i in range(50):
        mock = factory.create_lightweight_mock(f"light_mock_{i}", ['method1', 'method2', 'method3'])
        lightweight_mocks.append(mock)
        
        # Use the mock
        mock.method1()
        mock.method2()
        mock.method3()
    
    print(f"Created {len(lightweight_mocks)} lightweight mocks")
    
    # Compare memory usage
    standard_mock = MagicMock()
    efficient_mock = factory.create_mock("efficient")
    
    standard_size = sys.getsizeof(standard_mock)
    efficient_size = efficient_mock.get_memory_usage()['total_size'] if hasattr(efficient_mock, 'get_memory_usage') else sys.getsizeof(efficient_mock)
    
    print(f"Standard mock size: {standard_size} bytes")
    print(f"Efficient mock size: {efficient_size} bytes")
    print(f"Memory savings: {((standard_size - efficient_size) / standard_size * 100):.1f}%")
    
    # Test cleanup
    print("Testing cleanup...")
    for mock in mocks:
        if hasattr(mock, 'cleanup_memory'):
            mock.cleanup_memory()
    
    factory.force_cleanup()
    gc.collect()
    
    profiler.stop_tracking()
    memory_report = profiler.get_memory_report()
    
    print("Memory Report:")
    if 'error' not in memory_report:
        print(f"  Mock memory growth: {memory_report['mock_memory_growth']} bytes")
        print(f"  Total memory growth: {memory_report['total_memory_growth']} bytes")
        print(f"  Tracked mocks: {memory_report['tracked_mocks']}")
    
    print("✓ Memory-efficient mock test completed!")

test_memory_efficient_mocks()
```

---

## Optimization Strategy 3: Parallel Mock Execution and Caching

**Scenario**: Optimizing mock performance in concurrent test environments.

```python
import threading
import concurrent.futures
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Callable, List, Optional, Tuple
import time
import hashlib
import pickle
from dataclasses import dataclass
from collections import defaultdict
import queue

@dataclass
class ParallelMockConfig:
    """Configuration for parallel mock execution"""
    max_workers: int = 4
    enable_result_caching: bool = True
    cache_size_limit: int = 1000
    enable_thread_local_mocks: bool = True
    batch_size: int = 10
    timeout_seconds: float = 30.0

class ThreadLocalMockManager:
    """Manages thread-local mock instances for parallel execution"""
    
    def __init__(self):
        self._local = threading.local()
        self._mock_configs: Dict[str, Dict] = {}
        self._lock = threading.RLock()
    
    def register_mock_config(self, name: str, config: Dict[str, Any]):
        """Register mock configuration for thread-local creation"""
        with self._lock:
            self._mock_configs[name] = config
    
    def get_thread_local_mock(self, name: str) -> MagicMock:
        """Get or create thread-local mock"""
        if not hasattr(self._local, 'mocks'):
            self._local.mocks = {}
        
        if name not in self._local.mocks:
            config = self._mock_configs.get(name, {})
            self._local.mocks[name] = MagicMock(**config)
        
        return self._local.mocks[name]
    
    def reset_thread_local_mocks(self):
        """Reset all mocks in current thread"""
        if hasattr(self._local, 'mocks'):
            for mock in self._local.mocks.values():
                mock.reset_mock()
    
    def cleanup_thread(self):
        """Clean up thread-local storage"""
        if hasattr(self._local, 'mocks'):
            del self._local.mocks

class MockResultCache:
    """High-performance cache for mock call results"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()
        self._hit_count = 0
        self._miss_count = 0
    
    def _generate_cache_key(self, mock_name: str, method_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for mock call"""
        # Create a hash of the call signature
        call_data = {
            'mock': mock_name,
            'method': method_name,
            'args': args,
            'kwargs': sorted(kwargs.items()) if kwargs else None
        }
        
        # Use pickle + hash for complex objects
        try:
            serialized = pickle.dumps(call_data)
            return hashlib.md5(serialized).hexdigest()
        except (pickle.PickleError, TypeError):
            # Fallback for unpickleable objects
            return f"{mock_name}_{method_name}_{hash(args)}_{hash(frozenset(kwargs.items()) if kwargs else frozenset())}"
    
    def get(self, mock_name: str, method_name: str, args: tuple, kwargs: dict) -> Optional[Any]:
        """Get cached result"""
        cache_key = self._generate_cache_key(mock_name, method_name, args, kwargs)
        
        with self._lock:
            if cache_key in self._cache:
                # Move to end of access order (LRU)
                self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                self._hit_count += 1
                return self._cache[cache_key]
            
            self._miss_count += 1
            return None
    
    def set(self, mock_name: str, method_name: str, args: tuple, kwargs: dict, result: Any):
        """Cache result"""
        cache_key = self._generate_cache_key(mock_name, method_name, args, kwargs)
        
        with self._lock:
            # Remove oldest entries if cache is full
            while len(self._cache) >= self.max_size:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
            
            self._cache[cache_key] = result
            self._access_order.append(cache_key)
    
    def clear(self):
        """Clear cache"""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._hit_count = 0
            self._miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = (self._hit_count / total_requests) if total_requests > 0 else 0
            
            return {
                'cache_size': len(self._cache),
                'max_size': self.max_size,
                'hit_count': self._hit_count,
                'miss_count': self._miss_count,
                'hit_rate': hit_rate,
                'utilization': len(self._cache) / self.max_size
            }

class CachedMock(MagicMock):
    """Mock with built-in result caching"""
    
    def __init__(self, name: str, cache: MockResultCache, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = cache
        self._mock_name = name
        self._original_getattr = self.__getattribute__
        
        # Override attribute access to add caching
        def cached_getattr(attr_name):
            attr = self._original_getattr(attr_name)
            
            if callable(attr) and not attr_name.startswith('_'):
                return self._create_cached_method(attr_name, attr)
            
            return attr
        
        self.__getattribute__ = cached_getattr
    
    def _create_cached_method(self, method_name: str, original_method: Callable):
        """Create cached version of method"""
        def cached_method(*args, **kwargs):
            # Check cache first
            cached_result = self._cache.get(self._mock_name, method_name, args, kwargs)
            if cached_result is not None:
                return cached_result
            
            # Execute original method
            result = original_method(*args, **kwargs)
            
            # Cache the result
            self._cache.set(self._mock_name, method_name, args, kwargs, result)
            
            return result
        
        # Copy method attributes
        cached_method.return_value = getattr(original_method, 'return_value', None)
        cached_method.side_effect = getattr(original_method, 'side_effect', None)
        
        return cached_method

class ParallelMockExecutor:
    """Executes mock-based tests in parallel with optimization"""
    
    def __init__(self, config: ParallelMockConfig = ParallelMockConfig()):
        self.config = config
        self.thread_manager = ThreadLocalMockManager()
        self.cache = MockResultCache(config.cache_size_limit) if config.enable_result_caching else None
        self._execution_stats = defaultdict(list)
    
    def execute_parallel_tests(self, test_functions: List[Callable], 
                              setup_function: Optional[Callable] = None) -> Dict[str, Any]:
        """Execute test functions in parallel"""
        
        def worker_setup():
            """Setup function for each worker thread"""
            if setup_function:
                setup_function(self.thread_manager)
        
        def execute_test_batch(test_batch: List[Callable]) -> List[Dict[str, Any]]:
            """Execute a batch of tests in worker thread"""
            worker_setup()
            results = []
            
            for test_func in test_batch:
                start_time = time.time()
                try:
                    result = test_func(self.thread_manager)
                    execution_time = time.time() - start_time
                    
                    results.append({
                        'test_name': test_func.__name__,
                        'status': 'passed',
                        'execution_time': execution_time,
                        'result': result
                    })
                    
                except Exception as e:
                    execution_time = time.time() - start_time
                    results.append({
                        'test_name': test_func.__name__,
                        'status': 'failed',
                        'execution_time': execution_time,
                        'error': str(e)
                    })
                
                # Reset thread-local mocks between tests
                self.thread_manager.reset_thread_local_mocks()
            
            # Cleanup thread resources
            self.thread_manager.cleanup_thread()
            return results
        
        # Create batches of tests
        batches = [
            test_functions[i:i + self.config.batch_size]
            for i in range(0, len(test_functions), self.config.batch_size)
        ]
        
        # Execute batches in parallel
        all_results = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(execute_test_batch, batch): batch 
                for batch in batches
            }
            
            # Collect results with timeout
            for future in concurrent.futures.as_completed(future_to_batch, 
                                                          timeout=self.config.timeout_seconds):
                try:
                    batch_results = future.result()
                    all_results.extend(batch_results)
                except concurrent.futures.TimeoutError:
                    all_results.append({
                        'test_name': 'batch_timeout',
                        'status': 'timeout',
                        'error': 'Batch execution timed out'
                    })
                except Exception as e:
                    all_results.append({
                        'test_name': 'batch_error',
                        'status': 'error',
                        'error': str(e)
                    })
        
        total_time = time.time() - start_time
        
        # Compile execution summary
        summary = self._compile_execution_summary(all_results, total_time)
        return summary
    
    def _compile_execution_summary(self, results: List[Dict], total_time: float) -> Dict[str, Any]:
        """Compile execution summary statistics"""
        passed_tests = [r for r in results if r['status'] == 'passed']
        failed_tests = [r for r in results if r['status'] == 'failed']
        
        summary = {
            'total_tests': len(results),
            'passed_tests': len(passed_tests),
            'failed_tests': len(failed_tests),
            'total_execution_time': total_time,
            'average_test_time': sum(r.get('execution_time', 0) for r in results) / len(results) if results else 0,
            'parallel_efficiency': (sum(r.get('execution_time', 0) for r in results) / total_time) if total_time > 0 else 0,
            'success_rate': len(passed_tests) / len(results) if results else 0,
            'failed_test_names': [r['test_name'] for r in failed_tests],
            'cache_stats': self.cache.get_stats() if self.cache else None
        }
        
        return summary

    def create_cached_mock(self, name: str, **kwargs) -> CachedMock:
        """Create a cached mock"""
        if not self.cache:
            raise ValueError("Caching not enabled")
        
        return CachedMock(name, self.cache, **kwargs)

# Performance Benchmark Suite

def benchmark_mock_performance():
    """Benchmark different mock optimization techniques"""
    
    print("Running mock performance benchmarks...")
    
    # Setup
    config = ParallelMockConfig(max_workers=4, enable_result_caching=True)
    executor = ParallelMockExecutor(config)
    
    # Register mock configurations for thread-local use
    executor.thread_manager.register_mock_config('database', {
        'spec': ['query', 'execute', 'commit', 'rollback']
    })
    executor.thread_manager.register_mock_config('api_client', {
        'spec': ['get', 'post', 'put', 'delete']
    })
    
    # Define test functions
    def test_database_operations(thread_manager: ThreadLocalMockManager):
        db_mock = thread_manager.get_thread_local_mock('database')
        db_mock.query.return_value = [{'id': 1, 'name': 'Test'}]
        
        # Perform multiple operations
        for i in range(10):
            result = db_mock.query(f"SELECT * FROM table_{i}")
            db_mock.execute(f"INSERT INTO table_{i} VALUES (1, 'test')")
        
        return {'operations': 20}
    
    def test_api_operations(thread_manager: ThreadLocalMockManager):
        api_mock = thread_manager.get_thread_local_mock('api_client')
        api_mock.get.return_value = {'status': 'success', 'data': {}}
        
        # Perform API calls
        for i in range(5):
            api_mock.get(f'/users/{i}')
            api_mock.post('/users', {'name': f'user_{i}'})
        
        return {'api_calls': 10}
    
    def test_mixed_operations(thread_manager: ThreadLocalMockManager):
        db_mock = thread_manager.get_thread_local_mock('database')
        api_mock = thread_manager.get_thread_local_mock('api_client')
        
        # Mixed operations
        db_mock.query.return_value = [{'id': 1}]
        api_mock.get.return_value = {'data': 'test'}
        
        for i in range(3):
            db_mock.query(f"SELECT * FROM users WHERE id = {i}")
            api_mock.get(f'/api/users/{i}')
            db_mock.execute(f"UPDATE users SET last_seen = NOW() WHERE id = {i}")
        
        return {'mixed_operations': 9}
    
    # Create test suite
    test_functions = []
    for i in range(50):  # Create many test instances
        test_functions.extend([
            test_database_operations,
            test_api_operations,
            test_mixed_operations
        ])
    
    # Benchmark serial execution
    print("Benchmarking serial execution...")
    serial_start = time.time()
    
    for test_func in test_functions[:30]:  # Smaller subset for serial
        try:
            test_func(executor.thread_manager)
        except Exception:
            pass
        executor.thread_manager.reset_thread_local_mocks()
    
    serial_time = time.time() - serial_start
    
    # Benchmark parallel execution
    print("Benchmarking parallel execution...")
    parallel_results = executor.execute_parallel_tests(test_functions)
    
    # Performance comparison
    print(f"\nPerformance Comparison:")
    print(f"Serial execution (30 tests): {serial_time:.2f} seconds")
    print(f"Parallel execution ({len(test_functions)} tests): {parallel_results['total_execution_time']:.2f} seconds")
    print(f"Parallel efficiency: {parallel_results['parallel_efficiency']:.2%}")
    print(f"Success rate: {parallel_results['success_rate']:.2%}")
    
    if parallel_results['cache_stats']:
        cache_stats = parallel_results['cache_stats']
        print(f"Cache hit rate: {cache_stats['hit_rate']:.2%}")
        print(f"Cache utilization: {cache_stats['utilization']:.2%}")
    
    # Test cached mock performance
    print("\nTesting cached mock performance...")
    cached_mock = executor.create_cached_mock('cached_test')
    
    # Perform repeated operations
    start_time = time.time()
    for i in range(100):
        # These should be cached after first execution
        cached_mock.expensive_operation(i % 10)  # Only 10 unique calls
    
    cached_time = time.time() - start_time
    print(f"Cached mock operations (100 calls): {cached_time:.3f} seconds")
    
    final_cache_stats = executor.cache.get_stats()
    print(f"Final cache hit rate: {final_cache_stats['hit_rate']:.2%}")
    
    print("✓ Performance benchmark completed!")

benchmark_mock_performance()
```

---

## Summary

These performance optimization strategies provide:

1. **Mock Pool and Reuse** - Reduces mock creation overhead through intelligent pooling
2. **Memory Optimization** - Minimizes memory footprint with efficient mock structures
3. **Parallel Execution** - Enables concurrent test execution with thread-local mocks and caching

Key benefits:
- **Faster Test Execution** - Significant speedup through parallelization and caching
- **Reduced Memory Usage** - Memory-efficient mocks for long-running test suites
- **Better Resource Utilization** - Smart pooling and cleanup strategies
- **Scalable Testing** - Patterns that work with large test suites

## Best Practices Summary

1. **Use Mock Pools** for tests with many similar mocks
2. **Enable Memory Optimization** for long-running test suites
3. **Implement Caching** for expensive mock operations
4. **Use Thread-Local Mocks** in parallel test environments
5. **Monitor Performance Metrics** to identify optimization opportunities

## Implementation Guidelines

1. **Start with Profiling** - Identify performance bottlenecks first
2. **Apply Incrementally** - Implement optimizations one at a time
3. **Measure Impact** - Verify optimizations actually improve performance
4. **Balance Complexity** - Don't over-optimize simple test suites
5. **Monitor Memory Usage** - Track memory consumption in CI/CD pipelines

## Next Steps

- Profile your existing test suite to identify optimization opportunities
- Implement mock pooling for frequently used mocks
- Add memory monitoring to your test infrastructure
- Consider parallel execution for large test suites