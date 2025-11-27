# Advanced MagicMock Testing Patterns 🎖️

## Overview
This guide covers sophisticated testing patterns that professional developers use in production environments. These patterns solve complex testing challenges involving complex dependencies, state management, and integration scenarios.

---

## Pattern 1: Dynamic Mock Configuration

**Scenario**: Creating mocks that adapt their behavior based on input parameters.

```python
from unittest.mock import MagicMock
from typing import Dict, Any

class DynamicDatabaseMock(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tables = {
            'users': [
                {'id': 1, 'name': 'Alice', 'email': 'alice@test.com'},
                {'id': 2, 'name': 'Bob', 'email': 'bob@test.com'}
            ],
            'orders': [
                {'id': 101, 'user_id': 1, 'total': 99.99},
                {'id': 102, 'user_id': 2, 'total': 149.99}
            ]
        }
    
    def query(self, table: str, filters: Dict[str, Any] = None):
        """Dynamic query method that simulates database behavior"""
        if table not in self._tables:
            raise ValueError(f"Table '{table}' not found")
        
        data = self._tables[table]
        
        if filters:
            filtered_data = []
            for item in data:
                match = True
                for key, value in filters.items():
                    if key not in item or item[key] != value:
                        match = False
                        break
                if match:
                    filtered_data.append(item)
            return filtered_data
        
        return data

# Usage Example
def test_user_service():
    db_mock = DynamicDatabaseMock()
    
    # Test getting all users
    all_users = db_mock.query('users')
    assert len(all_users) == 2
    
    # Test filtering by ID
    user = db_mock.query('users', {'id': 1})
    assert user[0]['name'] == 'Alice'
    
    # Test non-existent table
    try:
        db_mock.query('invalid_table')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not found" in str(e)

test_user_service()
print("✓ Dynamic mock configuration test passed!")
```

---

## Pattern 2: State-Aware Mocks

**Scenario**: Mocks that maintain state across multiple calls, simulating real-world stateful services.

```python
from unittest.mock import MagicMock
from enum import Enum

class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"

class StatefulConnectionMock(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._state = ConnectionState.DISCONNECTED
        self._connected_time = None
        self._call_count = 0
        
        # Configure method behaviors
        self.connect.side_effect = self._connect
        self.disconnect.side_effect = self._disconnect
        self.send_data.side_effect = self._send_data
        self.get_status.side_effect = self._get_status
    
    def _connect(self):
        self._call_count += 1
        if self._state == ConnectionState.CONNECTED:
            return {"status": "already_connected"}
        
        # Simulate connection failure on third attempt
        if self._call_count == 3:
            self._state = ConnectionState.ERROR
            raise ConnectionError("Connection failed")
        
        self._state = ConnectionState.CONNECTED
        self._connected_time = "2023-01-01T10:00:00"
        return {"status": "connected", "time": self._connected_time}
    
    def _disconnect(self):
        if self._state != ConnectionState.CONNECTED:
            return {"status": "not_connected"}
        
        self._state = ConnectionState.DISCONNECTED
        self._connected_time = None
        return {"status": "disconnected"}
    
    def _send_data(self, data):
        if self._state != ConnectionState.CONNECTED:
            raise RuntimeError("Not connected")
        
        return {"status": "sent", "bytes": len(str(data))}
    
    def _get_status(self):
        return {
            "state": self._state.value,
            "connected_time": self._connected_time,
            "call_count": self._call_count
        }

# Usage Example
def test_connection_lifecycle():
    conn_mock = StatefulConnectionMock()
    
    # Test initial state
    status = conn_mock.get_status()
    assert status["state"] == "disconnected"
    
    # Test successful connection
    result = conn_mock.connect()
    assert result["status"] == "connected"
    
    # Test sending data when connected
    send_result = conn_mock.send_data("Hello World")
    assert send_result["bytes"] == 11
    
    # Test disconnection
    disc_result = conn_mock.disconnect()
    assert disc_result["status"] == "disconnected"
    
    # Test sending data when disconnected (should fail)
    try:
        conn_mock.send_data("Should fail")
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
    
    print("✓ Stateful mock test passed!")

test_connection_lifecycle()
```

---

## Pattern 3: Hierarchical Mock Structures

**Scenario**: Testing complex object hierarchies with nested dependencies.

```python
from unittest.mock import MagicMock, patch
from typing import List, Dict

class MockServiceHierarchy:
    """Creates a complete mock service hierarchy for testing complex systems"""
    
    def __init__(self):
        self.setup_mocks()
    
    def setup_mocks(self):
        # Top-level application mock
        self.app = MagicMock()
        
        # Service layer mocks
        self.user_service = MagicMock()
        self.order_service = MagicMock()
        self.payment_service = MagicMock()
        
        # Data layer mocks
        self.user_repository = MagicMock()
        self.order_repository = MagicMock()
        
        # External service mocks
        self.payment_gateway = MagicMock()
        self.email_service = MagicMock()
        self.notification_service = MagicMock()
        
        # Configure relationships
        self.app.services.user = self.user_service
        self.app.services.order = self.order_service
        self.app.services.payment = self.payment_service
        
        self.user_service.repository = self.user_repository
        self.order_service.repository = self.order_repository
        
        self.payment_service.gateway = self.payment_gateway
        self.user_service.email = self.email_service
        self.user_service.notifications = self.notification_service
        
        # Configure default behaviors
        self._configure_user_behaviors()
        self._configure_order_behaviors()
        self._configure_payment_behaviors()
    
    def _configure_user_behaviors(self):
        self.user_repository.find_by_id.return_value = {
            'id': 1, 'name': 'Test User', 'email': 'test@example.com'
        }
        
        self.user_repository.find_by_email.return_value = {
            'id': 1, 'name': 'Test User', 'email': 'test@example.com'
        }
        
        self.email_service.send.return_value = {'status': 'sent', 'id': 'email123'}
        self.notification_service.push.return_value = {'status': 'delivered'}
    
    def _configure_order_behaviors(self):
        self.order_repository.create.return_value = {
            'id': 101, 'user_id': 1, 'total': 99.99, 'status': 'pending'
        }
        
        self.order_repository.find_by_user.return_value = [
            {'id': 101, 'total': 99.99, 'status': 'completed'},
            {'id': 102, 'total': 149.99, 'status': 'pending'}
        ]
    
    def _configure_payment_behaviors(self):
        self.payment_gateway.charge.return_value = {
            'transaction_id': 'txn123',
            'status': 'success',
            'amount': 99.99
        }
    
    def get_mock_for_service(self, service_name: str):
        """Get a specific service mock by name"""
        return getattr(self, service_name, None)
    
    def verify_service_interactions(self, service_name: str, interactions: Dict):
        """Verify that a service had specific interactions"""
        service = self.get_mock_for_service(service_name)
        for method_name, expected_calls in interactions.items():
            method = getattr(service, method_name)
            assert method.call_count == expected_calls
    
    def reset_all_mocks(self):
        """Reset all mocks to clean state"""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, MagicMock):
                attr.reset_mock()

# Usage Example
def test_complex_order_processing():
    mocks = MockServiceHierarchy()
    
    # Simulate a complex order processing workflow
    def process_order(app, user_email: str, items: List[Dict], card_token: str):
        # Find user
        user = app.services.user.repository.find_by_email(user_email)
        
        # Create order
        order = app.services.order.repository.create({
            'user_id': user['id'],
            'items': items,
            'total': sum(item['price'] for item in items)
        })
        
        # Process payment
        payment = app.services.payment.gateway.charge(
            amount=order['total'],
            token=card_token
        )
        
        # Send confirmation
        app.services.user.email.send(
            to=user['email'],
            subject="Order Confirmation",
            body=f"Order #{order['id']} confirmed"
        )
        
        # Send push notification
        app.services.user.notifications.push(
            user_id=user['id'],
            message="Your order has been confirmed!"
        )
        
        return {
            'order': order,
            'payment': payment,
            'user': user
        }
    
    # Test the workflow
    result = process_order(
        mocks.app,
        "test@example.com",
        [{'name': 'Item 1', 'price': 49.99}, {'name': 'Item 2', 'price': 49.99}],
        "card_token_123"
    )
    
    # Verify all interactions occurred
    assert mocks.user_repository.find_by_email.called
    assert mocks.order_repository.create.called
    assert mocks.payment_gateway.charge.called
    assert mocks.email_service.send.called
    assert mocks.notification_service.push.called
    
    # Verify specific call arguments
    mocks.payment_gateway.charge.assert_called_with(
        amount=99.98,
        token="card_token_123"
    )
    
    print("✓ Complex hierarchical mock test passed!")

test_complex_order_processing()
```

---

## Pattern 4: Mock Decorators and Context Managers

**Scenario**: Creating reusable mock configurations through decorators and context managers.

```python
from unittest.mock import MagicMock, patch
from functools import wraps
from contextlib import contextmanager

class MockDecorators:
    """Collection of reusable mock decorators for common testing scenarios"""
    
    @staticmethod
    def with_database_mock(mock_data=None):
        """Decorator that provides a pre-configured database mock"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with patch('database.connection') as mock_db:
                    mock_db.execute.return_value = mock_data or []
                    mock_db.fetchall.return_value = mock_data or []
                    mock_db.commit.return_value = True
                    mock_db.rollback.return_value = True
                    
                    return func(mock_db, *args, **kwargs)
            return wrapper
        return decorator
    
    @staticmethod
    def with_external_apis():
        """Decorator that mocks multiple external API services"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with patch('requests.get') as mock_get, \
                     patch('requests.post') as mock_post, \
                     patch('redis.Redis') as mock_redis:
                    
                    # Configure HTTP mocks
                    mock_get.return_value.status_code = 200
                    mock_get.return_value.json.return_value = {'status': 'success'}
                    
                    mock_post.return_value.status_code = 201
                    mock_post.return_value.json.return_value = {'id': 123}
                    
                    # Configure Redis mock
                    mock_redis_instance = MagicMock()
                    mock_redis.return_value = mock_redis_instance
                    mock_redis_instance.get.return_value = b'cached_value'
                    mock_redis_instance.set.return_value = True
                    
                    mocks = {
                        'http_get': mock_get,
                        'http_post': mock_post,
                        'redis': mock_redis_instance
                    }
                    
                    return func(mocks, *args, **kwargs)
            return wrapper
        return decorator

@contextmanager
def mock_service_environment(config=None):
    """Context manager for complete service environment mocking"""
    config = config or {}
    
    with patch('os.environ') as mock_env, \
         patch('logging.getLogger') as mock_logger, \
         patch('time.time') as mock_time:
        
        # Configure environment
        mock_env.get.side_effect = lambda key, default=None: config.get(key, default)
        
        # Configure logger
        mock_logger_instance = MagicMock()
        mock_logger.return_value = mock_logger_instance
        
        # Configure time
        mock_time.return_value = 1640995200  # Fixed timestamp
        
        yield {
            'env': mock_env,
            'logger': mock_logger_instance,
            'time': mock_time
        }

# Usage Examples

@MockDecorators.with_database_mock([{'id': 1, 'name': 'Test User'}])
def test_user_repository(mock_db):
    # Simulate repository method
    def get_users():
        mock_db.execute("SELECT * FROM users")
        return mock_db.fetchall()
    
    users = get_users()
    assert len(users) == 1
    assert users[0]['name'] == 'Test User'
    mock_db.execute.assert_called_with("SELECT * FROM users")

@MockDecorators.with_external_apis()
def test_api_integration(mocks):
    # Simulate API calls
    import requests
    
    # Test GET request
    response = requests.get("https://api.example.com/users")
    assert response.status_code == 200
    assert response.json()['status'] == 'success'
    
    # Test Redis caching
    cache_value = mocks['redis'].get('user:1')
    assert cache_value == b'cached_value'
    
    mocks['redis'].set('user:1', 'new_value')
    mocks['redis'].set.assert_called_with('user:1', 'new_value')

def test_service_environment():
    config = {
        'DATABASE_URL': 'sqlite:///:memory:',
        'DEBUG': 'true'
    }
    
    with mock_service_environment(config) as mocks:
        import os
        
        # Test environment access
        db_url = os.environ.get('DATABASE_URL')
        assert db_url == 'sqlite:///:memory:'
        
        # Test logger usage
        logger = mocks['logger']
        logger.info("Test message")
        logger.info.assert_called_with("Test message")
        
        # Test time mocking
        import time
        current_time = time.time()
        assert current_time == 1640995200

# Run the tests
test_user_repository()
test_api_integration()
test_service_environment()
print("✓ All advanced pattern tests passed!")
```

---

## Pattern 5: Mock Performance Profiling

**Scenario**: Creating mocks that simulate performance characteristics and help with load testing.

```python
from unittest.mock import MagicMock
import time
import random
from typing import Dict, List, Callable

class PerformanceMock(MagicMock):
    """Mock that simulates realistic performance characteristics"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._call_stats = {}
        self._performance_config = {
            'base_latency': 0.1,  # Base latency in seconds
            'failure_rate': 0.05,  # 5% failure rate
            'timeout_threshold': 1.0,  # Max acceptable response time
            'rate_limit': 100,  # Max calls per minute
            'memory_leak_factor': 1.001  # Slight memory increase per call
        }
        self._call_count_per_minute = 0
        self._last_minute_reset = time.time()
    
    def configure_performance(self, **kwargs):
        """Configure performance characteristics"""
        self._performance_config.update(kwargs)
    
    def simulate_call(self, method_name: str, *args, **kwargs):
        """Simulate a method call with realistic performance characteristics"""
        current_time = time.time()
        
        # Reset rate limiting counter every minute
        if current_time - self._last_minute_reset > 60:
            self._call_count_per_minute = 0
            self._last_minute_reset = current_time
        
        # Check rate limiting
        if self._call_count_per_minute >= self._performance_config['rate_limit']:
            raise RuntimeError("Rate limit exceeded")
        
        self._call_count_per_minute += 1
        
        # Simulate latency
        latency = self._performance_config['base_latency']
        
        # Add some randomness to latency
        latency *= random.uniform(0.5, 2.0)
        
        # Simulate gradual performance degradation
        call_count = self._call_stats.get(method_name, {}).get('count', 0)
        degradation_factor = self._performance_config['memory_leak_factor'] ** call_count
        latency *= degradation_factor
        
        # Check for timeout
        if latency > self._performance_config['timeout_threshold']:
            raise TimeoutError(f"Method {method_name} timed out")
        
        # Simulate actual delay
        time.sleep(latency)
        
        # Track statistics
        if method_name not in self._call_stats:
            self._call_stats[method_name] = {
                'count': 0,
                'total_time': 0,
                'failures': 0
            }
        
        self._call_stats[method_name]['count'] += 1
        self._call_stats[method_name]['total_time'] += latency
        
        # Simulate random failures
        if random.random() < self._performance_config['failure_rate']:
            self._call_stats[method_name]['failures'] += 1
            raise RuntimeError(f"Random failure in {method_name}")
        
        return True
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics for all methods"""
        stats = {}
        for method_name, method_stats in self._call_stats.items():
            avg_response_time = (
                method_stats['total_time'] / method_stats['count'] 
                if method_stats['count'] > 0 else 0
            )
            failure_rate = (
                method_stats['failures'] / method_stats['count'] 
                if method_stats['count'] > 0 else 0
            )
            
            stats[method_name] = {
                'calls': method_stats['count'],
                'failures': method_stats['failures'],
                'avg_response_time': avg_response_time,
                'failure_rate': failure_rate,
                'total_time': method_stats['total_time']
            }
        
        return stats

class LoadTestingMock:
    """Mock for simulating load testing scenarios"""
    
    def __init__(self):
        self.database_mock = PerformanceMock()
        self.api_mock = PerformanceMock()
        self.cache_mock = PerformanceMock()
        
        # Configure different performance characteristics
        self.database_mock.configure_performance(
            base_latency=0.05,
            failure_rate=0.02,
            rate_limit=1000
        )
        
        self.api_mock.configure_performance(
            base_latency=0.2,
            failure_rate=0.1,
            rate_limit=50
        )
        
        self.cache_mock.configure_performance(
            base_latency=0.001,
            failure_rate=0.001,
            rate_limit=10000
        )
        
        # Attach simulation to method calls
        self.database_mock.query.side_effect = \
            lambda *args, **kwargs: self.database_mock.simulate_call('query', *args, **kwargs)
        
        self.api_mock.fetch.side_effect = \
            lambda *args, **kwargs: self.api_mock.simulate_call('fetch', *args, **kwargs)
        
        self.cache_mock.get.side_effect = \
            lambda *args, **kwargs: self.cache_mock.simulate_call('get', *args, **kwargs)
    
    def run_load_test(self, operations: List[Callable], concurrent_users: int = 10):
        """Run a basic load test simulation"""
        import concurrent.futures
        import threading
        
        results = []
        start_time = time.time()
        
        def run_operations():
            thread_results = []
            for operation in operations:
                try:
                    operation_start = time.time()
                    operation()
                    operation_time = time.time() - operation_start
                    thread_results.append(('success', operation_time))
                except Exception as e:
                    thread_results.append(('failure', str(e)))
            return thread_results
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(run_operations) for _ in range(concurrent_users)]
            
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())
        
        total_time = time.time() - start_time
        
        # Calculate summary statistics
        successful_ops = [r for r in results if r[0] == 'success']
        failed_ops = [r for r in results if r[0] == 'failure']
        
        if successful_ops:
            avg_response_time = sum(r[1] for r in successful_ops) / len(successful_ops)
        else:
            avg_response_time = 0
        
        return {
            'total_operations': len(results),
            'successful_operations': len(successful_ops),
            'failed_operations': len(failed_ops),
            'success_rate': len(successful_ops) / len(results) if results else 0,
            'average_response_time': avg_response_time,
            'total_test_time': total_time,
            'throughput': len(results) / total_time if total_time > 0 else 0
        }

# Usage Example
def test_performance_simulation():
    load_test = LoadTestingMock()
    
    # Define operations to test
    def db_operation():
        load_test.database_mock.query("SELECT * FROM users")
    
    def api_operation():
        load_test.api_mock.fetch("https://api.example.com/data")
    
    def cache_operation():
        load_test.cache_mock.get("user:123")
    
    # Run load test
    operations = [db_operation, api_operation, cache_operation]
    results = load_test.run_load_test(operations, concurrent_users=5)
    
    print("Load Test Results:")
    print(f"  Total Operations: {results['total_operations']}")
    print(f"  Success Rate: {results['success_rate']:.2%}")
    print(f"  Average Response Time: {results['average_response_time']:.3f}s")
    print(f"  Throughput: {results['throughput']:.2f} ops/sec")
    
    # Get detailed performance stats
    print("\nDetailed Performance Stats:")
    for service_name, service_mock in [
        ('Database', load_test.database_mock),
        ('API', load_test.api_mock),
        ('Cache', load_test.cache_mock)
    ]:
        stats = service_mock.get_performance_stats()
        print(f"\n{service_name} Performance:")
        for method, method_stats in stats.items():
            print(f"  {method}: {method_stats['calls']} calls, "
                  f"{method_stats['avg_response_time']:.3f}s avg, "
                  f"{method_stats['failure_rate']:.2%} failure rate")

# Note: This test will take some time to run due to simulated delays
print("Running performance simulation (this may take a moment)...")
test_performance_simulation()
print("✓ Performance simulation test completed!")
```

---

## Summary

These advanced patterns provide:

1. **Dynamic Mock Configuration** - Mocks that adapt behavior based on input
2. **State-Aware Mocks** - Mocks that maintain realistic state across calls
3. **Hierarchical Mock Structures** - Complex mock architectures for large systems
4. **Reusable Mock Decorators** - Common mock configurations as decorators
5. **Performance Simulation** - Mocks that simulate realistic performance characteristics

These patterns are essential for:
- Testing complex enterprise applications
- Simulating realistic service behaviors
- Performance testing and optimization
- Integration testing with multiple dependencies
- Creating maintainable test suites

## Next Steps

- Practice implementing these patterns in your real projects
- Combine patterns to create even more sophisticated test scenarios
- Explore integration with testing frameworks like pytest and unittest
- Consider creating a library of reusable mock patterns for your team