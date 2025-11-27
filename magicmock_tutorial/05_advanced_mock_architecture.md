# Advanced Mock Architecture & Design Patterns 🏗️

## Overview
This guide explores sophisticated architectural patterns for organizing and structuring mocks in large-scale testing environments. Learn how to create maintainable, scalable mock architectures that support complex testing scenarios.

---

## Architecture Pattern 1: Mock Factory Pattern

**Scenario**: Creating standardized mock objects with consistent configurations across your test suite.

```python
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Optional, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class MockType(Enum):
    DATABASE = "database"
    API_CLIENT = "api_client"
    MESSAGE_QUEUE = "message_queue"
    FILE_SYSTEM = "file_system"
    CACHE = "cache"

@dataclass
class MockConfiguration:
    """Configuration for mock behavior"""
    mock_type: MockType
    return_values: Dict[str, Any]
    side_effects: Dict[str, Any]
    failure_scenarios: Dict[str, Exception]
    call_tracking: bool = True
    spec_class: Optional[type] = None

class MockFactory(ABC):
    """Abstract factory for creating mocks"""
    
    @abstractmethod
    def create_mock(self, config: MockConfiguration) -> MagicMock:
        pass

class StandardMockFactory(MockFactory):
    """Standard implementation of mock factory"""
    
    def __init__(self):
        self._mock_registry = {}
        self._creation_strategies = {
            MockType.DATABASE: self._create_database_mock,
            MockType.API_CLIENT: self._create_api_client_mock,
            MockType.MESSAGE_QUEUE: self._create_message_queue_mock,
            MockType.FILE_SYSTEM: self._create_file_system_mock,
            MockType.CACHE: self._create_cache_mock,
        }
    
    def create_mock(self, config: MockConfiguration) -> MagicMock:
        """Create a mock based on configuration"""
        strategy = self._creation_strategies.get(config.mock_type)
        if not strategy:
            raise ValueError(f"Unknown mock type: {config.mock_type}")
        
        mock = strategy(config)
        
        if config.call_tracking:
            self._register_mock(config.mock_type.value, mock)
        
        return mock
    
    def _create_database_mock(self, config: MockConfiguration) -> MagicMock:
        """Create database connection mock"""
        mock = MagicMock(spec=config.spec_class)
        
        # Standard database methods
        mock.execute.return_value = config.return_values.get('execute', True)
        mock.fetchall.return_value = config.return_values.get('fetchall', [])
        mock.fetchone.return_value = config.return_values.get('fetchone', None)
        mock.commit.return_value = config.return_values.get('commit', None)
        mock.rollback.return_value = config.return_values.get('rollback', None)
        
        # Configure side effects
        for method, effect in config.side_effects.items():
            setattr(getattr(mock, method), 'side_effect', effect)
        
        # Configure failure scenarios
        for method, exception in config.failure_scenarios.items():
            setattr(getattr(mock, method), 'side_effect', exception)
        
        return mock
    
    def _create_api_client_mock(self, config: MockConfiguration) -> MagicMock:
        """Create API client mock"""
        mock = MagicMock(spec=config.spec_class)
        
        # HTTP methods
        mock.get.return_value.status_code = config.return_values.get('status_code', 200)
        mock.get.return_value.json.return_value = config.return_values.get('json_data', {})
        mock.post.return_value.status_code = config.return_values.get('post_status', 201)
        mock.put.return_value.status_code = config.return_values.get('put_status', 200)
        mock.delete.return_value.status_code = config.return_values.get('delete_status', 204)
        
        # Configure authentication
        mock.authenticate.return_value = config.return_values.get('auth_token', 'mock_token')
        
        return mock
    
    def _create_message_queue_mock(self, config: MockConfiguration) -> MagicMock:
        """Create message queue mock"""
        mock = MagicMock(spec=config.spec_class)
        
        # Queue operations
        mock.publish.return_value = config.return_values.get('publish_result', True)
        mock.consume.return_value = config.return_values.get('messages', [])
        mock.acknowledge.return_value = config.return_values.get('ack_result', True)
        
        return mock
    
    def _create_file_system_mock(self, config: MockConfiguration) -> MagicMock:
        """Create file system mock"""
        mock = MagicMock(spec=config.spec_class)
        
        # File operations
        mock.read.return_value = config.return_values.get('file_content', 'mock content')
        mock.write.return_value = config.return_values.get('write_result', True)
        mock.exists.return_value = config.return_values.get('file_exists', True)
        mock.delete.return_value = config.return_values.get('delete_result', True)
        
        return mock
    
    def _create_cache_mock(self, config: MockConfiguration) -> MagicMock:
        """Create cache mock"""
        mock = MagicMock(spec=config.spec_class)
        
        # Cache operations
        mock.get.return_value = config.return_values.get('cached_value', None)
        mock.set.return_value = config.return_values.get('set_result', True)
        mock.delete.return_value = config.return_values.get('delete_result', True)
        mock.clear.return_value = config.return_values.get('clear_result', True)
        
        return mock
    
    def _register_mock(self, name: str, mock: MagicMock):
        """Register mock for tracking"""
        self._mock_registry[name] = mock
    
    def get_registered_mock(self, name: str) -> Optional[MagicMock]:
        """Get a registered mock by name"""
        return self._mock_registry.get(name)
    
    def reset_all_mocks(self):
        """Reset all registered mocks"""
        for mock in self._mock_registry.values():
            mock.reset_mock()
    
    def get_call_summary(self) -> Dict[str, Dict]:
        """Get summary of all mock calls"""
        summary = {}
        for name, mock in self._mock_registry.items():
            summary[name] = {
                'call_count': mock.call_count,
                'called': mock.called,
                'method_calls': len(mock.method_calls)
            }
        return summary

# Usage Example
def test_mock_factory():
    factory = StandardMockFactory()
    
    # Create database mock
    db_config = MockConfiguration(
        mock_type=MockType.DATABASE,
        return_values={
            'fetchall': [{'id': 1, 'name': 'Test User'}],
            'execute': True
        },
        side_effects={},
        failure_scenarios={}
    )
    
    db_mock = factory.create_mock(db_config)
    
    # Use the mock
    db_mock.execute("SELECT * FROM users")
    users = db_mock.fetchall()
    
    assert len(users) == 1
    assert users[0]['name'] == 'Test User'
    
    # Create API client mock
    api_config = MockConfiguration(
        mock_type=MockType.API_CLIENT,
        return_values={
            'status_code': 200,
            'json_data': {'users': [{'id': 1, 'name': 'API User'}]}
        },
        side_effects={},
        failure_scenarios={}
    )
    
    api_mock = factory.create_mock(api_config)
    
    response = api_mock.get('/users')
    assert response.status_code == 200
    assert 'users' in response.json()
    
    # Get call summary
    summary = factory.get_call_summary()
    print(f"Mock call summary: {summary}")

test_mock_factory()
print("✓ Mock Factory pattern test passed!")
```

---

## Architecture Pattern 2: Mock Builder Pattern

**Scenario**: Building complex mock configurations step by step with a fluent interface.

```python
from unittest.mock import MagicMock
from typing import Dict, Any, List, Optional, Callable

class MockBuilder:
    """Builder for creating complex mock configurations"""
    
    def __init__(self, base_mock: Optional[MagicMock] = None):
        self._mock = base_mock or MagicMock()
        self._configurations = []
        self._validations = []
    
    def with_method(self, method_name: str) -> 'MethodBuilder':
        """Configure a specific method"""
        return MethodBuilder(self, method_name)
    
    def with_attribute(self, attr_name: str, value: Any) -> 'MockBuilder':
        """Set an attribute value"""
        setattr(self._mock, attr_name, value)
        self._configurations.append(f"Set attribute {attr_name} = {value}")
        return self
    
    def with_property(self, prop_name: str, getter=None, setter=None) -> 'MockBuilder':
        """Configure a property with getter/setter"""
        from unittest.mock import PropertyMock
        
        prop_mock = PropertyMock()
        if getter is not None:
            prop_mock.return_value = getter() if callable(getter) else getter
        
        setattr(type(self._mock), prop_name, prop_mock)
        self._configurations.append(f"Configured property {prop_name}")
        return self
    
    def with_context_manager(self, enter_value=None, exit_value=None) -> 'MockBuilder':
        """Configure as context manager"""
        self._mock.__enter__ = MagicMock(return_value=enter_value or self._mock)
        self._mock.__exit__ = MagicMock(return_value=exit_value)
        self._configurations.append("Configured as context manager")
        return self
    
    def with_iteration(self, items: List[Any]) -> 'MockBuilder':
        """Configure iteration behavior"""
        self._mock.__iter__ = MagicMock(return_value=iter(items))
        self._configurations.append(f"Configured iteration with {len(items)} items")
        return self
    
    def with_validation(self, validator: Callable[['MockBuilder'], bool], message: str) -> 'MockBuilder':
        """Add a validation check"""
        self._validations.append((validator, message))
        return self
    
    def build(self) -> MagicMock:
        """Build the final mock with validations"""
        # Run validations
        for validator, message in self._validations:
            if not validator(self):
                raise ValueError(f"Validation failed: {message}")
        
        return self._mock
    
    def get_configuration_summary(self) -> List[str]:
        """Get summary of applied configurations"""
        return self._configurations.copy()

class MethodBuilder:
    """Builder for configuring individual methods"""
    
    def __init__(self, parent_builder: MockBuilder, method_name: str):
        self._parent = parent_builder
        self._method_name = method_name
        self._method = getattr(parent_builder._mock, method_name)
    
    def returns(self, value: Any) -> 'MethodBuilder':
        """Set return value"""
        self._method.return_value = value
        self._parent._configurations.append(f"Method {self._method_name} returns {value}")
        return self
    
    def raises(self, exception: Exception) -> 'MethodBuilder':
        """Set exception to raise"""
        self._method.side_effect = exception
        self._parent._configurations.append(f"Method {self._method_name} raises {type(exception).__name__}")
        return self
    
    def calls(self, func: Callable) -> 'MethodBuilder':
        """Set custom function to call"""
        self._method.side_effect = func
        self._parent._configurations.append(f"Method {self._method_name} calls custom function")
        return self
    
    def returns_sequence(self, values: List[Any]) -> 'MethodBuilder':
        """Return different values on consecutive calls"""
        self._method.side_effect = values
        self._parent._configurations.append(f"Method {self._method_name} returns sequence of {len(values)} values")
        return self
    
    def with_args_validation(self, validator: Callable) -> 'MethodBuilder':
        """Add argument validation"""
        original_side_effect = self._method.side_effect
        
        def validated_call(*args, **kwargs):
            if not validator(*args, **kwargs):
                raise ValueError(f"Invalid arguments for {self._method_name}")
            
            if original_side_effect:
                if isinstance(original_side_effect, Exception):
                    raise original_side_effect
                elif callable(original_side_effect):
                    return original_side_effect(*args, **kwargs)
                elif isinstance(original_side_effect, list):
                    if hasattr(validated_call, '_call_index'):
                        validated_call._call_index += 1
                    else:
                        validated_call._call_index = 0
                    
                    if validated_call._call_index < len(original_side_effect):
                        return original_side_effect[validated_call._call_index]
            
            return self._method.return_value
        
        self._method.side_effect = validated_call
        self._parent._configurations.append(f"Method {self._method_name} has argument validation")
        return self
    
    def and_then(self) -> MockBuilder:
        """Return to parent builder for method chaining"""
        return self._parent

# Usage Examples

def test_mock_builder_basic():
    """Test basic mock builder functionality"""
    
    # Build a user service mock
    user_service = (MockBuilder()
                   .with_method('get_user').returns({'id': 1, 'name': 'Test User'}).and_then()
                   .with_method('create_user').returns({'id': 2, 'name': 'New User'}).and_then()
                   .with_method('delete_user').returns(True).and_then()
                   .with_attribute('service_name', 'UserService')
                   .build())
    
    # Test the mock
    user = user_service.get_user(1)
    assert user['name'] == 'Test User'
    
    new_user = user_service.create_user({'name': 'New User'})
    assert new_user['id'] == 2
    
    result = user_service.delete_user(1)
    assert result is True
    
    assert user_service.service_name == 'UserService'

def test_mock_builder_advanced():
    """Test advanced mock builder features"""
    
    # Build a complex database connection mock
    db_connection = (MockBuilder()
                    .with_method('execute')
                    .with_args_validation(lambda sql, *args: sql and isinstance(sql, str))
                    .returns(True)
                    .and_then()
                    .with_method('fetchall')
                    .returns_sequence([
                        [{'id': 1, 'name': 'User 1'}],
                        [{'id': 2, 'name': 'User 2'}],
                        []
                    ])
                    .and_then()
                    .with_method('close')
                    .calls(lambda: print("Connection closed"))
                    .and_then()
                    .with_context_manager(enter_value="Connected")
                    .with_validation(
                        lambda builder: hasattr(builder._mock, 'execute'),
                        "Database mock must have execute method"
                    )
                    .build())
    
    # Test the mock
    with db_connection as conn:
        db_connection.execute("SELECT * FROM users")
        users1 = db_connection.fetchall()
        users2 = db_connection.fetchall()
        users3 = db_connection.fetchall()
        
        assert len(users1) == 1
        assert len(users2) == 1
        assert len(users3) == 0
        
        # This will print "Connection closed"
        db_connection.close()

def test_mock_builder_with_iteration():
    """Test mock with iteration capabilities"""
    
    # Build a mock that can be iterated
    items = ['item1', 'item2', 'item3']
    iterator_mock = (MockBuilder()
                    .with_iteration(items)
                    .with_method('count').returns(len(items)).and_then()
                    .build())
    
    # Test iteration
    collected_items = list(iterator_mock)
    assert collected_items == items
    assert iterator_mock.count() == 3

# Advanced Mock Builder for Specific Domains

class DatabaseMockBuilder(MockBuilder):
    """Specialized builder for database mocks"""
    
    def with_table_data(self, table_name: str, data: List[Dict]) -> 'DatabaseMockBuilder':
        """Configure data for a specific table"""
        def execute_side_effect(sql, *args):
            if f"FROM {table_name}" in sql.upper():
                # Simulate returning data for this table
                return True
            return True
        
        def fetchall_side_effect():
            return data
        
        self.with_method('execute').calls(execute_side_effect).and_then()
        self.with_method('fetchall').calls(fetchall_side_effect).and_then()
        self._configurations.append(f"Configured table {table_name} with {len(data)} rows")
        return self
    
    def with_transaction_support(self) -> 'DatabaseMockBuilder':
        """Add transaction support"""
        transaction_state = {'active': False, 'rolled_back': False}
        
        def begin_transaction():
            transaction_state['active'] = True
            transaction_state['rolled_back'] = False
            return True
        
        def commit():
            if not transaction_state['active']:
                raise RuntimeError("No active transaction")
            transaction_state['active'] = False
            return True
        
        def rollback():
            if not transaction_state['active']:
                raise RuntimeError("No active transaction")
            transaction_state['active'] = False
            transaction_state['rolled_back'] = True
            return True
        
        (self.with_method('begin').calls(begin_transaction).and_then()
             .with_method('commit').calls(commit).and_then()
             .with_method('rollback').calls(rollback).and_then())
        
        self._configurations.append("Added transaction support")
        return self

def test_specialized_builders():
    """Test specialized builder for databases"""
    
    users_data = [
        {'id': 1, 'name': 'Alice', 'email': 'alice@test.com'},
        {'id': 2, 'name': 'Bob', 'email': 'bob@test.com'}
    ]
    
    db = (DatabaseMockBuilder()
          .with_table_data('users', users_data)
          .with_transaction_support()
          .with_attribute('connected', True)
          .build())
    
    # Test table data
    db.execute("SELECT * FROM users")
    users = db.fetchall()
    assert len(users) == 2
    assert users[0]['name'] == 'Alice'
    
    # Test transaction support
    db.begin()
    db.execute("INSERT INTO users VALUES (3, 'Charlie')")
    db.commit()
    
    # Verify all methods were called
    assert db.begin.called
    assert db.commit.called

# Run all tests
test_mock_builder_basic()
test_mock_builder_advanced()
test_mock_builder_with_iteration()
test_specialized_builders()

print("✓ All Mock Builder pattern tests passed!")
```

---

## Architecture Pattern 3: Mock Registry and Dependency Injection

**Scenario**: Managing mock dependencies across complex test suites with automatic injection.

```python
from unittest.mock import MagicMock, patch
from typing import Dict, Any, Type, Optional, Callable, List
from dataclasses import dataclass, field
import inspect
from contextlib import contextmanager

@dataclass
class MockDependency:
    """Represents a mock dependency"""
    name: str
    mock_class: Type[MagicMock] = MagicMock
    configuration: Dict[str, Any] = field(default_factory=dict)
    singleton: bool = False
    setup_callback: Optional[Callable[[MagicMock], None]] = None

class MockRegistry:
    """Registry for managing mock dependencies"""
    
    def __init__(self):
        self._dependencies: Dict[str, MockDependency] = {}
        self._instances: Dict[str, MagicMock] = {}
        self._patches: Dict[str, Any] = {}
    
    def register(self, dependency: MockDependency):
        """Register a mock dependency"""
        self._dependencies[dependency.name] = dependency
        
        # Create singleton instance if needed
        if dependency.singleton:
            self._create_instance(dependency.name)
    
    def register_mock(self, name: str, target_path: str, mock_class: Type[MagicMock] = MagicMock, 
                     **config) -> MockDependency:
        """Register a mock with target path for patching"""
        dependency = MockDependency(
            name=name,
            mock_class=mock_class,
            configuration={'target_path': target_path, **config}
        )
        self.register(dependency)
        return dependency
    
    def get_mock(self, name: str) -> MagicMock:
        """Get a mock instance"""
        if name not in self._dependencies:
            raise ValueError(f"Mock dependency '{name}' not registered")
        
        dependency = self._dependencies[name]
        
        # Return singleton instance if it exists
        if dependency.singleton and name in self._instances:
            return self._instances[name]
        
        # Create new instance
        return self._create_instance(name)
    
    def _create_instance(self, name: str) -> MagicMock:
        """Create a mock instance"""
        dependency = self._dependencies[name]
        
        # Create mock instance
        mock = dependency.mock_class(**dependency.configuration)
        
        # Apply setup callback
        if dependency.setup_callback:
            dependency.setup_callback(mock)
        
        # Store singleton instance
        if dependency.singleton:
            self._instances[name] = mock
        
        return mock
    
    @contextmanager
    def patch_context(self, *mock_names):
        """Context manager that patches all specified mocks"""
        patches = []
        mock_instances = {}
        
        try:
            for name in mock_names:
                if name not in self._dependencies:
                    raise ValueError(f"Mock dependency '{name}' not registered")
                
                dependency = self._dependencies[name]
                target_path = dependency.configuration.get('target_path')
                
                if target_path:
                    mock_instance = self.get_mock(name)
                    patcher = patch(target_path, mock_instance)
                    patches.append(patcher)
                    patcher.start()
                    mock_instances[name] = mock_instance
                else:
                    # Just create the mock without patching
                    mock_instances[name] = self.get_mock(name)
            
            yield mock_instances
            
        finally:
            # Stop all patches
            for patcher in patches:
                patcher.stop()
    
    def reset_all(self):
        """Reset all mock instances"""
        for mock in self._instances.values():
            mock.reset_mock()
    
    def get_call_summary(self) -> Dict[str, Dict]:
        """Get call summary for all mocks"""
        summary = {}
        for name, mock in self._instances.items():
            summary[name] = {
                'called': mock.called,
                'call_count': mock.call_count,
                'method_calls_count': len(mock.method_calls)
            }
        return summary

class MockInjector:
    """Dependency injector for automatic mock injection"""
    
    def __init__(self, registry: MockRegistry):
        self.registry = registry
    
    def inject_into_function(self, func: Callable, **manual_overrides) -> Callable:
        """Inject mocks into function parameters"""
        sig = inspect.signature(func)
        
        def wrapper(*args, **kwargs):
            # Get parameter names that need injection
            param_names = list(sig.parameters.keys())
            
            # Skip parameters that already have values
            provided_params = set(kwargs.keys())
            
            # Inject mocks for missing parameters
            for i, param_name in enumerate(param_names):
                # Skip if already provided in args
                if i < len(args):
                    continue
                
                # Skip if already provided in kwargs
                if param_name in provided_params:
                    continue
                
                # Skip if manual override provided
                if param_name in manual_overrides:
                    kwargs[param_name] = manual_overrides[param_name]
                    continue
                
                # Try to inject mock
                try:
                    mock = self.registry.get_mock(param_name)
                    kwargs[param_name] = mock
                except ValueError:
                    # Mock not registered for this parameter
                    pass
            
            return func(*args, **kwargs)
        
        return wrapper
    
    def auto_inject(self, **manual_overrides):
        """Decorator for automatic mock injection"""
        def decorator(func):
            return self.inject_into_function(func, **manual_overrides)
        return decorator

# Usage Examples

def test_mock_registry():
    """Test mock registry functionality"""
    
    registry = MockRegistry()
    
    # Register database mock
    def setup_database_mock(mock):
        mock.execute.return_value = True
        mock.fetchall.return_value = [{'id': 1, 'name': 'Test User'}]
        mock.commit.return_value = None
    
    db_dependency = MockDependency(
        name='database',
        configuration={'target_path': 'myapp.database.connection'},
        singleton=True,
        setup_callback=setup_database_mock
    )
    registry.register(db_dependency)
    
    # Register API client mock
    registry.register_mock(
        'api_client',
        target_path='myapp.external.api_client',
        spec=['get', 'post', 'authenticate']
    )
    
    # Test getting mocks
    db_mock = registry.get_mock('database')
    assert db_mock.execute() is True
    
    users = db_mock.fetchall()
    assert len(users) == 1
    assert users[0]['name'] == 'Test User'
    
    # Test singleton behavior
    db_mock2 = registry.get_mock('database')
    assert db_mock is db_mock2  # Same instance
    
    print("✓ Mock registry test passed!")

def test_mock_injection():
    """Test automatic mock injection"""
    
    registry = MockRegistry()
    injector = MockInjector(registry)
    
    # Setup mocks
    def setup_user_service(mock):
        mock.get_user.return_value = {'id': 1, 'name': 'Test User'}
        mock.create_user.return_value = {'id': 2, 'name': 'New User'}
    
    def setup_email_service(mock):
        mock.send_email.return_value = {'status': 'sent', 'id': 'email123'}
    
    registry.register(MockDependency(
        name='user_service',
        setup_callback=setup_user_service
    ))
    
    registry.register(MockDependency(
        name='email_service',
        setup_callback=setup_email_service
    ))
    
    # Function that needs dependencies
    @injector.auto_inject()
    def process_user_registration(user_data, user_service, email_service):
        # Create user
        user = user_service.create_user(user_data)
        
        # Send welcome email
        email_result = email_service.send_email(
            to=user_data['email'],
            subject='Welcome!',
            body='Welcome to our service!'
        )
        
        return {'user': user, 'email': email_result}
    
    # Test the function (mocks are automatically injected)
    result = process_user_registration({
        'name': 'John Doe',
        'email': 'john@example.com'
    })
    
    assert result['user']['name'] == 'New User'
    assert result['email']['status'] == 'sent'
    
    # Verify mocks were called
    user_service = registry.get_mock('user_service')
    email_service = registry.get_mock('email_service')
    
    user_service.create_user.assert_called_once()
    email_service.send_email.assert_called_once()
    
    print("✓ Mock injection test passed!")

def test_mock_context_patching():
    """Test context manager for mock patching"""
    
    registry = MockRegistry()
    
    # Register mocks with patch targets
    registry.register_mock('requests', 'requests.get')
    registry.register_mock('redis_client', 'redis.Redis')
    
    # Configure the mocks
    def setup_requests_mock(mock):
        mock.return_value.status_code = 200
        mock.return_value.json.return_value = {'message': 'success'}
    
    def setup_redis_mock(mock):
        mock.return_value.get.return_value = b'cached_data'
        mock.return_value.set.return_value = True
    
    registry._dependencies['requests'].setup_callback = setup_requests_mock
    registry._dependencies['redis_client'].setup_callback = setup_redis_mock
    
    # Function that uses external dependencies
    def fetch_user_data(user_id):
        import requests
        import redis
        
        # Try to get from cache first
        redis_client = redis.Redis()
        cached = redis_client.get(f'user:{user_id}')
        
        if cached:
            return {'source': 'cache', 'data': cached.decode()}
        
        # Fetch from API
        response = requests.get(f'https://api.example.com/users/{user_id}')
        
        if response.status_code == 200:
            data = response.json()
            # Cache the result
            redis_client.set(f'user:{user_id}', str(data))
            return {'source': 'api', 'data': data}
        
        return None
    
    # Test with mocked dependencies
    with registry.patch_context('requests', 'redis_client') as mocks:
        result = fetch_user_data(123)
        
        assert result['source'] == 'cache'
        assert result['data'] == 'cached_data'
        
        # Verify calls
        redis_mock = mocks['redis_client']
        redis_mock.get.assert_called_with('user:123')
    
    print("✓ Mock context patching test passed!")

# Complex Integration Test

def test_complete_integration():
    """Test complete integration of all patterns"""
    
    registry = MockRegistry()
    injector = MockInjector(registry)
    
    # Setup a complete service architecture
    services = [
        ('database', 'myapp.db.connection', lambda m: setattr(m, 'query', MagicMock(return_value=[]))),
        ('cache', 'myapp.cache.redis', lambda m: setattr(m, 'get', MagicMock(return_value=None))),
        ('message_queue', 'myapp.queue.publisher', lambda m: setattr(m, 'publish', MagicMock(return_value=True))),
        ('email_service', 'myapp.email.client', lambda m: setattr(m, 'send', MagicMock(return_value={'sent': True}))),
    ]
    
    for name, target, setup in services:
        registry.register_mock(name, target)
        registry._dependencies[name].setup_callback = setup
    
    @injector.auto_inject()
    def complex_business_operation(operation_data, database, cache, message_queue, email_service):
        # Check cache
        cached_result = cache.get(f"operation:{operation_data['id']}")
        if cached_result:
            return {'source': 'cache', 'result': cached_result}
        
        # Query database
        db_result = database.query("SELECT * FROM operations WHERE id = ?", operation_data['id'])
        
        # Publish event
        message_queue.publish({
            'event': 'operation_processed',
            'operation_id': operation_data['id']
        })
        
        # Send notification
        email_service.send({
            'to': operation_data['email'],
            'subject': 'Operation Complete',
            'body': 'Your operation has been processed'
        })
        
        return {'source': 'database', 'result': db_result}
    
    # Test the operation
    with registry.patch_context('database', 'cache', 'message_queue', 'email_service'):
        result = complex_business_operation({
            'id': 123,
            'email': 'user@example.com'
        })
        
        assert result['source'] == 'database'
        
        # Verify all services were called
        summary = registry.get_call_summary()
        for service_name in ['database', 'cache', 'message_queue', 'email_service']:
            mock = registry.get_mock(service_name)
            assert mock.called, f"{service_name} should have been called"
    
    print("✓ Complete integration test passed!")

# Run all tests
test_mock_registry()
test_mock_injection()
test_mock_context_patching()
test_complete_integration()

print("✓ All Mock Architecture tests passed!")
```

---

## Architecture Pattern 4: Mock State Management

**Scenario**: Managing complex state across multiple mock interactions in long-running test scenarios.

```python
from unittest.mock import MagicMock
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time

class MockState(Enum):
    INITIAL = "initial"
    ACTIVE = "active"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class StateTransition:
    """Represents a state transition"""
    from_state: MockState
    to_state: MockState
    condition: Callable[[], bool]
    action: Optional[Callable[[], None]] = None

@dataclass
class MockSnapshot:
    """Snapshot of mock state at a point in time"""
    timestamp: float
    state: MockState
    call_count: int
    last_method_called: Optional[str]
    data: Dict[str, Any] = field(default_factory=dict)

class StatefulMockManager:
    """Manages state across multiple mocks with history and rollback capabilities"""
    
    def __init__(self):
        self._mocks: Dict[str, MagicMock] = {}
        self._states: Dict[str, MockState] = {}
        self._transitions: Dict[str, List[StateTransition]] = {}
        self._history: Dict[str, List[MockSnapshot]] = {}
        self._global_state: Dict[str, Any] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
    
    def register_mock(self, name: str, mock: MagicMock, initial_state: MockState = MockState.INITIAL):
        """Register a mock with state management"""
        self._mocks[name] = mock
        self._states[name] = initial_state
        self._transitions[name] = []
        self._history[name] = []
        
        # Wrap methods to track state changes
        self._wrap_mock_methods(name, mock)
        
        # Create initial snapshot
        self._create_snapshot(name)
    
    def _wrap_mock_methods(self, name: str, mock: MagicMock):
        """Wrap mock methods to track calls and trigger state transitions"""
        original_getattr = mock.__getattribute__
        
        def wrapped_getattr(attr_name):
            attr = original_getattr(attr_name)
            
            if callable(attr) and not attr_name.startswith('_'):
                # Wrap the method call
                def wrapped_method(*args, **kwargs):
                    # Check state transitions before call
                    self._check_state_transitions(name)
                    
                    # Execute the original method
                    result = attr(*args, **kwargs)
                    
                    # Update state after call
                    self._update_state_after_call(name, attr_name, args, kwargs)
                    
                    # Trigger events
                    self._trigger_event(f"{name}_{attr_name}_called", {
                        'args': args,
                        'kwargs': kwargs,
                        'result': result
                    })
                    
                    return result
                
                # Replace the original method
                setattr(mock, attr_name, wrapped_method)
                return wrapped_method
            
            return attr
        
        mock.__getattribute__ = wrapped_getattr
    
    def add_state_transition(self, mock_name: str, transition: StateTransition):
        """Add a state transition rule"""
        if mock_name not in self._transitions:
            self._transitions[mock_name] = []
        self._transitions[mock_name].append(transition)
    
    def _check_state_transitions(self, mock_name: str):
        """Check and apply state transitions"""
        current_state = self._states[mock_name]
        
        for transition in self._transitions[mock_name]:
            if (transition.from_state == current_state and 
                transition.condition()):
                
                # Execute transition action
                if transition.action:
                    transition.action()
                
                # Change state
                old_state = current_state
                self._states[mock_name] = transition.to_state
                
                # Trigger state change event
                self._trigger_event(f"{mock_name}_state_changed", {
                    'from_state': old_state,
                    'to_state': transition.to_state
                })
                
                break
    
    def _update_state_after_call(self, mock_name: str, method_name: str, args: tuple, kwargs: dict):
        """Update mock state after a method call"""
        # Create snapshot
        self._create_snapshot(mock_name, method_name)
    
    def _create_snapshot(self, mock_name: str, last_method: Optional[str] = None):
        """Create a state snapshot"""
        mock = self._mocks[mock_name]
        snapshot = MockSnapshot(
            timestamp=time.time(),
            state=self._states[mock_name],
            call_count=mock.call_count,
            last_method_called=last_method,
            data=self._global_state.copy()
        )
        
        self._history[mock_name].append(snapshot)
        
        # Limit history size
        if len(self._history[mock_name]) > 100:
            self._history[mock_name] = self._history[mock_name][-50:]
    
    def get_state(self, mock_name: str) -> MockState:
        """Get current state of a mock"""
        return self._states.get(mock_name, MockState.INITIAL)
    
    def set_state(self, mock_name: str, state: MockState):
        """Manually set mock state"""
        old_state = self._states.get(mock_name)
        self._states[mock_name] = state
        
        self._trigger_event(f"{mock_name}_state_changed", {
            'from_state': old_state,
            'to_state': state
        })
    
    def rollback_to_snapshot(self, mock_name: str, steps_back: int = 1) -> bool:
        """Rollback mock to a previous snapshot"""
        history = self._history.get(mock_name, [])
        
        if len(history) < steps_back + 1:
            return False
        
        target_snapshot = history[-(steps_back + 1)]
        
        # Reset mock
        mock = self._mocks[mock_name]
        mock.reset_mock()
        
        # Restore state
        self._states[mock_name] = target_snapshot.state
        self._global_state.update(target_snapshot.data)
        
        # Remove future history
        self._history[mock_name] = history[:-(steps_back)]
        
        return True
    
    def get_history_summary(self, mock_name: str) -> List[Dict]:
        """Get summary of mock history"""
        history = self._history.get(mock_name, [])
        return [
            {
                'timestamp': snapshot.timestamp,
                'state': snapshot.state.value,
                'call_count': snapshot.call_count,
                'last_method': snapshot.last_method_called
            }
            for snapshot in history
        ]
    
    def set_global_state(self, key: str, value: Any):
        """Set global state variable"""
        self._global_state[key] = value
    
    def get_global_state(self, key: str) -> Any:
        """Get global state variable"""
        return self._global_state.get(key)
    
    def on_event(self, event_name: str, handler: Callable):
        """Register event handler"""
        if event_name not in self._event_handlers:
            self._event_handlers[event_name] = []
        self._event_handlers[event_name].append(handler)
    
    def _trigger_event(self, event_name: str, data: Dict[str, Any]):
        """Trigger event handlers"""
        handlers = self._event_handlers.get(event_name, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                print(f"Event handler error: {e}")
    
    def export_state(self, mock_name: str) -> str:
        """Export mock state as JSON"""
        state_data = {
            'current_state': self._states[mock_name].value,
            'call_count': self._mocks[mock_name].call_count,
            'global_state': self._global_state,
            'history': self.get_history_summary(mock_name)
        }
        return json.dumps(state_data, indent=2)
    
    def import_state(self, mock_name: str, state_json: str):
        """Import mock state from JSON"""
        state_data = json.loads(state_json)
        
        # Restore state
        self._states[mock_name] = MockState(state_data['current_state'])
        self._global_state.update(state_data['global_state'])

# Usage Example

def test_stateful_mock_manager():
    """Test stateful mock manager"""
    
    manager = StatefulMockManager()
    
    # Create connection mock
    connection_mock = MagicMock()
    connection_mock.connect.return_value = True
    connection_mock.disconnect.return_value = True
    connection_mock.send_data.return_value = "sent"
    
    manager.register_mock('connection', connection_mock, MockState.INITIAL)
    
    # Define state transitions
    call_count = {'value': 0}
    
    def connection_condition():
        call_count['value'] += 1
        return call_count['value'] >= 3
    
    def activate_connection():
        manager.set_global_state('connected', True)
        print("Connection activated!")
    
    def error_condition():
        return manager.get_global_state('error_triggered', False)
    
    # Add transitions
    manager.add_state_transition('connection', StateTransition(
        from_state=MockState.INITIAL,
        to_state=MockState.ACTIVE,
        condition=connection_condition,
        action=activate_connection
    ))
    
    manager.add_state_transition('connection', StateTransition(
        from_state=MockState.ACTIVE,
        to_state=MockState.ERROR,
        condition=error_condition
    ))
    
    # Add event handlers
    def on_state_change(data):
        print(f"State changed: {data['from_state'].value} -> {data['to_state'].value}")
    
    manager.on_event('connection_state_changed', on_state_change)
    
    # Test the mock
    assert manager.get_state('connection') == MockState.INITIAL
    
    # Make some calls to trigger state change
    connection_mock.connect()
    connection_mock.connect()
    connection_mock.connect()  # This should trigger state change to ACTIVE
    
    assert manager.get_state('connection') == MockState.ACTIVE
    assert manager.get_global_state('connected') is True
    
    # Trigger error state
    manager.set_global_state('error_triggered', True)
    connection_mock.send_data("test")  # This should trigger error state
    
    assert manager.get_state('connection') == MockState.ERROR
    
    # Test rollback
    success = manager.rollback_to_snapshot('connection', 2)
    assert success
    
    # Export and import state
    state_export = manager.export_state('connection')
    print("Exported state:", state_export)
    
    print("✓ Stateful mock manager test passed!")

test_stateful_mock_manager()
print("✓ All Mock State Management tests passed!")
```

---

## Summary

These advanced architectural patterns provide:

1. **Mock Factory Pattern** - Standardized mock creation with consistent configurations
2. **Mock Builder Pattern** - Fluent interface for building complex mock configurations
3. **Mock Registry & Dependency Injection** - Centralized management and automatic injection of mocks
4. **Mock State Management** - Complex state tracking with history and rollback capabilities

These patterns enable:
- **Maintainable Test Suites** - Consistent mock creation and configuration
- **Complex Scenario Testing** - State management across multiple interactions
- **Enterprise-Grade Testing** - Dependency injection and centralized mock management
- **Advanced Debugging** - History tracking and state rollback for test debugging
- **Team Collaboration** - Shared mock configurations and patterns

## Best Practices

1. **Use factories** for consistent mock creation across teams
2. **Implement builders** for complex mock configurations
3. **Centralize mock management** with registries for large projects
4. **Track state and history** for complex integration scenarios
5. **Create reusable patterns** that can be shared across projects

## Next Steps

- Implement these patterns in your current projects
- Create a mock library for your organization
- Integrate with your existing testing frameworks
- Train your team on these advanced patterns