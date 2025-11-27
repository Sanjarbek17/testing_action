# Enterprise MagicMock Testing Strategies 🏢

## Overview
This guide covers enterprise-level testing strategies using MagicMock for large-scale applications, microservices architectures, and complex business systems. Learn industry best practices used by Fortune 500 companies.

---

## Strategy 1: Microservices Testing Architecture

**Scenario**: Testing complex microservices with multiple service dependencies, circuit breakers, and distributed transactions.

```python
from unittest.mock import MagicMock, patch
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import random
import json
from contextlib import contextmanager

class ServiceHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"

@dataclass
class ServiceEndpoint:
    """Represents a microservice endpoint"""
    service_name: str
    endpoint: str
    method: str
    expected_response_time: float = 0.1
    failure_rate: float = 0.0
    health_status: ServiceHealth = ServiceHealth.HEALTHY

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3

class MicroserviceMockOrchestrator:
    """Orchestrates mocks for microservices testing"""
    
    def __init__(self):
        self._services: Dict[str, MagicMock] = {}
        self._service_configs: Dict[str, Dict] = {}
        self._circuit_breakers: Dict[str, Dict] = {}
        self._health_checks: Dict[str, ServiceHealth] = {}
        self._service_dependencies: Dict[str, List[str]] = {}
        self._distributed_transaction_state: Dict[str, Any] = {}
        
    def register_service(self, service_name: str, endpoints: List[ServiceEndpoint]):
        """Register a microservice with its endpoints"""
        service_mock = MagicMock()
        
        # Configure each endpoint
        for endpoint in endpoints:
            self._configure_endpoint(service_mock, endpoint)
        
        self._services[service_name] = service_mock
        self._service_configs[service_name] = {
            'endpoints': endpoints,
            'call_count': 0,
            'errors': 0
        }
        
        # Initialize circuit breaker
        self._circuit_breakers[service_name] = {
            'state': 'closed',
            'failure_count': 0,
            'last_failure_time': None,
            'config': CircuitBreakerConfig()
        }
        
        self._health_checks[service_name] = ServiceHealth.HEALTHY
        
    def _configure_endpoint(self, service_mock: MagicMock, endpoint: ServiceEndpoint):
        """Configure a service endpoint with realistic behavior"""
        method_name = endpoint.endpoint.replace('/', '_').replace('-', '_')
        method_mock = getattr(service_mock, method_name)
        
        def endpoint_call(*args, **kwargs):
            service_name = endpoint.service_name
            
            # Simulate response time
            time.sleep(endpoint.expected_response_time * random.uniform(0.8, 1.2))
            
            # Check circuit breaker
            if self._is_circuit_breaker_open(service_name):
                raise Exception(f"Circuit breaker open for {service_name}")
            
            # Update call count
            self._service_configs[service_name]['call_count'] += 1
            
            # Simulate failures based on health status
            failure_probability = self._calculate_failure_probability(endpoint)
            
            if random.random() < failure_probability:
                self._record_failure(service_name)
                if endpoint.health_status == ServiceHealth.UNHEALTHY:
                    raise Exception(f"Service {service_name} is unhealthy")
                elif endpoint.health_status == ServiceHealth.DEGRADED:
                    raise Exception(f"Service {service_name} is degraded")
                else:
                    raise Exception(f"Random failure in {service_name}")
            
            # Successful response
            return self._generate_response(endpoint, args, kwargs)
        
        method_mock.side_effect = endpoint_call
    
    def _calculate_failure_probability(self, endpoint: ServiceEndpoint) -> float:
        """Calculate failure probability based on health status"""
        base_rate = endpoint.failure_rate
        
        if endpoint.health_status == ServiceHealth.UNHEALTHY:
            return min(0.8, base_rate * 10)
        elif endpoint.health_status == ServiceHealth.DEGRADED:
            return min(0.3, base_rate * 3)
        elif endpoint.health_status == ServiceHealth.MAINTENANCE:
            return 1.0
        
        return base_rate
    
    def _is_circuit_breaker_open(self, service_name: str) -> bool:
        """Check if circuit breaker is open"""
        breaker = self._circuit_breakers[service_name]
        config = breaker['config']
        
        if breaker['state'] == 'open':
            # Check if recovery timeout has passed
            if (time.time() - breaker['last_failure_time']) > config.recovery_timeout:
                breaker['state'] = 'half-open'
                breaker['failure_count'] = 0
                return False
            return True
        
        return False
    
    def _record_failure(self, service_name: str):
        """Record a service failure"""
        breaker = self._circuit_breakers[service_name]
        config = breaker['config']
        
        breaker['failure_count'] += 1
        breaker['last_failure_time'] = time.time()
        self._service_configs[service_name]['errors'] += 1
        
        # Open circuit breaker if threshold reached
        if breaker['failure_count'] >= config.failure_threshold:
            breaker['state'] = 'open'
    
    def _generate_response(self, endpoint: ServiceEndpoint, args: tuple, kwargs: dict) -> Dict:
        """Generate realistic response based on endpoint"""
        base_response = {
            'status': 'success',
            'service': endpoint.service_name,
            'endpoint': endpoint.endpoint,
            'timestamp': time.time()
        }
        
        # Add service-specific data
        if 'user' in endpoint.service_name.lower():
            base_response['data'] = {'id': 123, 'name': 'Test User', 'email': 'user@test.com'}
        elif 'order' in endpoint.service_name.lower():
            base_response['data'] = {'id': 456, 'total': 99.99, 'status': 'confirmed'}
        elif 'payment' in endpoint.service_name.lower():
            base_response['data'] = {'transaction_id': 'txn_789', 'amount': 99.99}
        elif 'notification' in endpoint.service_name.lower():
            base_response['data'] = {'message_id': 'msg_101', 'status': 'sent'}
        
        return base_response
    
    def set_service_health(self, service_name: str, health: ServiceHealth):
        """Set service health status"""
        self._health_checks[service_name] = health
        
        # Update endpoint health
        if service_name in self._service_configs:
            for endpoint in self._service_configs[service_name]['endpoints']:
                endpoint.health_status = health
    
    def add_service_dependency(self, service_name: str, depends_on: List[str]):
        """Define service dependencies"""
        self._service_dependencies[service_name] = depends_on
    
    def simulate_cascade_failure(self, root_service: str):
        """Simulate cascade failure from a root service"""
        # Mark root service as unhealthy
        self.set_service_health(root_service, ServiceHealth.UNHEALTHY)
        
        # Find and mark dependent services as degraded
        for service, dependencies in self._service_dependencies.items():
            if root_service in dependencies:
                self.set_service_health(service, ServiceHealth.DEGRADED)
                
                # Further cascade
                for dependent_service, deps in self._service_dependencies.items():
                    if service in deps:
                        self.set_service_health(dependent_service, ServiceHealth.DEGRADED)
    
    def start_distributed_transaction(self, transaction_id: str, participants: List[str]):
        """Start a distributed transaction"""
        self._distributed_transaction_state[transaction_id] = {
            'participants': participants,
            'status': 'started',
            'completed_phases': [],
            'failed_participants': []
        }
    
    def simulate_distributed_transaction_test(self, transaction_id: str) -> Dict:
        """Simulate a two-phase commit transaction"""
        if transaction_id not in self._distributed_transaction_state:
            raise ValueError(f"Transaction {transaction_id} not found")
        
        transaction = self._distributed_transaction_state[transaction_id]
        participants = transaction['participants']
        
        # Phase 1: Prepare
        prepare_results = {}
        for participant in participants:
            if participant in self._services:
                service = self._services[participant]
                try:
                    result = service.prepare_transaction(transaction_id)
                    prepare_results[participant] = 'prepared'
                except Exception:
                    prepare_results[participant] = 'failed'
                    transaction['failed_participants'].append(participant)
        
        # Phase 2: Commit or Abort
        if len(transaction['failed_participants']) == 0:
            # All prepared successfully, commit
            commit_results = {}
            for participant in participants:
                service = self._services[participant]
                try:
                    service.commit_transaction(transaction_id)
                    commit_results[participant] = 'committed'
                except Exception:
                    commit_results[participant] = 'commit_failed'
            
            transaction['status'] = 'committed'
            return {
                'transaction_id': transaction_id,
                'status': 'committed',
                'prepare_results': prepare_results,
                'commit_results': commit_results
            }
        else:
            # Some failed, abort all
            abort_results = {}
            for participant in participants:
                if participant not in transaction['failed_participants']:
                    service = self._services[participant]
                    try:
                        service.abort_transaction(transaction_id)
                        abort_results[participant] = 'aborted'
                    except Exception:
                        abort_results[participant] = 'abort_failed'
            
            transaction['status'] = 'aborted'
            return {
                'transaction_id': transaction_id,
                'status': 'aborted',
                'prepare_results': prepare_results,
                'abort_results': abort_results
            }
    
    def get_service_metrics(self) -> Dict[str, Dict]:
        """Get metrics for all services"""
        metrics = {}
        for service_name, config in self._service_configs.items():
            breaker = self._circuit_breakers[service_name]
            metrics[service_name] = {
                'call_count': config['call_count'],
                'error_count': config['errors'],
                'error_rate': config['errors'] / max(1, config['call_count']),
                'circuit_breaker_state': breaker['state'],
                'health_status': self._health_checks[service_name].value
            }
        return metrics
    
    def get_service(self, name: str) -> MagicMock:
        """Get a service mock by name"""
        return self._services.get(name)

# Usage Example for Microservices Testing

def test_microservices_orchestration():
    """Test complex microservices scenarios"""
    
    orchestrator = MicroserviceMockOrchestrator()
    
    # Define services
    services_config = [
        ('user_service', [
            ServiceEndpoint('user_service', '/users', 'GET', expected_response_time=0.05),
            ServiceEndpoint('user_service', '/users', 'POST', expected_response_time=0.1)
        ]),
        ('order_service', [
            ServiceEndpoint('order_service', '/orders', 'POST', expected_response_time=0.15),
            ServiceEndpoint('order_service', '/orders/{id}', 'GET', expected_response_time=0.08)
        ]),
        ('payment_service', [
            ServiceEndpoint('payment_service', '/payments', 'POST', expected_response_time=0.2, failure_rate=0.05)
        ]),
        ('notification_service', [
            ServiceEndpoint('notification_service', '/notify', 'POST', expected_response_time=0.1)
        ])
    ]
    
    # Register all services
    for service_name, endpoints in services_config:
        orchestrator.register_service(service_name, endpoints)
    
    # Define dependencies
    orchestrator.add_service_dependency('order_service', ['user_service', 'payment_service'])
    orchestrator.add_service_dependency('notification_service', ['order_service'])
    
    # Configure transaction support for distributed operations
    for service_name, _ in services_config:
        service = orchestrator.get_service(service_name)
        service.prepare_transaction.return_value = {'status': 'prepared'}
        service.commit_transaction.return_value = {'status': 'committed'}
        service.abort_transaction.return_value = {'status': 'aborted'}
    
    # Test normal operations
    user_service = orchestrator.get_service('user_service')
    order_service = orchestrator.get_service('order_service')
    
    # Create user
    user_response = user_service.users()
    assert user_response['status'] == 'success'
    
    # Create order
    order_response = order_service.orders()
    assert order_response['status'] == 'success'
    
    print("✓ Normal operations test passed")
    
    # Test distributed transaction
    orchestrator.start_distributed_transaction('txn_123', [
        'user_service', 'order_service', 'payment_service'
    ])
    
    transaction_result = orchestrator.simulate_distributed_transaction_test('txn_123')
    assert transaction_result['status'] == 'committed'
    print("✓ Distributed transaction test passed")
    
    # Test cascade failure
    orchestrator.simulate_cascade_failure('payment_service')
    
    # Verify cascade effect
    metrics = orchestrator.get_service_metrics()
    assert metrics['payment_service']['health_status'] == 'unhealthy'
    assert metrics['order_service']['health_status'] == 'degraded'
    
    print("✓ Cascade failure test passed")
    
    print(f"Service metrics: {json.dumps(metrics, indent=2)}")

test_microservices_orchestration()
```

---

## Strategy 2: Enterprise Integration Testing

**Scenario**: Testing enterprise systems with legacy integrations, message queues, databases, and external APIs.

```python
from unittest.mock import MagicMock, patch
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import xml.etree.ElementTree as ET
import json
import time

class IntegrationProtocol(Enum):
    REST_API = "rest_api"
    SOAP = "soap"
    MESSAGE_QUEUE = "message_queue"
    DATABASE = "database"
    FILE_TRANSFER = "file_transfer"
    EDI = "edi"

@dataclass
class SystemIntegration:
    """Represents an external system integration"""
    system_name: str
    protocol: IntegrationProtocol
    endpoint: str
    auth_method: str = "none"
    data_format: str = "json"
    timeout: float = 30.0
    retry_attempts: int = 3
    circuit_breaker: bool = True

class EnterpriseIntegrationMocker:
    """Handles mocking of enterprise system integrations"""
    
    def __init__(self):
        self._integrations: Dict[str, SystemIntegration] = {}
        self._mocks: Dict[str, MagicMock] = {}
        self._message_queues: Dict[str, List[Dict]] = {}
        self._databases: Dict[str, Dict[str, List[Dict]]] = {}
        self._file_systems: Dict[str, Dict[str, Any]] = {}
        
    def register_integration(self, integration: SystemIntegration):
        """Register an external system integration"""
        self._integrations[integration.system_name] = integration
        mock = self._create_integration_mock(integration)
        self._mocks[integration.system_name] = mock
        
        # Initialize protocol-specific storage
        if integration.protocol == IntegrationProtocol.MESSAGE_QUEUE:
            self._message_queues[integration.system_name] = []
        elif integration.protocol == IntegrationProtocol.DATABASE:
            self._databases[integration.system_name] = {}
        elif integration.protocol == IntegrationProtocol.FILE_TRANSFER:
            self._file_systems[integration.system_name] = {}
    
    def _create_integration_mock(self, integration: SystemIntegration) -> MagicMock:
        """Create a mock for specific integration type"""
        mock = MagicMock()
        
        if integration.protocol == IntegrationProtocol.REST_API:
            self._configure_rest_api_mock(mock, integration)
        elif integration.protocol == IntegrationProtocol.SOAP:
            self._configure_soap_mock(mock, integration)
        elif integration.protocol == IntegrationProtocol.MESSAGE_QUEUE:
            self._configure_message_queue_mock(mock, integration)
        elif integration.protocol == IntegrationProtocol.DATABASE:
            self._configure_database_mock(mock, integration)
        elif integration.protocol == IntegrationProtocol.FILE_TRANSFER:
            self._configure_file_transfer_mock(mock, integration)
        elif integration.protocol == IntegrationProtocol.EDI:
            self._configure_edi_mock(mock, integration)
        
        return mock
    
    def _configure_rest_api_mock(self, mock: MagicMock, integration: SystemIntegration):
        """Configure REST API mock"""
        def make_request(method: str, endpoint: str, data: Any = None, headers: Dict = None):
            # Simulate authentication
            if integration.auth_method != "none" and not self._validate_auth(headers):
                return {'status': 401, 'error': 'Unauthorized'}
            
            # Simulate timeout
            time.sleep(integration.timeout * 0.001)  # Scaled down for testing
            
            # Generate response based on endpoint
            if method.upper() == 'GET':
                return self._generate_rest_response(endpoint, method)
            elif method.upper() == 'POST':
                return self._generate_rest_response(endpoint, method, data)
            elif method.upper() == 'PUT':
                return self._generate_rest_response(endpoint, method, data)
            elif method.upper() == 'DELETE':
                return {'status': 204, 'message': 'Deleted successfully'}
            
            return {'status': 405, 'error': 'Method not allowed'}
        
        mock.get.side_effect = lambda endpoint, **kwargs: make_request('GET', endpoint, **kwargs)
        mock.post.side_effect = lambda endpoint, data=None, **kwargs: make_request('POST', endpoint, data, **kwargs)
        mock.put.side_effect = lambda endpoint, data=None, **kwargs: make_request('PUT', endpoint, data, **kwargs)
        mock.delete.side_effect = lambda endpoint, **kwargs: make_request('DELETE', endpoint, **kwargs)
    
    def _configure_soap_mock(self, mock: MagicMock, integration: SystemIntegration):
        """Configure SOAP service mock"""
        def call_soap_method(method_name: str, soap_body: str):
            # Parse SOAP XML
            try:
                root = ET.fromstring(soap_body)
                
                # Extract method parameters
                params = self._extract_soap_params(root, method_name)
                
                # Generate SOAP response
                return self._generate_soap_response(method_name, params)
                
            except ET.ParseError:
                return self._generate_soap_fault("Invalid XML")
        
        mock.call.side_effect = call_soap_method
        mock.get_wsdl.return_value = self._generate_wsdl(integration.system_name)
    
    def _configure_message_queue_mock(self, mock: MagicMock, integration: SystemIntegration):
        """Configure message queue mock"""
        queue_name = integration.system_name
        
        def publish_message(queue: str, message: Dict, routing_key: str = None):
            if queue not in self._message_queues:
                self._message_queues[queue] = []
            
            message_wrapper = {
                'id': f"msg_{len(self._message_queues[queue]) + 1}",
                'timestamp': time.time(),
                'routing_key': routing_key,
                'payload': message,
                'attempts': 0
            }
            
            self._message_queues[queue].append(message_wrapper)
            return {'message_id': message_wrapper['id'], 'status': 'published'}
        
        def consume_messages(queue: str, count: int = 1):
            if queue not in self._message_queues:
                return []
            
            messages = self._message_queues[queue][:count]
            self._message_queues[queue] = self._message_queues[queue][count:]
            return messages
        
        def acknowledge_message(queue: str, message_id: str):
            return {'message_id': message_id, 'status': 'acknowledged'}
        
        mock.publish.side_effect = publish_message
        mock.consume.side_effect = consume_messages
        mock.ack.side_effect = acknowledge_message
        mock.get_queue_depth.side_effect = lambda q: len(self._message_queues.get(q, []))
    
    def _configure_database_mock(self, mock: MagicMock, integration: SystemIntegration):
        """Configure legacy database mock"""
        db_name = integration.system_name
        
        def execute_query(sql: str, params: tuple = None):
            # Simple SQL parsing for testing
            sql_upper = sql.upper().strip()
            
            if sql_upper.startswith('SELECT'):
                return self._handle_select_query(db_name, sql, params)
            elif sql_upper.startswith('INSERT'):
                return self._handle_insert_query(db_name, sql, params)
            elif sql_upper.startswith('UPDATE'):
                return self._handle_update_query(db_name, sql, params)
            elif sql_upper.startswith('DELETE'):
                return self._handle_delete_query(db_name, sql, params)
            else:
                return {'status': 'executed', 'affected_rows': 0}
        
        def execute_stored_procedure(proc_name: str, params: Dict):
            return self._execute_mock_stored_procedure(db_name, proc_name, params)
        
        mock.execute.side_effect = execute_query
        mock.call_procedure.side_effect = execute_stored_procedure
        mock.begin_transaction.return_value = {'transaction_id': 'tx_123'}
        mock.commit.return_value = {'status': 'committed'}
        mock.rollback.return_value = {'status': 'rolled_back'}
    
    def _configure_file_transfer_mock(self, mock: MagicMock, integration: SystemIntegration):
        """Configure file transfer mock (FTP, SFTP, etc.)"""
        system_name = integration.system_name
        
        def upload_file(local_path: str, remote_path: str, content: bytes):
            if system_name not in self._file_systems:
                self._file_systems[system_name] = {}
            
            self._file_systems[system_name][remote_path] = {
                'content': content,
                'size': len(content),
                'upload_time': time.time(),
                'local_path': local_path
            }
            
            return {
                'status': 'uploaded',
                'remote_path': remote_path,
                'size': len(content)
            }
        
        def download_file(remote_path: str):
            if (system_name in self._file_systems and 
                remote_path in self._file_systems[system_name]):
                file_info = self._file_systems[system_name][remote_path]
                return {
                    'status': 'downloaded',
                    'content': file_info['content'],
                    'size': file_info['size']
                }
            else:
                raise FileNotFoundError(f"File {remote_path} not found")
        
        def list_files(remote_dir: str = "/"):
            if system_name not in self._file_systems:
                return []
            
            return [
                {
                    'path': path,
                    'size': info['size'],
                    'modified': info['upload_time']
                }
                for path, info in self._file_systems[system_name].items()
                if path.startswith(remote_dir)
            ]
        
        mock.upload.side_effect = upload_file
        mock.download.side_effect = download_file
        mock.list_files.side_effect = list_files
        mock.connect.return_value = {'status': 'connected'}
        mock.disconnect.return_value = {'status': 'disconnected'}
    
    def _configure_edi_mock(self, mock: MagicMock, integration: SystemIntegration):
        """Configure EDI (Electronic Data Interchange) mock"""
        def send_edi_document(document_type: str, edi_content: str):
            # Parse EDI segments (simplified)
            segments = edi_content.split('~')
            
            # Generate EDI acknowledgment
            ack_segments = [
                "ISA*00*          *00*          *ZZ*SENDER         *ZZ*RECEIVER       *230101*1200*^*00501*000000001*0*P*:~",
                "GS*FA*SENDER*RECEIVER*20230101*1200*1*X*005010~",
                f"ST*997*0001~",
                "AK1*{document_type}*1~",
                "AK9*A*1*1*1~",
                "SE*4*0001~",
                "GE*1*1~",
                "IEA*1*000000001~"
            ]
            
            return {
                'status': 'sent',
                'document_type': document_type,
                'control_number': '000000001',
                'acknowledgment': '~'.join(ack_segments)
            }
        
        def receive_edi_document():
            # Simulate receiving an EDI document
            sample_edi = [
                "ISA*00*          *00*          *ZZ*PARTNER        *ZZ*COMPANY        *230101*1200*^*00501*000000002*0*P*:~",
                "GS*PO*PARTNER*COMPANY*20230101*1200*2*X*005010~",
                "ST*850*0002~",
                "BEG*00*SA*PO123456*20230101~",
                "PER*BD*John Doe*TE*555-1234~",
                "N1*ST*Ship To Location~",
                "N4*New York*NY*10001*US~",
                "PO1*1*100*EA*10.50*PE*SKU123*BP*PART123~",
                "CTT*1*100~",
                "SE*8*0002~",
                "GE*1*2~",
                "IEA*1*000000002~"
            ]
            
            return {
                'document_type': '850',  # Purchase Order
                'control_number': '000000002',
                'content': '~'.join(sample_edi),
                'received_time': time.time()
            }
        
        mock.send_document.side_effect = send_edi_document
        mock.receive_document.side_effect = receive_edi_document
        mock.validate_document.return_value = {'status': 'valid', 'errors': []}
    
    # Helper methods for generating realistic responses
    
    def _validate_auth(self, headers: Optional[Dict]) -> bool:
        """Validate authentication headers"""
        if not headers:
            return False
        return 'Authorization' in headers or 'X-API-Key' in headers
    
    def _generate_rest_response(self, endpoint: str, method: str, data: Any = None) -> Dict:
        """Generate realistic REST response"""
        if 'customer' in endpoint.lower():
            if method == 'GET':
                return {
                    'status': 200,
                    'data': {
                        'id': 123,
                        'name': 'Enterprise Customer',
                        'type': 'corporate',
                        'credit_limit': 50000.00
                    }
                }
            elif method == 'POST':
                return {
                    'status': 201,
                    'data': {
                        'id': 124,
                        'message': 'Customer created successfully'
                    }
                }
        elif 'order' in endpoint.lower():
            return {
                'status': 200,
                'data': {
                    'order_id': 'ORD-789',
                    'total': 1299.99,
                    'status': 'pending',
                    'items': [
                        {'sku': 'ITEM-001', 'quantity': 2, 'price': 649.99}
                    ]
                }
            }
        
        return {'status': 200, 'message': 'Success'}
    
    def _generate_soap_response(self, method_name: str, params: Dict) -> str:
        """Generate SOAP response"""
        if method_name == 'GetCustomerInfo':
            return """<?xml version="1.0" encoding="UTF-8"?>
            <soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
                <soap:Body>
                    <GetCustomerInfoResponse>
                        <CustomerId>12345</CustomerId>
                        <CustomerName>Enterprise Corp</CustomerName>
                        <Status>Active</Status>
                        <CreditLimit>100000.00</CreditLimit>
                    </GetCustomerInfoResponse>
                </soap:Body>
            </soap:Envelope>"""
        
        return """<?xml version="1.0" encoding="UTF-8"?>
        <soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
            <soap:Body>
                <GenericResponse>
                    <Status>Success</Status>
                    <Message>Operation completed</Message>
                </GenericResponse>
            </soap:Body>
        </soap:Envelope>"""
    
    def _generate_soap_fault(self, error_message: str) -> str:
        """Generate SOAP fault response"""
        return f"""<?xml version="1.0" encoding="UTF-8"?>
        <soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
            <soap:Body>
                <soap:Fault>
                    <faultcode>Client</faultcode>
                    <faultstring>{error_message}</faultstring>
                </soap:Fault>
            </soap:Body>
        </soap:Envelope>"""
    
    def get_integration_mock(self, system_name: str) -> MagicMock:
        """Get integration mock by system name"""
        return self._mocks.get(system_name)
    
    def get_system_metrics(self) -> Dict[str, Dict]:
        """Get metrics for all integrated systems"""
        metrics = {}
        for system_name, mock in self._mocks.items():
            integration = self._integrations[system_name]
            metrics[system_name] = {
                'protocol': integration.protocol.value,
                'call_count': mock.call_count,
                'last_called': getattr(mock, '_last_call_time', None),
                'status': 'active'
            }
            
            # Add protocol-specific metrics
            if integration.protocol == IntegrationProtocol.MESSAGE_QUEUE:
                metrics[system_name]['queue_depth'] = len(self._message_queues.get(system_name, []))
            elif integration.protocol == IntegrationProtocol.FILE_TRANSFER:
                metrics[system_name]['files_count'] = len(self._file_systems.get(system_name, {}))
        
        return metrics

# Usage Example

def test_enterprise_integration():
    """Test enterprise system integrations"""
    
    mocker = EnterpriseIntegrationMocker()
    
    # Register various enterprise integrations
    integrations = [
        SystemIntegration(
            system_name="legacy_mainframe",
            protocol=IntegrationProtocol.DATABASE,
            endpoint="jdbc:db2://mainframe:50000/PROD",
            auth_method="basic",
            timeout=10.0
        ),
        SystemIntegration(
            system_name="partner_api",
            protocol=IntegrationProtocol.REST_API,
            endpoint="https://partner.example.com/api/v1",
            auth_method="oauth2",
            data_format="json"
        ),
        SystemIntegration(
            system_name="edi_trading_partner",
            protocol=IntegrationProtocol.EDI,
            endpoint="edi://partner.tradingnetwork.com",
            data_format="x12"
        ),
        SystemIntegration(
            system_name="message_broker",
            protocol=IntegrationProtocol.MESSAGE_QUEUE,
            endpoint="amqp://broker.company.com:5672",
            auth_method="certificate"
        ),
        SystemIntegration(
            system_name="file_server",
            protocol=IntegrationProtocol.FILE_TRANSFER,
            endpoint="sftp://files.company.com",
            auth_method="key_pair"
        )
    ]
    
    for integration in integrations:
        mocker.register_integration(integration)
    
    # Test REST API integration
    partner_api = mocker.get_integration_mock("partner_api")
    
    # Test with authentication
    headers = {'Authorization': 'Bearer token123'}
    customer_response = partner_api.get('/customers/123', headers=headers)
    assert customer_response['status'] == 200
    assert 'Enterprise Customer' in customer_response['data']['name']
    
    # Test database integration
    mainframe_db = mocker.get_integration_mock("legacy_mainframe")
    
    query_result = mainframe_db.execute("SELECT * FROM CUSTOMER_MASTER WHERE CUST_ID = ?", (123,))
    assert query_result['status'] == 'success'
    
    # Test stored procedure
    proc_result = mainframe_db.call_procedure("GET_CUSTOMER_ORDERS", {"customer_id": 123})
    assert 'data' in proc_result
    
    # Test message queue integration
    message_broker = mocker.get_integration_mock("message_broker")
    
    # Publish message
    publish_result = message_broker.publish("orders", {
        "order_id": "ORD-123",
        "customer_id": 456,
        "total": 999.99
    })
    assert publish_result['status'] == 'published'
    
    # Consume message
    messages = message_broker.consume("orders", 1)
    assert len(messages) == 1
    assert messages[0]['payload']['order_id'] == "ORD-123"
    
    # Test EDI integration
    edi_partner = mocker.get_integration_mock("edi_trading_partner")
    
    # Send purchase order
    po_edi = "ST*850*0001~BEG*00*SA*PO123*20230101~SE*2*0001~"
    edi_response = edi_partner.send_document("850", po_edi)
    assert edi_response['status'] == 'sent'
    assert 'acknowledgment' in edi_response
    
    # Receive document
    received_doc = edi_partner.receive_document()
    assert received_doc['document_type'] == '850'
    
    # Test file transfer integration
    file_server = mocker.get_integration_mock("file_server")
    
    # Upload file
    test_content = b"Customer data export - 2023-01-01"
    upload_result = file_server.upload("/local/export.csv", "/remote/exports/export.csv", test_content)
    assert upload_result['status'] == 'uploaded'
    
    # Download file
    download_result = file_server.download("/remote/exports/export.csv")
    assert download_result['content'] == test_content
    
    # List files
    file_list = file_server.list_files("/remote/exports/")
    assert len(file_list) == 1
    assert file_list[0]['path'] == "/remote/exports/export.csv"
    
    # Get overall metrics
    metrics = mocker.get_system_metrics()
    print(f"Integration metrics: {json.dumps(metrics, indent=2)}")
    
    # Verify all systems are working
    for system_name in ['legacy_mainframe', 'partner_api', 'edi_trading_partner', 'message_broker', 'file_server']:
        assert metrics[system_name]['call_count'] > 0
        assert metrics[system_name]['status'] == 'active'
    
    print("✓ All enterprise integration tests passed!")

test_enterprise_integration()
```

---

## Strategy 3: Performance and Load Testing with Mocks

**Scenario**: Using mocks to simulate high-load scenarios and performance bottlenecks.

```python
from unittest.mock import MagicMock
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass
import time
import threading
import concurrent.futures
from collections import defaultdict
import statistics

@dataclass
class PerformanceProfile:
    """Performance characteristics for a mock"""
    avg_response_time: float
    response_time_variance: float
    throughput_per_second: int
    error_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float

@dataclass
class LoadTestScenario:
    """Load test scenario configuration"""
    name: str
    concurrent_users: int
    duration_seconds: int
    ramp_up_seconds: int
    operations_per_user: List[Callable]
    expected_throughput: int
    max_acceptable_response_time: float

class PerformanceTestingMockFramework:
    """Framework for performance testing with realistic mocks"""
    
    def __init__(self):
        self._performance_profiles: Dict[str, PerformanceProfile] = {}
        self._mocks: Dict[str, MagicMock] = {}
        self._performance_data: Dict[str, List[Dict]] = defaultdict(list)
        self._load_test_results: List[Dict] = []
        
    def register_performance_mock(self, name: str, profile: PerformanceProfile) -> MagicMock:
        """Register a mock with performance characteristics"""
        self._performance_profiles[name] = profile
        
        mock = MagicMock()
        self._configure_performance_mock(mock, name, profile)
        self._mocks[name] = mock
        
        return mock
    
    def _configure_performance_mock(self, mock: MagicMock, name: str, profile: PerformanceProfile):
        """Configure mock with performance simulation"""
        call_count = {'value': 0}
        start_time = {'value': time.time()}
        
        original_getattr = mock.__getattribute__
        
        def performance_aware_getattr(attr_name):
            attr = original_getattr(attr_name)
            
            if callable(attr) and not attr_name.startswith('_'):
                def performance_wrapper(*args, **kwargs):
                    call_start = time.time()
                    call_count['value'] += 1
                    
                    # Check throughput limits
                    elapsed_time = call_start - start_time['value']
                    if elapsed_time > 0:
                        current_throughput = call_count['value'] / elapsed_time
                        if current_throughput > profile.throughput_per_second:
                            # Throttle the call
                            delay = (call_count['value'] / profile.throughput_per_second) - elapsed_time
                            if delay > 0:
                                time.sleep(delay)
                    
                    # Simulate response time
                    import random
                    response_time = random.gauss(
                        profile.avg_response_time, 
                        profile.response_time_variance
                    )
                    response_time = max(0.001, response_time)  # Minimum 1ms
                    
                    time.sleep(response_time)
                    
                    # Simulate errors
                    if random.random() < profile.error_rate:
                        raise Exception(f"Simulated error in {name}.{attr_name}")
                    
                    # Record performance data
                    call_end = time.time()
                    self._record_performance_data(name, attr_name, {
                        'call_start': call_start,
                        'call_end': call_end,
                        'response_time': response_time,
                        'args_count': len(args),
                        'kwargs_count': len(kwargs)
                    })
                    
                    # Generate appropriate response
                    return self._generate_realistic_response(name, attr_name, args, kwargs)
                
                setattr(mock, attr_name, performance_wrapper)
                return performance_wrapper
            
            return attr
        
        mock.__getattribute__ = performance_aware_getattr
    
    def _record_performance_data(self, mock_name: str, method_name: str, data: Dict):
        """Record performance metrics"""
        key = f"{mock_name}.{method_name}"
        self._performance_data[key].append(data)
    
    def _generate_realistic_response(self, mock_name: str, method_name: str, args: tuple, kwargs: dict) -> Any:
        """Generate realistic response based on mock and method"""
        if 'database' in mock_name.lower():
            if 'query' in method_name.lower() or 'select' in method_name.lower():
                return [{'id': i, 'name': f'Record {i}'} for i in range(1, 11)]
            elif 'insert' in method_name.lower() or 'create' in method_name.lower():
                return {'id': 123, 'status': 'created'}
            elif 'update' in method_name.lower():
                return {'affected_rows': 1, 'status': 'updated'}
            elif 'delete' in method_name.lower():
                return {'affected_rows': 1, 'status': 'deleted'}
        
        elif 'api' in mock_name.lower():
            return {
                'status': 'success',
                'data': {'result': 'API response'},
                'timestamp': time.time()
            }
        
        elif 'cache' in mock_name.lower():
            if 'get' in method_name.lower():
                return {'cached_value': 'test_data', 'hit': True}
            elif 'set' in method_name.lower():
                return {'status': 'cached'}
        
        return {'status': 'success', 'message': f'Response from {mock_name}.{method_name}'}
    
    def run_load_test(self, scenario: LoadTestScenario) -> Dict[str, Any]:
        """Run a load test scenario"""
        print(f"Starting load test: {scenario.name}")
        print(f"Users: {scenario.concurrent_users}, Duration: {scenario.duration_seconds}s")
        
        results = {
            'scenario_name': scenario.name,
            'start_time': time.time(),
            'end_time': None,
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'response_times': [],
            'throughput_per_second': 0,
            'errors': [],
            'performance_summary': {}
        }
        
        # Thread-safe counters
        operation_count = threading.Lock()
        success_count = threading.Lock()
        failure_count = threading.Lock()
        response_times = threading.Lock()
        errors = threading.Lock()
        
        counters = {
            'operations': 0,
            'successes': 0,
            'failures': 0,
            'times': [],
            'error_list': []
        }
        
        def user_simulation(user_id: int):
            """Simulate a single user's operations"""
            user_start_delay = (user_id / scenario.concurrent_users) * scenario.ramp_up_seconds
            time.sleep(user_start_delay)
            
            user_end_time = time.time() + scenario.duration_seconds - user_start_delay
            
            while time.time() < user_end_time:
                for operation in scenario.operations_per_user:
                    if time.time() >= user_end_time:
                        break
                    
                    operation_start = time.time()
                    
                    try:
                        operation()
                        
                        with success_count:
                            counters['successes'] += 1
                        
                    except Exception as e:
                        with failure_count:
                            counters['failures'] += 1
                        
                        with errors:
                            counters['error_list'].append(str(e))
                    
                    operation_end = time.time()
                    operation_time = operation_end - operation_start
                    
                    with operation_count:
                        counters['operations'] += 1
                    
                    with response_times:
                        counters['times'].append(operation_time)
        
        # Run load test with thread pool
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=scenario.concurrent_users) as executor:
            futures = [
                executor.submit(user_simulation, user_id)
                for user_id in range(scenario.concurrent_users)
            ]
            
            # Wait for all users to complete
            concurrent.futures.wait(futures)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Compile results
        results.update({
            'end_time': end_time,
            'total_operations': counters['operations'],
            'successful_operations': counters['successes'],
            'failed_operations': counters['failures'],
            'response_times': counters['times'],
            'throughput_per_second': counters['operations'] / total_duration if total_duration > 0 else 0,
            'errors': counters['error_list'][:100],  # Limit errors to avoid memory issues
            'actual_duration': total_duration
        })
        
        # Calculate performance statistics
        if counters['times']:
            results['performance_summary'] = {
                'avg_response_time': statistics.mean(counters['times']),
                'median_response_time': statistics.median(counters['times']),
                'p95_response_time': self._calculate_percentile(counters['times'], 95),
                'p99_response_time': self._calculate_percentile(counters['times'], 99),
                'min_response_time': min(counters['times']),
                'max_response_time': max(counters['times']),
                'std_deviation': statistics.stdev(counters['times']) if len(counters['times']) > 1 else 0
            }
        
        # Analyze results against expectations
        results['test_passed'] = self._analyze_test_results(scenario, results)
        
        self._load_test_results.append(results)
        return results
    
    def _calculate_percentile(self, values: List[float], percentile: int) -> float:
        """Calculate percentile value"""
        sorted_values = sorted(values)
        index = int((percentile / 100) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]
    
    def _analyze_test_results(self, scenario: LoadTestScenario, results: Dict) -> bool:
        """Analyze if test results meet expectations"""
        performance = results['performance_summary']
        
        # Check throughput
        if results['throughput_per_second'] < scenario.expected_throughput * 0.8:  # 80% threshold
            return False
        
        # Check response time
        if performance.get('avg_response_time', float('inf')) > scenario.max_acceptable_response_time:
            return False
        
        # Check error rate (should be < 5%)
        error_rate = results['failed_operations'] / max(1, results['total_operations'])
        if error_rate > 0.05:
            return False
        
        return True
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'test_summary': {
                'total_scenarios': len(self._load_test_results),
                'passed_scenarios': sum(1 for r in self._load_test_results if r['test_passed']),
                'failed_scenarios': sum(1 for r in self._load_test_results if not r['test_passed'])
            },
            'scenarios': self._load_test_results,
            'mock_performance': {}
        }
        
        # Aggregate mock performance data
        for mock_method, data_points in self._performance_data.items():
            if data_points:
                response_times = [dp['response_time'] for dp in data_points]
                report['mock_performance'][mock_method] = {
                    'total_calls': len(data_points),
                    'avg_response_time': statistics.mean(response_times),
                    'p95_response_time': self._calculate_percentile(response_times, 95),
                    'throughput': len(data_points) / (data_points[-1]['call_end'] - data_points[0]['call_start'])
                }
        
        return report

# Usage Example

def test_performance_testing_framework():
    """Test the performance testing framework"""
    
    framework = PerformanceTestingMockFramework()
    
    # Register performance mocks
    database_profile = PerformanceProfile(
        avg_response_time=0.05,
        response_time_variance=0.01,
        throughput_per_second=1000,
        error_rate=0.01,
        memory_usage_mb=100,
        cpu_usage_percent=20
    )
    
    api_profile = PerformanceProfile(
        avg_response_time=0.2,
        response_time_variance=0.05,
        throughput_per_second=500,
        error_rate=0.02,
        memory_usage_mb=50,
        cpu_usage_percent=15
    )
    
    cache_profile = PerformanceProfile(
        avg_response_time=0.001,
        response_time_variance=0.0002,
        throughput_per_second=5000,
        error_rate=0.001,
        memory_usage_mb=200,
        cpu_usage_percent=5
    )
    
    # Register mocks
    db_mock = framework.register_performance_mock("database_service", database_profile)
    api_mock = framework.register_performance_mock("external_api", api_profile)
    cache_mock = framework.register_performance_mock("cache_service", cache_profile)
    
    # Define test operations
    def database_operation():
        results = db_mock.query("SELECT * FROM users WHERE active = 1")
        return results
    
    def api_operation():
        response = api_mock.fetch_user_data(user_id=123)
        return response
    
    def cache_operation():
        cached_data = cache_mock.get("user:123")
        if not cached_data['hit']:
            cache_mock.set("user:123", {"name": "Test User"})
        return cached_data
    
    # Define load test scenarios
    light_load_scenario = LoadTestScenario(
        name="Light Load Test",
        concurrent_users=10,
        duration_seconds=30,
        ramp_up_seconds=5,
        operations_per_user=[database_operation, cache_operation],
        expected_throughput=100,
        max_acceptable_response_time=0.1
    )
    
    medium_load_scenario = LoadTestScenario(
        name="Medium Load Test",
        concurrent_users=50,
        duration_seconds=60,
        ramp_up_seconds=10,
        operations_per_user=[database_operation, api_operation, cache_operation],
        expected_throughput=200,
        max_acceptable_response_time=0.3
    )
    
    heavy_load_scenario = LoadTestScenario(
        name="Heavy Load Test",
        concurrent_users=100,
        duration_seconds=120,
        ramp_up_seconds=20,
        operations_per_user=[database_operation, api_operation, cache_operation],
        expected_throughput=300,
        max_acceptable_response_time=0.5
    )
    
    # Run load tests
    scenarios = [light_load_scenario, medium_load_scenario, heavy_load_scenario]
    
    for scenario in scenarios:
        print(f"\n{'='*50}")
        result = framework.run_load_test(scenario)
        
        print(f"Scenario: {result['scenario_name']}")
        print(f"Operations: {result['total_operations']}")
        print(f"Success Rate: {(result['successful_operations']/result['total_operations']*100):.2f}%")
        print(f"Throughput: {result['throughput_per_second']:.2f} ops/sec")
        print(f"Avg Response Time: {result['performance_summary']['avg_response_time']:.3f}s")
        print(f"P95 Response Time: {result['performance_summary']['p95_response_time']:.3f}s")
        print(f"Test Passed: {result['test_passed']}")
    
    # Generate comprehensive report
    report = framework.get_performance_report()
    
    print(f"\n{'='*50}")
    print("PERFORMANCE REPORT")
    print(f"{'='*50}")
    print(f"Total Scenarios: {report['test_summary']['total_scenarios']}")
    print(f"Passed: {report['test_summary']['passed_scenarios']}")
    print(f"Failed: {report['test_summary']['failed_scenarios']}")
    
    print(f"\nMock Performance Summary:")
    for mock_method, perf_data in report['mock_performance'].items():
        print(f"  {mock_method}:")
        print(f"    Calls: {perf_data['total_calls']}")
        print(f"    Avg Response: {perf_data['avg_response_time']:.3f}s")
        print(f"    Throughput: {perf_data['throughput']:.2f} calls/sec")

# Note: This test will take a few minutes to complete due to the load testing
print("Starting comprehensive performance testing (this will take a few minutes)...")
test_performance_testing_framework()
print("✓ Performance testing framework test completed!")
```

---

## Summary

These enterprise testing strategies provide:

1. **Microservices Testing Architecture** - Complex service orchestration with circuit breakers and distributed transactions
2. **Enterprise Integration Testing** - Legacy systems, EDI, message queues, and multiple protocols
3. **Performance and Load Testing** - Realistic performance simulation and load testing capabilities

These strategies enable:
- **Complex System Testing** - End-to-end testing of enterprise architectures
- **Integration Validation** - Testing multiple protocols and legacy systems
- **Performance Validation** - Load testing with realistic performance characteristics
- **Risk Mitigation** - Circuit breaker and failure cascade testing
- **Compliance Testing** - EDI and enterprise protocol compliance

## Implementation Guidelines

1. **Start Small** - Begin with one integration pattern and expand
2. **Use Realistic Data** - Mock with production-like data volumes and patterns
3. **Monitor Performance** - Track mock performance metrics during tests
4. **Automate Testing** - Integrate with CI/CD pipelines for continuous testing
5. **Document Patterns** - Create reusable patterns for your organization

## Next Steps

- Implement these strategies in your enterprise projects
- Create organization-specific mock libraries
- Train teams on enterprise testing patterns
- Integrate with your existing testing infrastructure