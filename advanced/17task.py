from unittest.mock import patch

def process_payment(amount):
    from payment_gateway import change_card
    from email_service import send_receipt
    from logger import log_transaction
    
    transaction_id = change_card(amount)
    send_receipt(transaction_id)
    log_transaction(transaction_id, amount)
    return transaction_id

with patch('payment_gateway.change_card', return_value='TX123') as mock_charge, \
     patch('email_service.send_receipt') as mock_email, \
     patch('logger.log_transaction') as mock_log:
    
    tx_id = process_payment(100)
    print(f"Transaction ID: {tx_id}")  # Output: Transaction ID: TX123
    
    mock_charge.assert_called_once_with(100)
    mock_email.assert_called_once_with('TX123')
    mock_log.assert_called_once_with('TX123', 100)