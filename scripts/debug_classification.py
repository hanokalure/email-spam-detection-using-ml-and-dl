#!/usr/bin/env python3
"""
Debug the classification logic in detail
"""
import sys
import os
import re

def test_classification_logic():
    """Test the classification logic step by step"""
    
    messages = [
        "Info: ₹2500 cash withdrawn using ATM card ending 7890. Transaction Date: 15-Jan-24, Time: 14:45. Current Balance: ₹12500.",
        "Payment of $75 has been processed successfully. Debited from account ending 5678. Transaction ID: TXN123456789. Balance: $425.",
        "Dear Customer, your mini statement: 1. 14-Jan: ₹2000 CR, 2. 15-Jan: ₹500 DR. Current Balance: ₹47500. Account statement available."
    ]
    
    probabilities = [0.999521, 0.999500, 0.999507]
    
    print("🔍 Debug Classification Logic:")
    print("=" * 80)
    
    for i, (text, probability) in enumerate(zip(messages, probabilities), 1):
        text_lower = text.lower()
        print(f"\n{i}. Message: {text}")
        print(f"   Probability: {probability:.6f}")
        print(f"   Text (lower): {text_lower}")
        
        # Test money detection
        has_money = (any(symbol in text for symbol in ['₹', '$', '€', '£', '¥']) or
                    any(pattern in text_lower for pattern in ['$', 'usd', 'inr', 'eur', 'gbp']) or
                    any(word in text_lower for word in ['balance', 'account', 'payment', 'debit', 'credit', 'transaction']))
        print(f"   has_money: {has_money}")
        
        if has_money and probability >= 0.8:
            # Test spam patterns
            spam_patterns = [
                'click here', 'urgent', 'suspended', 'verify now',
                'act now', 'limited time', 'congratulations',
                'you have won', 'claim now', 'free money',
                'guaranteed', 'investment opportunity', 'double your money'
            ]
            spam_score = sum(1 for pattern in spam_patterns if pattern in text_lower)
            print(f"   spam_score: {spam_score}")
            
            # Test clean financial phrases
            clean_financial_phrases = [
                'your account balance is', 'balance is ₹', 'balance is $', 
                'payment has been', 'transaction successful',
                'receipt for your payment', 'successfully processed',
                'invoice is attached', 'payment of ₹', 'payment of $',
                'has been debited from', 'has been credited to',
                'debited from your account', 'credited to your account',
                'available balance:', 'current balance:', 'account balance:',
                '₹ has been debited', '$ has been debited',
                '₹ has been credited', '$ has been credited',
                'last transaction:', 'minimum balance maintained',
                'thank you for banking',
                # Enhanced ATM withdrawal patterns
                '₹ withdrawn using', '$ withdrawn using',
                'withdrawn using atm', 'cash withdrawal of ₹',
                'cash withdrawal of $', 'cash withdrawn using',
                'withdrawal of ₹', 'withdrawal of $',
                'withdrawn using', 'atm card ending',
                'atm card ending in', 'card ending in',
                'card ending with', 'card ending ',
                'at atm on', 'at atm located',
                'balance: ₹', 'balance: $',
                'available bal:', 'avl bal:',
                'bal after txn:', 'balance after transaction:',
                # Transaction notification patterns
                'transaction date:', 'transaction time:',
                'txn date:', 'txn time:',
                'ref no:', 'reference no:',
                'transaction id:', 'txn id:', 'rrn:', 'time:',
                # Common ATM message formats
                'dear customer,', 'dear valued customer,',
                'notification: ₹', 'notification: $',
                'alert: ₹', 'alert: $',
                'info: ₹', 'info: $', 'info:',
                # Payment processing patterns
                'payment has been processed', 'has been processed',
                'processed successfully', 'successfully processed',
                'debited from account ending',
                # Statement patterns
                'mini statement:', 'statement:', 'account statement',
                'statement available',
                # Additional patterns
                'reference number:', 'transaction id:',
                'txn id:', 'debited from account ending',
                'account ending ', 'account statement available'
            ]
            
            clean_phrase_matches = sum(1 for phrase in clean_financial_phrases if phrase in text_lower)
            print(f"   clean_phrase_matches: {clean_phrase_matches}")
            
            # Test timestamp detection
            timestamp_patterns = [
                r'\d{1,2}[-/]\w{3}[-/]\d{4}',  # 01-Jan-2024
                r'\d{1,2}:\d{2}\s*(AM|PM)',    # 2:30 PM
                r'\d{2}/\d{2}/\d{4}',          # 01/15/2024
                r'\d{2}-\d{2}-\d{4}',          # 01-15-2024
                r'\d{4}-\d{2}-\d{2}',          # 2024-01-15
                r'\d{1,2}\s+\w{3}\s+\d{4}',   # 15 Jan 2024
                r'\w{3}\s+\d{1,2},?\s+\d{4}', # Jan 15, 2024
                r'\d{1,2}:\d{2}:\d{2}',       # 14:30:45
                r'\d{2}-\w{3}-\d{2}',          # 15-Jan-24
                r'on\s+\d{1,2}[-/]\d{1,2}',   # on 15/01
                r'\d{1,2}[a-z]{2}\s+\w+',     # 15th Jan
            ]
            has_timestamp = any(re.search(pattern, text, re.IGNORECASE) for pattern in timestamp_patterns)
            print(f"   has_timestamp: {has_timestamp}")
            
            # Test masked account detection
            masked_account_patterns = [
                r'xxxx\d{4}',                 # xxxx1234
                r'account.*xxxx',             # account xxxx
                r'ending.*\d{4}',             # ending 1234
                r'ending\s+in\s+\d{4}',       # ending in 1234
                r'ending\s+with\s+\d{4}',     # ending with 1234
                r'card\s+ending\s+\d{4}',     # card ending 1234
                r'\*{4,}\d{4}',               # ****1234
                r'\d{4}\*{4,}\d{4}',          # 1234****5678
                r'account\s+no\.?\s*:\s*x+\d+' # account no: xxx1234
            ]
            has_masked_account = any(re.search(pattern, text_lower) for pattern in masked_account_patterns)
            print(f"   has_masked_account: {has_masked_account}")
            
            # Test banking context
            banking_context_indicators = [
                'minimum balance', 'thank you for banking', 'last transaction',
                'transaction details', 'transaction summary', 'balance enquiry',
                'mini statement', 'account statement', 'dear customer',
                'dear valued customer', 'notification from', 'alert from',
                'message from your bank', 'update from', 'cardholder',
                'account holder', 'cash withdrawal', 'withdrawal transaction',
                'atm withdrawal', 'funds withdrawn', 'amount withdrawn',
                'transaction history', 'balance after transaction'
            ]
            has_legitimate_banking_context = any(indicator in text_lower for indicator in banking_context_indicators)
            print(f"   has_legitimate_banking_context: {has_legitimate_banking_context}")
            
            # Test classification conditions
            print(f"\\n   Classification tests:")
            print(f"   - clean_phrase_matches >= 3 and spam_score == 0 and banking context: {clean_phrase_matches >= 3 and spam_score == 0 and (has_timestamp or has_masked_account or has_legitimate_banking_context)}")
            print(f"   - clean_phrase_matches >= 5 and spam_score == 0: {clean_phrase_matches >= 5 and spam_score == 0}")
            print(f"   - clean_phrase_matches >= 3 and spam_score == 0: {clean_phrase_matches >= 3 and spam_score == 0}")
            
            # Test thresholds
            print(f"\\n   Threshold tests:")
            print(f"   - probability >= 0.99995: {probability >= 0.99995}")
            print(f"   - probability >= 0.9999: {probability >= 0.9999}")
            print(f"   - probability >= 0.9998: {probability >= 0.9998}")

if __name__ == "__main__":
    test_classification_logic()