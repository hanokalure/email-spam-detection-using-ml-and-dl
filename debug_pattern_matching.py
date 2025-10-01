#!/usr/bin/env python3
"""
Debug pattern matching for the problematic messages
"""

def test_patterns():
    """Test pattern matching for specific messages"""
    
    # Messages that should be HAM but are classified as SPAM
    messages = [
        "Info: ₹2500 cash withdrawn using ATM card ending 7890. Transaction Date: 15-Jan-24, Time: 14:45. Current Balance: ₹12500.",
        "Payment of $75 has been processed successfully. Debited from account ending 5678. Transaction ID: TXN123456789. Balance: $425.",
        "Dear Customer, your mini statement: 1. 14-Jan: ₹2000 CR, 2. 15-Jan: ₹500 DR. Current Balance: ₹47500. Account statement available."
    ]
    
    # Clean financial message patterns from the classifier
    clean_financial_phrases = [
        'your account balance is',
        'balance is ₹', 'balance is $', 
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
    
    print("🔍 Testing Pattern Matching:")
    print("=" * 80)
    
    for i, text in enumerate(messages, 1):
        text_lower = text.lower()
        print(f"\n{i}. Message: {text}")
        print(f"   Lowercase: {text_lower}")
        
        # Check which patterns match
        matching_patterns = []
        for pattern in clean_financial_phrases:
            if pattern in text_lower:
                matching_patterns.append(pattern)
        
        print(f"   Matching patterns ({len(matching_patterns)}):")
        for pattern in matching_patterns:
            print(f"     • '{pattern}'")
        
        if not matching_patterns:
            print("     ❌ NO PATTERNS MATCHED!")
        
        # Check spam patterns
        spam_patterns = [
            'click here', 'urgent', 'suspended', 'verify now',
            'act now', 'limited time', 'congratulations',
            'you have won', 'claim now', 'free money',
            'guaranteed', 'investment opportunity', 'double your money'
        ]
        
        spam_matches = [pattern for pattern in spam_patterns if pattern in text_lower]
        print(f"   Spam patterns ({len(spam_matches)}):")
        for pattern in spam_matches:
            print(f"     • '{pattern}'")

if __name__ == "__main__":
    test_patterns()