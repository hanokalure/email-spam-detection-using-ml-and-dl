#!/usr/bin/env python3
"""
Test the Enhanced Transformer with ATM withdrawal and banking notification messages
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from predict_enhanced_transformer import EnhancedTransformerPredictor

def test_atm_messages():
    """Test with legitimate ATM and banking messages that might be misclassified"""
    
    # Check if model exists
    model_path = "models/enhanced_transformer_best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("💡 Make sure you've trained the Enhanced Transformer model first")
        return
    
    # Load predictor
    predictor = EnhancedTransformerPredictor(model_path)
    
    # Test messages - legitimate ATM/banking notifications
    test_messages = [
        # ATM withdrawal notifications
        "Dear Customer, ₹5000 withdrawn using ATM card ending 1234 at ATM on 15-Jan-2024 14:30. Avl Bal: ₹45000. Thank you for banking with us.",
        
        "Alert: $200 cash withdrawal from ATM at Main St Branch on 01/15/2024 2:30 PM. Card ending 5678. Balance: $1800.",
        
        "Notification: ₹3000 withdrawal transaction successful. ATM card ending in 9012 used at ATM located at Park Avenue. Balance after transaction: ₹27000.",
        
        "Dear Valued Customer, withdrawal of $150 completed at ATM on 15th Jan. Card ending with 3456. Available balance: $950. Ref No: ATM123456789.",
        
        "Info: ₹2500 cash withdrawn using ATM card ending 7890. Transaction Date: 15-Jan-24, Time: 14:45. Current Balance: ₹12500.",
        
        # Account balance notifications  
        "Your account balance is ₹75000 as on 15-Jan-2024. Last transaction: ₹5000 debited on 14-Jan-2024. Minimum balance maintained.",
        
        "Account Balance: $2500. Available Balance: $2450. Last Transaction: $50 debit on 01/14/2024. Thank you for banking with us.",
        
        # Transaction notifications
        "Transaction successful. ₹1000 has been debited from your account xxxx1234 on 15-Jan-2024 at 2:30 PM. Balance: ₹49000.",
        
        "Payment of $75 has been processed successfully. Debited from account ending 5678. Transaction ID: TXN123456789. Balance: $425.",
        
        # Banking statements
        "Dear Customer, your mini statement: 1. 14-Jan: ₹2000 CR, 2. 15-Jan: ₹500 DR. Current Balance: ₹47500. Account statement available.",
        
        # Mixed legitimate messages that might be challenging
        "Alert from YourBank: ₹10000 cash withdrawal at ATM. Card ending 1111 on 15-Jan-2024 14:30. Urgent: Check if this transaction was made by you. Bal: ₹40000.",
        
        "URGENT: Large withdrawal detected. $500 withdrawn using ATM card ending 2222 at Downtown ATM on 01/15/2024. If not you, contact customer care immediately. Balance: $1500."
    ]
    
    print("🧪 Testing Enhanced Transformer with ATM/Banking Messages:")
    print("=" * 80)
    print("🎯 Goal: Reduce false positives on legitimate banking notifications")
    print("=" * 80)
    
    for i, text in enumerate(test_messages, 1):
        result = predictor.predict(text, return_details=True)
        icon = "🚨" if result['is_spam'] else "✅"
        confidence_pct = result['confidence'] * 100
        prob_pct = result['probability'] * 100
        
        print(f"\n{i:2d}. {icon} {result['prediction']} ({confidence_pct:.1f}% confidence, {prob_pct:.1f}% spam prob)")
        print(f"     Message: {text}")
        
        # Show features detected
        features = result['preprocessing_features']
        detected_features = [f.replace('has_', '').upper() for f, present in features.items() 
                           if f.startswith('has_') and present]
        if detected_features:
            print(f"     Features: {', '.join(detected_features)}")

if __name__ == "__main__":
    test_atm_messages()