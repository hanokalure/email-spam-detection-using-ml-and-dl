#!/usr/bin/env python3
"""
Evaluate Enhanced Transformer across multiple spam categories and diverse HAM.
This is a lightweight, non-interactive check to estimate spam recall and HAM FP.
"""
import sys
import os
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import predictor from project root
sys.path.insert(0, os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, ROOT)

from predict_enhanced_transformer import EnhancedTransformerPredictor


def build_dataset() -> Tuple[Dict[str, List[str]], List[str]]:
    """Return (spam_categories, ham_samples)."""
    spam = {
        'phishing': [
            "🚨 URGENT: Your Apple ID is locked. Verify now at appleid-security.com to avoid suspension.",
            "Security Alert: Unusual sign-in detected. Confirm your account at secure-paypal-login.net",
            "Microsoft Account Notice: Reset your password immediately at ms-support-reset.com"
        ],
        'financial_scam': [
            "Congratulations! You've won $10,000 cash! Click here to claim your prize now!",
            "Get a personal loan approved instantly with no credit check! Apply today!",
            "Exclusive investment opportunity: double your money in 7 days! Limited time."
        ],
        'work_from_home': [
            "Make $5000/week working from home. No experience required. 100% guaranteed!",
            "Earn ₹50,000 per week from your phone. Join our program today!",
            "Data entry jobs available. Start earning immediately!"
        ],
        'romance': [
            "My dear, I am lonely and looking for true love. Send me money for my visa.",
            "I am your soulmate. I just need help with my ticket. Please wire $300.",
            "I love you. I lost my wallet, can you send me gift cards?"
        ],
        'tech_support': [
            "Your computer is infected with a virus! Call Microsoft support now: 1-800-555-1234",
            "Windows alert: Your license is expired. Renew at windows-support-verify.com",
            "System security warning! Immediate action required."
        ],
        'crypto': [
            "CRYPTO ALERT: Get 5x returns by investing in this new coin!",
            "Bitcoin giveaway: Send 0.1 BTC and receive 1 BTC back! Limited!",
            "New token pre-sale! Guaranteed profits."
        ],
        'lottery': [
            "You have won the International Lottery of $1,000,000. Claim now!",
            "Winner! Your number was selected. Send bank details to receive funds.",
            "Jackpot winner announcement! Provide details to process your prize."
        ],
        'fake_delivery': [
            "Your package is on hold due to unpaid customs fees. Pay here: sh1p-update.co",
            "Delivery attempt failed. Reschedule now at delivery-fix-now.info",
            "Parcel pending. Confirm address and pay fee to release."
        ],
        'job_offer': [
            "We saw your profile. High salary job with minimal work. Apply here now!",
            "Immediate hiring: No interview needed. Limited slots. Apply today!",
            "Part-time jobs available. Earn fast money without skills."
        ],
        'adult_content': [
            "Hot singles in your area want to meet. Click to chat now!",
            "Adult content unlocked. Subscribe now for explicit videos.",
            "18+ only. Exclusive content limited time offer."
        ]
    }

    ham = [
        # General work messages
        "Hi team, attaching the project report. Let's review on Friday at 2 PM.",
        "Please see the meeting notes and the action items for next week.",
        "Can we move our call to tomorrow?",
        # Order confirmations / shipping legit
        "Your Amazon order has shipped. Track your package in your account.",
        "Order #12345 confirmed. Estimated delivery: Oct 10.",
        # Banking legitimate (non-ATM)
        "Payment of $120 received for invoice #456. Thank you.",
        "Your monthly statement is available in your online banking.",
        "Account balance: $2,450. Last transaction: $50 debit on 09/28/2025.",
        # Social / personal
        "Happy birthday! Wishing you a wonderful day!",
        "Let's meet at the cafe at 5pm.",
        # Delivery legit
        "Your package was delivered today. If you didn't receive it, contact support.",
        # Tech newsletter legit
        "Weekly newsletter: New features released in v2.3.0."
    ]

    return spam, ham


def evaluate(predictor: EnhancedTransformerPredictor) -> None:
    spam, ham = build_dataset()

    total_spam = 0
    detected_spam = 0
    category_breakdown = []

    print("\n🧪 Comprehensive spam category evaluation:")
    print("=" * 80)

    # Evaluate spam categories
    for cat, msgs in spam.items():
        cat_total = len(msgs)
        cat_detected = 0
        for m in msgs:
            r = predictor.predict(m)
            if r['is_spam']:
                cat_detected += 1
        category_breakdown.append((cat, cat_detected, cat_total))
        total_spam += cat_total
        detected_spam += cat_detected

    # Print spam summary
    for cat, det, tot in category_breakdown:
        print(f" - {cat:15s}: {det}/{tot} spam detected")

    spam_recall = 0.0 if total_spam == 0 else detected_spam / total_spam
    print(f"\n📈 Spam recall (all categories): {detected_spam}/{total_spam} = {spam_recall*100:.2f}%")

    # Evaluate HAM
    ham_total = len(ham)
    ham_correct = 0
    false_positives = []
    for m in ham:
        r = predictor.predict(m)
        if not r['is_spam']:
            ham_correct += 1
        else:
            false_positives.append(m)

    ham_accuracy = 0.0 if ham_total == 0 else ham_correct / ham_total
    print(f"✅ HAM accuracy: {ham_correct}/{ham_total} = {ham_accuracy*100:.2f}%")

    if false_positives:
        print("\n⚠️ HAM false positives:")
        for i, m in enumerate(false_positives, 1):
            print(f"  {i}. {m}")

    # Exit code style summary for CI-like checks
    print("\n==== SUMMARY ====")
    print(f"Spam recall: {spam_recall*100:.2f}% (target: 100%)")
    print(f"HAM accuracy: {ham_accuracy*100:.2f}% (higher is better)")


def main():
    model_path = os.path.join(ROOT, 'models', 'enhanced_transformer_best.pt')
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        print("💡 Train or provide the model before evaluation.")
        return 1

    predictor = EnhancedTransformerPredictor(model_path)
    evaluate(predictor)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
