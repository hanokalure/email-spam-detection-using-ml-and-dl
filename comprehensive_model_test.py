#!/usr/bin/env python3
"""
Comprehensive Spam Detection Model Performance Test
Tests all 4 models (BalancedSpamNet, Enhanced Transformer, SVM, CatBoost)
with 40 diverse test cases (20 SPAM + 20 HAM) across multiple categories
"""

import os
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'predictors'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import all predictors
try:
    from predict_balanced_spam_net import BalancedSpamNetPredictor
    from predict_enhanced_transformer import EnhancedTransformerPredictor
    from predict_svm import SVMPredictor
    from predict_catboost import CatBoostPredictor
    print("[+] All predictors imported successfully")
except ImportError as e:
    print(f"[!] Import error: {e}")
    sys.exit(1)


class ComprehensiveModelTester:
    """Comprehensive test suite for all spam detection models"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models = {}
        self.test_cases = self._create_test_cases()
        self.results = {}
        
    def _create_test_cases(self):
        """Create comprehensive test cases across different categories"""
        return {
            # SPAM Categories (20 total)
            "SPAM": {
                "Financial Scams": [
                    "URGENT: Your account has been suspended! Click here to verify your bank details and avoid permanent closure.",
                    "Congratulations! You've won $50,000 in the International Lottery! Send $500 processing fee to claim your prize.",
                    "FINAL NOTICE: Pay outstanding debt of $2,847 immediately or face legal action. Call now!",
                    "Investment opportunity: Double your money in 30 days! Guaranteed returns of 200%. Limited time offer!"
                ],
                "Phishing": [
                    "Security Alert: Suspicious activity detected on your PayPal account. Verify identity: http://paypal-security.fake.com",
                    "Your Microsoft account will be deleted in 24 hours. Update payment info immediately: click.here.now",
                    "Amazon Prime membership expired. Update credit card to continue service: amazon-update-billing.com",
                    "Apple ID locked due to suspicious login. Verify account: appleid-verification-secure.net"
                ],
                "Product Promotion": [
                    "Amazing weight loss pills! Lose 30 pounds in 10 days without diet or exercise! Order now for $19.99!",
                    "Male enhancement pills - Guaranteed results in 3 days! Discreet packaging. 50% off today only!",
                    "Miracle anti-aging cream! Look 20 years younger overnight! Limited stock - order now!",
                    "Revolutionary hair growth formula! Regrow full hair in 2 weeks! FDA approved* (*not really)"
                ],
                "Romance/Dating": [
                    "Hello beautiful, I am a wealthy businessman from Nigeria looking for true love. Can we meet?",
                    "Lonely? Meet hot singles in your area tonight! No strings attached. Click here for instant hookups!",
                    "My name is Maria, I'm 25 and looking for serious relationship. I have $2M inheritance to share.",
                    "SEXY RUSSIAN GIRLS want to chat with YOU! Free registration for limited time. Adults only!"
                ],
                "Health/Medical": [
                    "CURE DIABETES NATURALLY! Secret remedy doctors don't want you to know! 100% natural, no side effects!",
                    "COVID-19 immunity booster! Prevent infection with this one weird trick! Doctors hate this!",
                    "Lose weight fast with this illegal diet pill celebrities use! Banned in 12 countries!",
                    "CANCER CURE DISCOVERED! Big Pharma hiding this $5 treatment that works 100% of the time!"
                ]
            },
            
            # HAM Categories (20 total)  
            "HAM": {
                "Banking Legitimate": [
                    "Account Alert: $500.00 has been debited from your account ending xxxx1234 on 15-Jan-2024 at 14:30. Available balance: $2,847.50. Ref: TXN123456789",
                    "Your account balance is $1,234.56 as of today. Last transaction: $50 withdrawal at ATM on Main Street. Thank you for banking with us.",
                    "Payment of $1,200 has been successfully processed for invoice #INV-2024-001. Transaction ID: PAY789012345. Receipt attached.",
                    "Dear Customer, your monthly statement is now available. Login to view transactions for January 2024. Customer Service: 1-800-BANK"
                ],
                "Business Professional": [
                    "Meeting scheduled for tomorrow 2 PM in Conference Room A. Please bring Q4 financial reports and marketing projections.",
                    "Project deadline extended to March 15th. Team meeting rescheduled to discuss new timeline and resource allocation.",
                    "Your expense report for January has been approved. Reimbursement of $847.32 will be processed in next payroll cycle.",
                    "Client presentation went well. They're interested in our proposal. Next step: contract negotiation meeting Thursday."
                ],
                "Personal Communication": [
                    "Hey! Hope you're doing well. Want to grab coffee this weekend? I have some exciting news to share with you.",
                    "Thanks for dinner last night! The restaurant was amazing. We should definitely go there again soon.",
                    "Can you pick up milk and bread on your way home? Also, don't forget we have dinner with parents tomorrow.",
                    "Happy birthday! Hope you have a wonderful day. Looking forward to celebrating with you this evening."
                ],
                "E-commerce Legitimate": [
                    "Your Amazon order #112-7234567-1234567 has been shipped. Expected delivery: Jan 18th. Track your package here.",
                    "Thank you for your purchase! Your eBay item has been dispatched. Tracking number: 1Z999AA1234567890. Seller: TechStore",
                    "Order confirmation: iPhone 15 Pro - $999. Payment processed successfully. Estimated delivery: 3-5 business days.",
                    "Your return has been processed. Refund of $249.99 for wireless headphones will appear in your account within 5-7 days."
                ],
                "Educational": [
                    "Assignment reminder: Research paper on sustainable energy is due Friday. Office hours available Tuesday 2-4 PM.",
                    "Class canceled tomorrow due to weather conditions. Online lecture will be posted on course portal by 6 PM.",
                    "Congratulations on completing your certification course. Your diploma will be mailed within 2 weeks.",
                    "Final exam schedule posted. Please check your student portal and contact advisor with any scheduling conflicts."
                ]
            }
        }
    
    def load_models(self):
        """Load all available models"""
        model_files = {
            "BalancedSpamNet": "balanced_spam_net_best.pt",
            "Enhanced_Transformer": "enhanced_transformer_99recall.pt", 
            "SVM": "svm_full.pkl",
            "CatBoost": "catboost_tuned.pkl"
        }
        
        for model_name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                try:
                    if model_name == "BalancedSpamNet":
                        self.models[model_name] = BalancedSpamNetPredictor(str(model_path))
                    elif model_name == "Enhanced_Transformer":
                        self.models[model_name] = EnhancedTransformerPredictor(str(model_path))
                    elif model_name == "SVM":
                        self.models[model_name] = SVMPredictor(str(model_path))
                    elif model_name == "CatBoost":
                        self.models[model_name] = CatBoostPredictor(str(model_path))
                    
                    print(f"[+] {model_name} loaded successfully")
                except Exception as e:
                    print(f"[!] Failed to load {model_name}: {e}")
            else:
                print(f"[!] Model file not found: {filename}")
    
    def test_model(self, model_name, model):
        """Test a single model with all test cases"""
        print(f"\n{'='*60}")
        print(f"Testing {model_name}")
        print(f"{'='*60}")
        
        results = {
            "model_name": model_name,
            "total_tests": 0,
            "correct_predictions": 0,
            "spam_tests": 0,
            "spam_correct": 0,
            "ham_tests": 0, 
            "ham_correct": 0,
            "category_results": defaultdict(lambda: {"total": 0, "correct": 0}),
            "predictions": []
        }
        
        # Test all categories
        for spam_type, categories in self.test_cases.items():
            expected_label = spam_type
            
            for category, texts in categories.items():
                print(f"\nTesting {category} ({spam_type})...")
                
                for i, text in enumerate(texts):
                    try:
                        # Make prediction
                        start_time = time.time()
                        prediction_result = model.predict(text)
                        pred_time = time.time() - start_time
                        
                        # Handle different return formats
                        if isinstance(prediction_result, dict):
                            predicted_label = prediction_result.get('prediction', 'HAM')
                            confidence = prediction_result.get('confidence', 0.5)
                        else:
                            predicted_label, confidence = prediction_result
                        
                        # Normalize labels
                        predicted_label = predicted_label.upper()
                        expected_label_norm = expected_label.upper()
                        
                        # Check if prediction is correct
                        is_correct = predicted_label == expected_label_norm
                        
                        # Update counters
                        results["total_tests"] += 1
                        if is_correct:
                            results["correct_predictions"] += 1
                            results["category_results"][category]["correct"] += 1
                        
                        results["category_results"][category]["total"] += 1
                        
                        if expected_label_norm == "SPAM":
                            results["spam_tests"] += 1
                            if is_correct:
                                results["spam_correct"] += 1
                        else:
                            results["ham_tests"] += 1
                            if is_correct:
                                results["ham_correct"] += 1
                        
                        # Store prediction details
                        results["predictions"].append({
                            "category": category,
                            "expected": expected_label_norm,
                            "predicted": predicted_label,
                            "confidence": confidence,
                            "correct": is_correct,
                            "text": text[:100] + "..." if len(text) > 100 else text,
                            "time_ms": pred_time * 1000
                        })
                        
                        # Print result
                        status = "✅" if is_correct else "❌"
                        print(f"  {i+1}. {status} {predicted_label} (conf: {confidence:.3f}) - {text[:80]}...")
                        
                    except Exception as e:
                        print(f"  {i+1}. ❌ ERROR: {e}")
                        results["predictions"].append({
                            "category": category,
                            "expected": expected_label_norm,
                            "predicted": "ERROR",
                            "confidence": 0.0,
                            "correct": False,
                            "text": text[:100] + "..." if len(text) > 100 else text,
                            "error": str(e)
                        })
        
        return results
    
    def analyze_results(self, results):
        """Analyze test results and create summary"""
        model_name = results["model_name"]
        
        # Calculate accuracies
        overall_accuracy = (results["correct_predictions"] / results["total_tests"]) * 100 if results["total_tests"] > 0 else 0
        spam_accuracy = (results["spam_correct"] / results["spam_tests"]) * 100 if results["spam_tests"] > 0 else 0
        ham_accuracy = (results["ham_correct"] / results["ham_tests"]) * 100 if results["ham_tests"] > 0 else 0
        
        # Analyze category performance
        perfect_categories = []
        high_accuracy_categories = []
        low_accuracy_categories = []
        
        for category, stats in results["category_results"].items():
            accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            
            if accuracy == 100:
                perfect_categories.append(category)
            elif accuracy >= 75:
                high_accuracy_categories.append(category)
            elif accuracy < 50:
                low_accuracy_categories.append(category)
        
        # Create summary
        summary = {
            "model_name": model_name,
            "overall_accuracy": overall_accuracy,
            "spam_detection_accuracy": spam_accuracy,
            "ham_detection_accuracy": ham_accuracy,
            "perfect_categories": perfect_categories,
            "high_accuracy_categories": high_accuracy_categories,
            "low_accuracy_categories": low_accuracy_categories,
            "total_tests": results["total_tests"],
            "category_breakdown": {}
        }
        
        # Add category breakdown
        for category, stats in results["category_results"].items():
            accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            summary["category_breakdown"][category] = {
                "accuracy": accuracy,
                "correct": stats["correct"],
                "total": stats["total"]
            }
        
        return summary
    
    def print_summary(self, summary):
        """Print formatted summary like the PDF report"""
        print(f"\n{'='*60}")
        print(f"{summary['model_name']} - Summary")
        print(f"{'='*60}")
        
        # Summary table
        print(f"{'Metric':<30} {'Value'}")
        print(f"{'-'*40}")
        print(f"{'Overall Accuracy':<30} {summary['overall_accuracy']:.1f}%")
        print(f"{'SPAM Detection Accuracy':<30} {summary['spam_detection_accuracy']:.1f}%")
        print(f"{'HAM Detection Accuracy':<30} {summary['ham_detection_accuracy']:.1f}%")
        print(f"{'Perfect Categories':<30} {len(summary['perfect_categories'])}")
        
        # Perfect categories
        print(f"\n{'Perfect Categories (100% Accuracy)':>40}")
        print(f"{'-'*40}")
        if summary['perfect_categories']:
            for cat in summary['perfect_categories']:
                print(f"  • {cat}")
        else:
            print("  None")
        
        # High accuracy categories
        print(f"\n{'High Accuracy Categories (≥75%)':>40}")
        print(f"{'-'*40}")
        if summary['high_accuracy_categories']:
            for cat in summary['high_accuracy_categories']:
                acc = summary['category_breakdown'][cat]['accuracy']
                print(f"  • {cat} ({acc:.1f}%)")
        else:
            print("  None")
        
        # Low accuracy categories
        print(f"\n{'Low Accuracy Categories (<50%)':>40}")
        print(f"{'-'*40}")
        if summary['low_accuracy_categories']:
            for cat in summary['low_accuracy_categories']:
                acc = summary['category_breakdown'][cat]['accuracy']
                print(f"  • {cat} ({acc:.1f}%)")
        else:
            print("  None")
        
        # Category breakdown
        print(f"\n{'Category Performance Breakdown':>40}")
        print(f"{'-'*40}")
        for category, stats in summary['category_breakdown'].items():
            print(f"{category:<25} {stats['correct']:>2}/{stats['total']} ({stats['accuracy']:>5.1f}%)")
    
    def save_detailed_results(self, all_results):
        """Save detailed results to JSON file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_model_test_results_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Detailed results saved to: {filename}")
    
    def run_comprehensive_test(self):
        """Run comprehensive test on all models"""
        print("🚀 Comprehensive Spam Detection Model Performance Test")
        print("=" * 80)
        print(f"Test Overview:")
        print(f"• Models: BalancedSpamNet, Enhanced Transformer, SVM, CatBoost")
        print(f"• SPAM Categories: 5 (4 samples each = 20 total)")
        print(f"• HAM Categories: 5 (4 samples each = 20 total)")
        print(f"• Total Tests per Model: 40")
        print("=" * 80)
        
        self.load_models()
        
        if not self.models:
            print("❌ No models loaded! Please check model files.")
            return
        
        all_results = {
            "test_info": {
                "timestamp": datetime.now().isoformat(),
                "total_models": len(self.models),
                "test_cases_per_model": 40,
                "spam_categories": list(self.test_cases["SPAM"].keys()),
                "ham_categories": list(self.test_cases["HAM"].keys())
            },
            "model_results": {},
            "summaries": {}
        }
        
        # Test each model
        for model_name, model in self.models.items():
            try:
                results = self.test_model(model_name, model)
                summary = self.analyze_results(results)
                
                all_results["model_results"][model_name] = results
                all_results["summaries"][model_name] = summary
                
                self.print_summary(summary)
                
            except Exception as e:
                print(f"❌ Error testing {model_name}: {e}")
        
        # Print comparison
        self.print_model_comparison(all_results["summaries"])
        
        # Save results
        self.save_detailed_results(all_results)
        
        return all_results
    
    def print_model_comparison(self, summaries):
        """Print model comparison table"""
        print(f"\n{'='*80}")
        print(f"{'Model Comparison Summary':^80}")
        print(f"{'='*80}")
        
        print(f"{'Model':<20} {'Overall':<12} {'SPAM Det.':<12} {'HAM Det.':<12} {'Perfect Cat.'}")
        print(f"{'-'*75}")
        
        for model_name, summary in summaries.items():
            print(f"{model_name:<20} "
                  f"{summary['overall_accuracy']:>7.1f}%     "
                  f"{summary['spam_detection_accuracy']:>7.1f}%      "
                  f"{summary['ham_detection_accuracy']:>7.1f}%      "
                  f"{len(summary['perfect_categories']):>6}")


def main():
    """Main function to run comprehensive test"""
    tester = ComprehensiveModelTester()
    results = tester.run_comprehensive_test()
    
    print(f"\n🎯 Test completed! Check the JSON file for detailed results.")
    print(f"🏆 Models tested: {len(results['model_results'])}")


if __name__ == "__main__":
    main()