#!/usr/bin/env python3
"""
Comprehensive Spam Detection Model Testing Suite
Tests all 3 models (Enhanced Transformer, SVM, CatBoost) across diverse spam/ham categories
Generates detailed analysis of which model performs best for each category
"""

import os
import sys
import json
import time
from collections import defaultdict
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import all predictors
from predict_enhanced_transformer import EnhancedTransformerPredictor
from predict_svm import SVMPredictor
from predict_catboost import CatBoostPredictor

class ComprehensiveSpamTest:
    """Comprehensive testing suite for spam detection models"""
    
    def __init__(self):
        self.models = {}
        self.test_results = {}
        self.load_models()
        
        # Test categories with 20 examples each
        self.spam_categories = {
            "financial_scams": [
                "Congratulations! You've won $50,000 in the International Lottery! Click here to claim now!",
                "URGENT: Your bank account will be closed unless you verify immediately. Click: secure-bank-verify.com",
                "Make $5000 working from home! No experience required. Start today!",
                "Investment opportunity of a lifetime! Double your money in 30 days guaranteed!",
                "You have inherited $2.5 million from a distant relative. Reply with your details.",
                "FOREX trading secret that banks don't want you to know! Make millions!",
                "Your credit card has been charged $499. Click here to dispute if unauthorized.",
                "Free money from government grants! No repayment required. Apply now!",
                "Bitcoin investment alert! Buy now before price skyrockets to $100k!",
                "Debt consolidation offer - eliminate all debts instantly. Call now!",
                "Pre-approved for $25,000 loan with no credit check required.",
                "Tax refund of $3,847 is waiting for you. Click here to claim.",
                "Your PayPal account shows suspicious activity. Verify now to avoid suspension.",
                "Earn $200 per day answering simple surveys online. Join millions!",
                "Stock tip: ACME Corp will increase 500% next week. Buy now!",
                "Your stimulus check of $1,200 is ready. Enter SSN to claim.",
                "Cryptocurrency mining opportunity - earn $1000 daily guaranteed!",
                "Your retirement account needs immediate attention. Click to secure.",
                "Win $10,000 cash prize! You're our lucky winner today!",
                "Exclusive investment club invitation - minimum return 300% annually."
            ],
            "phishing": [
                "Your Amazon account has been compromised. Click here to secure: amazon-security.net",
                "Netflix subscription expires today. Update payment: netflix-billing.com",
                "Apple ID locked due to suspicious activity. Verify: apple-support.net",
                "Your Google account will be deleted in 24 hours. Confirm: google-verify.com",
                "Microsoft security alert: Login detected from unknown device. Verify now.",
                "Facebook account suspended. Click to appeal: facebook-appeals.net",
                "Instagram verification required. Confirm identity: instagram-verify.com",
                "WhatsApp will be deactivated unless verified within 2 hours.",
                "Twitter account compromise detected. Secure now: twitter-security.net",
                "LinkedIn premium subscription auto-renewed. Cancel: linkedin-billing.com",
                "Dropbox storage full. Upgrade now or lose files: dropbox-upgrade.net",
                "Spotify account accessed from new device. Authorize: spotify-auth.com",
                "Steam account trade hold. Verify ownership: steam-support.net",
                "eBay seller protection expired. Renew immediately: ebay-protection.com",
                "Outlook email quota exceeded. Expand storage: outlook-storage.net",
                "iCloud backup failed. Fix now: icloud-fix.com",
                "GitHub security vulnerability detected in your account.",
                "Discord server ownership transfer requires verification.",
                "TikTok copyright strike received. Appeal now: tiktok-appeals.net",
                "Zoom meeting security update required. Install: zoom-update.com"
            ],
            "product_promotion": [
                "Amazing weight loss pills! Lose 30 pounds in 30 days guaranteed!",
                "Male enhancement - increase size by 3 inches in 2 weeks!",
                "Miracle skin cream removes wrinkles overnight! Before/after photos!",
                "Hair growth formula - regrow full head of hair in 60 days!",
                "Revolutionary diet pill - eat anything and still lose weight!",
                "Teeth whitening strips - Hollywood white smile in 7 days!",
                "Memory enhancement supplement - boost IQ by 40 points!",
                "Anti-aging breakthrough - look 20 years younger in weeks!",
                "Muscle building supplement - gain 20 lbs muscle in 30 days!",
                "Vision improvement drops - throw away glasses forever!",
                "Cellulite removal cream - smooth skin in 14 days guaranteed!",
                "Sleep aid that cures insomnia permanently after one use!",
                "Detox tea that melts belly fat while you sleep!",
                "Energy drink that replaces 8 hours of sleep with 30 minutes!",
                "Pain relief patch - eliminates arthritis pain instantly!",
                "Blood pressure medicine - normalize BP without prescription!",
                "Diabetes cure discovered - no more insulin needed!",
                "Cancer fighting superfood - 99% cure rate in studies!",
                "Brain pill increases intelligence by 200% overnight!",
                "Fountain of youth serum - reverse aging completely!"
            ],
            "romance_dating": [
                "Beautiful Russian woman seeks American husband. View photos: dating-site.com",
                "Lonely widow with $2M inheritance looking for companion.",
                "Hot singles in your area want to meet tonight! Join free!",
                "Find your soulmate today - 100% compatibility guaranteed!",
                "Ashley Madison affair site - discreet encounters available now.",
                "Local women want casual hookups - no strings attached!",
                "Sugar daddy seeking arrangement with college student.",
                "International dating - meet exotic women from Asia!",
                "Mature dating site for seniors - find love after 50!",
                "LGBT dating community - find your perfect match today!",
                "Military singles seeking pen pals and romance overseas.",
                "Christian singles congregation - faith-based relationships.",
                "Professional matchmaking service - millionaire clients only.",
                "Speed dating event tonight - 20 dates in 2 hours!",
                "Online dating profile optimization - guarantee more matches!",
                "Relationship counseling via chat - save your marriage today!",
                "Dating app premium features unlocked for free this week!",
                "Video chat with beautiful models - private sessions available!",
                "Mail order bride service - traditional family values guaranteed!",
                "Celebrity dating app - match with famous personalities!"
            ],
            "health_medical": [
                "Doctors hate this simple trick to cure diabetes naturally!",
                "FDA banned this weight loss secret - but we still have it!",
                "Alternative cancer treatment suppressed by big pharma!",
                "Cure any disease with this one weird frequency therapy!",
                "Oxygen therapy reverses aging and cures everything naturally!",
                "Homeopathic remedy eliminates need for all prescription drugs!",
                "Magnetic therapy bracelet heals arthritis and pain instantly!",
                "Essential oils cure protocol for cancer, diabetes, and heart disease!",
                "Crystal healing session removes toxins and negative energy!",
                "Alkaline water machine prevents all diseases and aging!",
                "Colon cleanse removes 30 pounds of toxic waste overnight!",
                "Raw food diet cures autism, ADHD, and mental illness!",
                "Meditation technique eliminates need for sleep and food!",
                "Chelation therapy removes heavy metals and cures autism!",
                "Pranic healing fixes broken bones without surgery!",
                "Herbal supplement regrows limbs and organs naturally!",
                "Sound therapy frequency heals DNA and reverses mutations!",
                "Acupuncture point injection cures paralysis in one session!",
                "Vitamin C megadose prevents and cures all viral infections!",
                "Energy healing transmission cures depression and anxiety instantly!"
            ]
        }
        
        self.ham_categories = {
            "banking_legitimate": [
                "Your account balance is $1,247.83. Available balance: $1,200.00. Last transaction: ATM withdrawal $50.00 on 01-Oct-2025.",
                "Payment of $89.99 has been processed for Netflix subscription. Account ending 4567. Transaction ID: TXN12345.",
                "Your monthly statement is now available. Login to view transactions for September 2025.",
                "ATM withdrawal of $200 using card ending 1234 at Main Street ATM on 01-Oct-2025 at 14:30. Available balance: $850.00.",
                "Direct deposit of $2,500.00 from ACME CORP has been credited to your account. Balance: $3,247.50.",
                "Auto-pay setup confirmed for electric bill. $125.50 will be debited on 15th each month.",
                "Your credit card payment of $450.00 has been received. Thank you. Current balance: $1,200.00.",
                "Debit card transaction declined at Amazon.com for $299.99 due to insufficient funds.",
                "Foreign transaction fee of $3.50 applied for purchase in EUR. Contact us if traveling frequently.",
                "Minimum balance maintained. No service charges for this month. Thank you for banking with us.",
                "Wire transfer of $1,000 to John Smith completed. Reference number: WIRE789. Fee: $25.00.",
                "Your loan payment of $680.00 is due on October 5th. Setup autopay to avoid late fees.",
                "Certificate of deposit matured. $10,500 has been transferred to your savings account.",
                "Overdraft protection activated. $100 transferred from savings to checking to cover transaction.",
                "Your new debit card will arrive in 7-10 business days. Activate upon receipt.",
                "Interest earned this quarter: $12.45. Annual percentage yield: 0.45%.",
                "International wire transfer received: $2,000 from David Johnson. Processing fee: $15.",
                "Your account has been upgraded to premium. Enjoy enhanced features and priority support.",
                "Mobile deposit of $567.89 processed. Funds available immediately in your account.",
                "Thank you for 10 years of banking with us. Special rewards program enrollment available."
            ],
            "business_professional": [
                "Board meeting scheduled for Thursday 3 PM in Conference Room A. Please review quarterly reports.",
                "Project deadline extended to Friday due to client feedback. Team meeting tomorrow at 10 AM.",
                "Please submit your timesheet by EOD Monday. Late submissions will affect payroll processing.",
                "Quarterly performance reviews begin next week. Schedule your 1-on-1 with your manager.",
                "Company-wide training on new software Tuesday 2-4 PM. Attendance mandatory for all staff.",
                "Budget proposal for Q4 marketing campaign approved. Implementation starts immediately.",
                "New employee orientation Monday 9 AM. Please bring required documents and ID.",
                "IT maintenance scheduled Saturday 6-8 AM. Systems will be unavailable during this time.",
                "Congratulations to Sarah Johnson on her promotion to Senior Manager effective immediately.",
                "Office closure announced for December 24th and 31st. Plan projects accordingly.",
                "Expense report submission deadline is Friday. Use updated forms available on intranet.",
                "Client presentation went well. They've approved the proposal and want to move forward.",
                "Health insurance enrollment period ends October 15th. Review options and make selections.",
                "Conference call with investors rescheduled to Wednesday 2 PM. Dial-in details attached.",
                "Annual company retreat planning underway. Destination survey closing Friday at noon.",
                "Sales targets exceeded by 15% this quarter. Team celebration lunch Friday 12:30 PM.",
                "New security protocols effective immediately. Badge access required for all areas.",
                "Professional development budget available. Submit training requests to HR by month end.",
                "Vendor contract renewal due next month. Legal review scheduled for Tuesday morning.",
                "Office supplies inventory low. Submit requests through procurement portal by Thursday."
            ],
            "personal_communication": [
                "Hey Sarah! Hope you're doing well. Want to grab coffee this weekend and catch up?",
                "Happy birthday! Hope you have a wonderful day. Looking forward to the party tonight!",
                "Thanks for helping me move yesterday. Couldn't have done it without you. Dinner's on me!",
                "Can you pick up milk and bread on your way home? We're out and stores close early today.",
                "Great meeting you at the conference. Here's my card. Let's stay in touch professionally.",
                "Reminder: Doctor's appointment tomorrow at 2 PM. Don't forget to bring insurance card.",
                "Flight delayed by 2 hours. Will arrive at 8 PM instead of 6. Can you adjust pickup time?",
                "Kids' soccer game moved to Sunday due to rain. New time: 10 AM at Central Park field.",
                "Thank you note for the lovely dinner party. Everything was delicious and we had great fun!",
                "Carpooling to work next week - I can drive Monday, Wednesday, Friday if you want Tuesday, Thursday.",
                "Family reunion planning meeting Sunday 3 PM at Mom's house. Please bring photo albums.",
                "Book club selection for next month: 'The Seven Husbands of Evelyn Hugo'. Meeting date: 20th.",
                "Garage sale Saturday 8 AM - 4 PM. Furniture, clothes, books, electronics. Great deals!",
                "Wedding invitation attached. RSVP by March 15th. So excited to celebrate with you!",
                "Recipe you asked for attached. Secret ingredient is a pinch of cardamom. Enjoy cooking!",
                "Vacation photos uploaded to shared album. Password is 'beach2025'. Had amazing time!",
                "Pet sitter needed July 10-17. $50/day for feeding cat and watering plants. Are you available?",
                "Study group forming for chemistry exam. Meeting library Wednesday 6 PM. Join us!",
                "Marathon training update: completed 15-mile run today. Legs tired but feeling strong!",
                "Anniversary dinner reservations confirmed for Saturday 7 PM at Giovanni's. Dress code: formal."
            ],
            "educational_institutional": [
                "Assignment submission deadline extended to Friday 11:59 PM due to technical issues with portal.",
                "Guest lecture by Nobel Prize winner scheduled for next Tuesday. Registration required by Monday.",
                "Mid-term exam schedule posted on course website. Study guides available in library reserve section.",
                "Research paper topics must be approved by advisor before November 1st. Schedule appointments now.",
                "Campus career fair October 15-16. Over 100 employers participating. Dress professionally.",
                "Student ID cards can be renewed at registrar office. Hours: Monday-Friday 8 AM - 5 PM.",
                "Library hours extended during finals week. 24/7 access with valid student ID starting December 1st.",
                "Scholarship application deadline approaching. Need-based and merit awards available. Apply online.",
                "Course registration for spring semester opens October 20th for seniors, 22nd for underclassmen.",
                "Laboratory safety training mandatory for all chemistry students. Sessions available daily this week.",
                "Graduation ceremony tickets limited to 4 per student. Distribution begins November 15th.",
                "Study abroad information session Thursday 4 PM in Student Union. Application deadline: January 15th.",
                "Thesis defense scheduled for November 30th at 2 PM. Committee members and room confirmed.",
                "Drop/add period ends Friday. Changes after this date require dean's approval and documentation.",
                "Student health services offering flu shots Monday-Wednesday. No appointment necessary, just walk in.",
                "Parking permits for spring semester on sale now. Early bird discount ends October 31st.",
                "Academic probation meeting scheduled with advisor. Please bring transcript and degree audit.",
                "Teaching assistant positions available for spring. Applications due November 1st. Competitive stipend offered.",
                "Campus dining plan changes for next semester. New options include kosher and vegan meal plans.",
                "Final grades will be posted December 20th. Disputes must be filed within 30 days of posting."
            ],
            "ecommerce_legitimate": [
                "Your Amazon order #123-456789 has shipped. Tracking: 1Z999AA1234567890. Delivery expected Friday.",
                "eBay auction ending in 2 hours. Current bid: $25.50. You are currently the highest bidder.",
                "Etsy shop sale: 20% off handmade jewelry through Sunday. Use code FALL20 at checkout.",
                "Walmart grocery pickup ready. Order #WM987654321. Pickup window: 4-5 PM today at store entrance.",
                "Best Buy Geek Squad appointment confirmed for Tuesday 2 PM. Bring laptop and power cord.",
                "Target order modification successful. Upgraded to 2-day shipping for $5.99. New delivery: Thursday.",
                "PayPal payment to Johnson Photography processed successfully. Amount: $250.00. Invoice #INV-001.",
                "Shopify store analytics: 15 orders this week totaling $487.50. Top selling item: blue ceramic mug.",
                "Costco membership expires next month. Renew now to avoid service interruption. Executive upgrade available.",
                "Home Depot order ready for pickup. Lumber and hardware in aisle 1. Bring confirmation email.",
                "Wayfair furniture delivery scheduled Thursday 10 AM - 2 PM. Please ensure someone is home to receive.",
                "Nordstrom return processed. $89.99 refund will appear on credit card within 3-5 business days.",
                "REI dividend notice: $47.85 member dividend available. Use by December 31st or forfeit unused amount.",
                "Zara online order packed and shipped. Express delivery upgrade applied. Arrives tomorrow by noon.",
                "Apple Store pickup ready. iPhone case and screen protector. Bring ID and order confirmation.",
                "StockX authentication complete. Sneakers verified authentic and shipped to buyer. Payment released.",
                "Sephora Beauty Insider points balance: 1,247 points = $62.35 value. Redeem online or in-store.",
                "Williams Sonoma wedding registry updated. 3 items purchased by guests. Thank you cards recommended.",
                "Lululemon size exchange approved. Return original item and new size will ship within 24 hours.",
                "Chewy auto-ship order dispatched. Dog food and treats arriving Tuesday. Modify schedule anytime online."
            ]
        }
    
    def load_models(self):
        """Load all available models"""
        model_paths = {
            'Enhanced_Transformer': 'models/enhanced_transformer_99recall.pt',
            'SVM': 'models/svm_full.pkl',
            'CatBoost': 'models/catboost_tuned.pkl'
        }
        
        print("🔄 Loading models...")
        
        for model_name, model_path in model_paths.items():
            if os.path.exists(model_path):
                try:
                    if model_name == 'Enhanced_Transformer':
                        self.models[model_name] = EnhancedTransformerPredictor(model_path)
                    elif model_name == 'SVM':
                        self.models[model_name] = SVMPredictor(model_path)
                    elif model_name == 'CatBoost':
                        self.models[model_name] = CatBoostPredictor(model_path)
                    print(f"✅ {model_name} loaded successfully")
                except Exception as e:
                    print(f"❌ Failed to load {model_name}: {e}")
            else:
                print(f"❌ Model file not found: {model_path}")
    
    def run_comprehensive_test(self):
        """Run comprehensive test across all categories and models"""
        print(f"\n🧪 Starting Comprehensive Spam Detection Test")
        print("=" * 70)
        
        all_results = {}
        category_stats = {}
        
        # Test spam categories
        for category, examples in self.spam_categories.items():
            print(f"\n📧 Testing SPAM category: {category}")
            results = self._test_category(examples, expected_label='SPAM', category_name=category)
            all_results[f"SPAM_{category}"] = results
            category_stats[f"SPAM_{category}"] = self._calculate_category_stats(results, 'SPAM')
        
        # Test ham categories  
        for category, examples in self.ham_categories.items():
            print(f"\n📧 Testing HAM category: {category}")
            results = self._test_category(examples, expected_label='HAM', category_name=category)
            all_results[f"HAM_{category}"] = results
            category_stats[f"HAM_{category}"] = self._calculate_category_stats(results, 'HAM')
        
        # Calculate overall stats
        overall_stats = self._calculate_overall_stats(category_stats)
        
        # Generate reports
        self._save_detailed_results(all_results, category_stats, overall_stats)
        self._print_summary_report(category_stats, overall_stats)
        
        return all_results, category_stats, overall_stats
    
    def _test_category(self, examples: List[str], expected_label: str, category_name: str) -> Dict:
        """Test a single category across all models"""
        results = {model_name: [] for model_name in self.models.keys()}
        
        for i, text in enumerate(examples, 1):
            print(f"  Testing example {i:2d}/20...", end="")
            
            for model_name, model in self.models.items():
                try:
                    start_time = time.time()
                    prediction = model.predict(text)
                    end_time = time.time()
                    
                    results[model_name].append({
                        'text': text,
                        'prediction': prediction['prediction'],
                        'confidence': float(prediction['confidence']),
                        'probability': float(prediction['probability']),
                        'is_spam': bool(prediction['is_spam']),
                        'correct': bool(prediction['prediction'] == expected_label),
                        'time_ms': float((end_time - start_time) * 1000),
                        'expected': expected_label
                    })
                except Exception as e:
                    print(f"❌ Error with {model_name}: {e}")
                    results[model_name].append({
                        'text': text,
                        'prediction': 'ERROR',
                        'confidence': float(0.0),
                        'probability': float(0.0),
                        'is_spam': bool(False),
                        'correct': bool(False),
                        'time_ms': float(0.0),
                        'expected': expected_label,
                        'error': str(e)
                    })
            
            print(" ✅")
        
        return results
    
    def _calculate_category_stats(self, results: Dict, expected_label: str) -> Dict:
        """Calculate statistics for a category"""
        stats = {}
        
        for model_name, predictions in results.items():
            correct_predictions = sum(1 for p in predictions if p['correct'])
            total_predictions = len(predictions)
            accuracy = correct_predictions / total_predictions * 100
            
            avg_confidence = sum(p['confidence'] for p in predictions) / total_predictions
            avg_time = sum(p['time_ms'] for p in predictions) / total_predictions
            
            # Calculate precision, recall for this category
            if expected_label == 'SPAM':
                true_positives = sum(1 for p in predictions if p['prediction'] == 'SPAM' and p['correct'])
                false_positives = sum(1 for p in predictions if p['prediction'] == 'SPAM' and not p['correct'])
                false_negatives = sum(1 for p in predictions if p['prediction'] == 'HAM' and not p['correct'])
                
                precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
                recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            else:  # HAM
                true_negatives = sum(1 for p in predictions if p['prediction'] == 'HAM' and p['correct'])
                false_negatives = sum(1 for p in predictions if p['prediction'] == 'HAM' and not p['correct'])
                false_positives = sum(1 for p in predictions if p['prediction'] == 'SPAM' and not p['correct'])
                
                precision = true_negatives / (true_negatives + false_negatives) if (true_negatives + false_negatives) > 0 else 0
                recall = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
            
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            stats[model_name] = {
                'accuracy': accuracy,
                'precision': precision * 100,
                'recall': recall * 100,
                'f1_score': f1_score * 100,
                'avg_confidence': avg_confidence,
                'avg_time_ms': avg_time,
                'correct_predictions': correct_predictions,
                'total_predictions': total_predictions
            }
        
        return stats
    
    def _calculate_overall_stats(self, category_stats: Dict) -> Dict:
        """Calculate overall statistics across all categories"""
        overall = {model_name: {
            'total_correct': 0,
            'total_predictions': 0,
            'avg_time_ms': 0,
            'categories_won': 0,
            'spam_accuracy': 0,
            'ham_accuracy': 0
        } for model_name in self.models.keys()}
        
        # Aggregate stats
        for category_name, model_stats in category_stats.items():
            for model_name, stats in model_stats.items():
                overall[model_name]['total_correct'] += stats['correct_predictions']
                overall[model_name]['total_predictions'] += stats['total_predictions']
                overall[model_name]['avg_time_ms'] += stats['avg_time_ms']
        
        # Calculate final metrics
        num_categories = len(category_stats)
        for model_name in overall.keys():
            total_correct = overall[model_name]['total_correct']
            total_predictions = overall[model_name]['total_predictions']
            overall[model_name]['overall_accuracy'] = (total_correct / total_predictions * 100) if total_predictions > 0 else 0
            overall[model_name]['avg_time_ms'] = overall[model_name]['avg_time_ms'] / num_categories
            
            # Count categories where this model performed best
            for category_name, model_stats in category_stats.items():
                best_model = max(model_stats.keys(), key=lambda m: model_stats[m]['accuracy'])
                if best_model == model_name:
                    overall[model_name]['categories_won'] += 1
            
            # Calculate spam vs ham accuracy
            spam_stats = [stats for cat_name, stats in category_stats.items() if cat_name.startswith('SPAM_')]
            ham_stats = [stats for cat_name, stats in category_stats.items() if cat_name.startswith('HAM_')]
            
            if spam_stats:
                spam_accuracy = sum(stats[model_name]['accuracy'] for stats in spam_stats) / len(spam_stats)
                overall[model_name]['spam_accuracy'] = spam_accuracy
            
            if ham_stats:
                ham_accuracy = sum(stats[model_name]['accuracy'] for stats in ham_stats) / len(ham_stats)
                overall[model_name]['ham_accuracy'] = ham_accuracy
        
        return overall
    
    def _save_detailed_results(self, all_results: Dict, category_stats: Dict, overall_stats: Dict):
        """Save detailed results to JSON file"""
        output_data = {
            'test_metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'models_tested': list(self.models.keys()),
                'total_categories': len(category_stats),
                'examples_per_category': 20
            },
            'detailed_results': all_results,
            'category_statistics': category_stats,
            'overall_statistics': overall_stats
        }
        
        output_file = 'comprehensive_spam_test_results.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📊 Detailed results saved to: {output_file}")
    
    def _print_summary_report(self, category_stats: Dict, overall_stats: Dict):
        """Print comprehensive summary report"""
        print("\n" + "=" * 80)
        print("📈 COMPREHENSIVE SPAM DETECTION MODEL ANALYSIS")
        print("=" * 80)
        
        # Overall performance
        print("\n🏆 OVERALL PERFORMANCE:")
        print("-" * 50)
        for model_name, stats in overall_stats.items():
            print(f"{model_name:20} | Accuracy: {stats['overall_accuracy']:6.2f}% | "
                  f"Speed: {stats['avg_time_ms']:6.1f}ms | "
                  f"Categories Won: {stats['categories_won']:2d}")
        
        # Best model overall
        best_overall = max(overall_stats.keys(), key=lambda m: overall_stats[m]['overall_accuracy'])
        fastest_model = min(overall_stats.keys(), key=lambda m: overall_stats[m]['avg_time_ms'])
        most_versatile = max(overall_stats.keys(), key=lambda m: overall_stats[m]['categories_won'])
        
        print(f"\n🥇 CHAMPIONS:")
        print(f"Best Overall Accuracy: {best_overall} ({overall_stats[best_overall]['overall_accuracy']:.2f}%)")
        print(f"Fastest Model: {fastest_model} ({overall_stats[fastest_model]['avg_time_ms']:.1f}ms)")
        print(f"Most Versatile: {most_versatile} ({overall_stats[most_versatile]['categories_won']} categories won)")
        
        # Category-wise performance
        print(f"\n📋 CATEGORY-WISE ANALYSIS:")
        print("-" * 80)
        print(f"{'Category':<25} | {'Best Model':<20} | {'Accuracy':<8} | {'Runner-up'}")
        print("-" * 80)
        
        for category_name, model_stats in category_stats.items():
            # Sort models by accuracy for this category
            sorted_models = sorted(model_stats.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            best_model, best_stats = sorted_models[0]
            runner_up, runner_stats = sorted_models[1] if len(sorted_models) > 1 else (None, None)
            
            category_display = category_name.replace('SPAM_', '📧S:').replace('HAM_', '📬H:')[:24]
            runner_up_text = f"{runner_up} ({runner_stats['accuracy']:.1f}%)" if runner_up else "N/A"
            
            print(f"{category_display:<25} | {best_model:<20} | {best_stats['accuracy']:6.2f}% | {runner_up_text}")
        
        # Spam vs Ham specialization
        print(f"\n🎯 SPAM vs HAM SPECIALIZATION:")
        print("-" * 50)
        for model_name, stats in overall_stats.items():
            print(f"{model_name:20} | SPAM: {stats['spam_accuracy']:6.2f}% | HAM: {stats['ham_accuracy']:6.2f}%")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print("-" * 30)
        print(f"• For maximum accuracy: Use {best_overall}")
        print(f"• For speed-critical applications: Use {fastest_model}")
        print(f"• For diverse content types: Use {most_versatile}")
        
        # Performance insights
        spam_specialist = max(overall_stats.keys(), key=lambda m: overall_stats[m]['spam_accuracy'])
        ham_specialist = max(overall_stats.keys(), key=lambda m: overall_stats[m]['ham_accuracy'])
        
        print(f"• For spam detection: {spam_specialist} ({overall_stats[spam_specialist]['spam_accuracy']:.1f}% accuracy)")
        print(f"• For ham detection: {ham_specialist} ({overall_stats[ham_specialist]['ham_accuracy']:.1f}% accuracy)")
        
        print("\n" + "=" * 80)


def main():
    """Run the comprehensive test suite"""
    print("🚀 Comprehensive Spam Detection Model Testing Suite")
    print("Testing Enhanced Transformer, SVM, and CatBoost models")
    print("Across 10 categories with 20 examples each (200 total tests per model)")
    
    tester = ComprehensiveSpamTest()
    
    if not tester.models:
        print("❌ No models loaded successfully. Please check model files.")
        return
    
    print(f"\n✅ Models loaded: {list(tester.models.keys())}")
    
    try:
        all_results, category_stats, overall_stats = tester.run_comprehensive_test()
        print("\n🎉 Comprehensive testing completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()