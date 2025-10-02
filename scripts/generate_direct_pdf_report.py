#!/usr/bin/env python3
"""
Direct PDF Report Generator - Runs tests and creates PDF without JSON dependency
"""

import os
import sys
import time
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

def install_reportlab():
    """Install ReportLab if not available"""
    print("📦 Installing ReportLab for PDF generation...")
    import subprocess
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
        print("✅ ReportLab installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install ReportLab")
        return False

class DirectPDFReportGenerator:
    """Direct PDF report generation from live model testing"""
    
    def __init__(self):
        self.models = {}
        self.load_models()
        
        # Smaller test sets for direct generation
        self.spam_examples = {
            "Financial Scams": [
                "Congratulations! You've won $50,000 in the International Lottery!",
                "URGENT: Your bank account will be closed unless you verify immediately.",
                "Make $5000 working from home! No experience required. Start today!",
                "Investment opportunity of a lifetime! Double your money guaranteed!",
                "Free money from government grants! No repayment required."
            ],
            "Phishing": [
                "Your Amazon account has been compromised. Click here to secure.",
                "Netflix subscription expires today. Update payment immediately.",
                "Apple ID locked due to suspicious activity. Verify now.",
                "Your Google account will be deleted in 24 hours. Confirm now.",
                "PayPal account shows suspicious activity. Verify to avoid suspension."
            ],
            "Product Promotion": [
                "Amazing weight loss pills! Lose 30 pounds in 30 days guaranteed!",
                "Male enhancement - increase size by 3 inches in 2 weeks!",
                "Miracle skin cream removes wrinkles overnight!",
                "Revolutionary diet pill - eat anything and still lose weight!",
                "Teeth whitening strips - Hollywood white smile in 7 days!"
            ],
            "Romance/Dating": [
                "Beautiful Russian woman seeks American husband. View photos now.",
                "Lonely widow with $2M inheritance looking for companion.",
                "Hot singles in your area want to meet tonight! Join free!",
                "Find your soulmate today - 100% compatibility guaranteed!",
                "Sugar daddy seeking arrangement with college student."
            ],
            "Health/Medical": [
                "Doctors hate this simple trick to cure diabetes naturally!",
                "FDA banned this weight loss secret - but we still have it!",
                "Alternative cancer treatment suppressed by big pharma!",
                "Cure any disease with this one weird frequency therapy!",
                "Oxygen therapy reverses aging and cures everything naturally!"
            ]
        }
        
        self.ham_examples = {
            "Banking Legitimate": [
                "Your account balance is $1,247.83. Available balance: $1,200.00.",
                "Payment of $89.99 has been processed for Netflix subscription.",
                "ATM withdrawal of $200 using card ending 1234 completed.",
                "Direct deposit of $2,500.00 from ACME CORP has been credited.",
                "Auto-pay setup confirmed for electric bill. $125.50 will be debited."
            ],
            "Business Professional": [
                "Board meeting scheduled for Thursday 3 PM in Conference Room A.",
                "Project deadline extended to Friday due to client feedback.",
                "Please submit your timesheet by EOD Monday.",
                "Quarterly performance reviews begin next week.",
                "Company-wide training on new software Tuesday 2-4 PM."
            ],
            "Personal Communication": [
                "Hey Sarah! Hope you're doing well. Want to grab coffee this weekend?",
                "Happy birthday! Hope you have a wonderful day.",
                "Thanks for helping me move yesterday. Dinner's on me!",
                "Can you pick up milk and bread on your way home?",
                "Flight delayed by 2 hours. Will arrive at 8 PM instead of 6."
            ],
            "E-commerce Legitimate": [
                "Your Amazon order #123-456789 has shipped. Delivery expected Friday.",
                "eBay auction ending in 2 hours. You are currently the highest bidder.",
                "PayPal payment processed successfully. Amount: $250.00.",
                "Target order modification successful. New delivery: Thursday.",
                "Apple Store pickup ready. iPhone case and screen protector."
            ],
            "Educational": [
                "Assignment submission deadline extended to Friday 11:59 PM.",
                "Guest lecture by Nobel Prize winner scheduled for next Tuesday.",
                "Mid-term exam schedule posted on course website.",
                "Campus career fair October 15-16. Over 100 employers participating.",
                "Library hours extended during finals week."
            ]
        }
    
    def load_models(self):
        """Load all available models"""
        print("🔄 Loading models...")
        
        try:
            from predict_enhanced_transformer import EnhancedTransformerPredictor
            self.models['Enhanced Transformer'] = EnhancedTransformerPredictor('models/enhanced_transformer_99recall.pt')
            print("✅ Enhanced Transformer loaded")
        except Exception as e:
            print(f"❌ Enhanced Transformer failed: {e}")
        
        try:
            from predict_svm import SVMPredictor
            self.models['SVM'] = SVMPredictor('models/svm_full.pkl')
            print("✅ SVM loaded")
        except Exception as e:
            print(f"❌ SVM failed: {e}")
        
        try:
            from predict_catboost import CatBoostPredictor
            self.models['CatBoost'] = CatBoostPredictor('models/catboost_tuned.pkl')
            print("✅ CatBoost loaded")
        except Exception as e:
            print(f"❌ CatBoost failed: {e}")
    
    def test_categories(self):
        """Test all models across categories"""
        results = {}
        
        print("\n🧪 Testing models across categories...")
        
        # Test spam categories
        for category, examples in self.spam_examples.items():
            print(f"  📧 Testing SPAM: {category}")
            category_results = {}
            
            for model_name, model in self.models.items():
                correct = 0
                total = len(examples)
                confidences = []
                
                for text in examples:
                    try:
                        result = model.predict(text)
                        if result['prediction'] == 'SPAM':
                            correct += 1
                        confidences.append(result['confidence'])
                    except:
                        pass
                
                accuracy = (correct / total) * 100 if total > 0 else 0
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                category_results[model_name] = {
                    'accuracy': accuracy,
                    'avg_confidence': avg_confidence,
                    'correct': correct,
                    'total': total
                }
            
            results[f'SPAM_{category}'] = category_results
        
        # Test ham categories
        for category, examples in self.ham_examples.items():
            print(f"  📬 Testing HAM: {category}")
            category_results = {}
            
            for model_name, model in self.models.items():
                correct = 0
                total = len(examples)
                confidences = []
                
                for text in examples:
                    try:
                        result = model.predict(text)
                        if result['prediction'] == 'HAM':
                            correct += 1
                        confidences.append(result['confidence'])
                    except:
                        pass
                
                accuracy = (correct / total) * 100 if total > 0 else 0
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                
                category_results[model_name] = {
                    'accuracy': accuracy,
                    'avg_confidence': avg_confidence,
                    'correct': correct,
                    'total': total
                }
            
            results[f'HAM_{category}'] = category_results
        
        return results
    
    def analyze_results(self, results):
        """Analyze test results for each model"""
        analysis = {}
        
        for model_name in self.models.keys():
            analysis[model_name] = {
                'perfect_categories': [],
                'excellent_categories': [],
                'good_categories': [],
                'okay_categories': [],
                'poor_categories': [],
                'spam_accuracy': 0,
                'ham_accuracy': 0,
                'overall_accuracy': 0
            }
        
        # Categorize performance
        for category_name, category_results in results.items():
            for model_name, stats in category_results.items():
                if model_name in analysis:
                    accuracy = stats['accuracy']
                    
                    if accuracy == 100:
                        analysis[model_name]['perfect_categories'].append(category_name)
                    elif accuracy >= 95:
                        analysis[model_name]['excellent_categories'].append(category_name)
                    elif accuracy >= 85:
                        analysis[model_name]['good_categories'].append(category_name)
                    elif accuracy >= 70:
                        analysis[model_name]['okay_categories'].append(category_name)
                    else:
                        analysis[model_name]['poor_categories'].append(category_name)
        
        # Calculate overall stats
        for model_name in analysis:
            spam_scores = []
            ham_scores = []
            all_scores = []
            
            for category_name, category_results in results.items():
                if model_name in category_results:
                    accuracy = category_results[model_name]['accuracy']
                    all_scores.append(accuracy)
                    
                    if category_name.startswith('SPAM_'):
                        spam_scores.append(accuracy)
                    elif category_name.startswith('HAM_'):
                        ham_scores.append(accuracy)
            
            analysis[model_name]['spam_accuracy'] = sum(spam_scores) / len(spam_scores) if spam_scores else 0
            analysis[model_name]['ham_accuracy'] = sum(ham_scores) / len(ham_scores) if ham_scores else 0
            analysis[model_name]['overall_accuracy'] = sum(all_scores) / len(all_scores) if all_scores else 0
        
        return analysis, results
    
    def create_pdf_report(self, analysis, results, output_filename="Spam_Detection_Performance_Report.pdf"):
        """Create PDF report"""
        if not REPORTLAB_AVAILABLE:
            if not install_reportlab():
                return False
            
            # Re-import after installation
            global SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
            global getSampleStyleSheet, ParagraphStyle, colors, letter, TA_CENTER, TA_LEFT, inch
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            from reportlab.lib.enums import TA_CENTER, TA_LEFT
        
        doc = SimpleDocTemplate(output_filename, pagesize=letter, 
                              rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        story = []
        
        # Title
        story.append(Paragraph("Spam Detection Model Performance Report", title_style))
        story.append(Spacer(1, 20))
        
        # Summary
        story.append(Paragraph("Executive Summary", heading_style))
        
        summary_text = f"""
        <b>Test Overview:</b><br/>
        • Models Tested: {', '.join(self.models.keys())}<br/>
        • SPAM Categories: {len(self.spam_examples)}<br/>
        • HAM Categories: {len(self.ham_examples)}<br/>
        • Examples per Category: 5<br/>
        • Total Tests: {len(self.spam_examples) * 5 + len(self.ham_examples) * 5} per model<br/>
        <br/>
        This report analyzes spam detection model performance across diverse content types.
        """
        
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # For your requested short summary format, we skip cross-model comparison and focus per model.
        
        # Per-model sections
        for model_name in analysis:
            story.append(PageBreak())
            story.append(Paragraph(f"{model_name} - Summary", heading_style))

            # Metrics table
            metrics_data = [
                ['Metric', 'Value'],
                ['Overall Accuracy', f"{analysis[model_name]['overall_accuracy']:.1f}%"],
                ['SPAM Detection Accuracy', f"{analysis[model_name]['spam_accuracy']:.1f}%"],
                ['HAM Detection Accuracy', f"{analysis[model_name]['ham_accuracy']:.1f}%"],
                ['Number of Perfect Categories', str(len(analysis[model_name]['perfect_categories']))]
            ]
            metrics_table = Table(metrics_data, colWidths=[2.7*inch, 3.3*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(metrics_table)
            story.append(Spacer(1, 12))

            # Helper: partition categories by type for perfect categories
            perfect_spam = []
            perfect_ham = []
            for cat in analysis[model_name]['perfect_categories']:
                if cat.startswith('SPAM_'):
                    perfect_spam.append(cat.replace('SPAM_', ''))
                elif cat.startswith('HAM_'):
                    perfect_ham.append(cat.replace('HAM_', ''))

            # Perfectly works on table
            story.append(Paragraph("Perfectly Works On", heading_style))
            perfect_data = [
                ['Type', 'Categories'],
                ['SPAM', Paragraph('<br/>'.join(perfect_spam) if perfect_spam else 'None', styles['Normal'])],
                ['HAM', Paragraph('<br/>'.join(perfect_ham) if perfect_ham else 'None', styles['Normal'])]
            ]
            perfect_table = Table(perfect_data, colWidths=[1.0*inch, 5.0*inch])
            perfect_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            story.append(perfect_table)
            story.append(Spacer(1, 12))

            # Not recommended table (poor performance: <70%)
            story.append(Paragraph("Not Recommended (Low Accuracy)", heading_style))
            poor_spam = []
            poor_ham = []
            # Build from results using threshold
            for category_name, category_results in results.items():
                if model_name in category_results:
                    acc = category_results[model_name]['accuracy']
                    if acc < 70:
                        if category_name.startswith('SPAM_'):
                            poor_spam.append(category_name.replace('SPAM_', ''))
                        elif category_name.startswith('HAM_'):
                            poor_ham.append(category_name.replace('HAM_', ''))
            poor_data = [
                ['Type', 'Categories'],
                ['SPAM', Paragraph('<br/>'.join(poor_spam) if poor_spam else 'None', styles['Normal'])],
                ['HAM', Paragraph('<br/>'.join(poor_ham) if poor_ham else 'None', styles['Normal'])]
            ]
            poor_table = Table(poor_data, colWidths=[1.0*inch, 5.0*inch])
            poor_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.red),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            story.append(poor_table)
            story.append(Spacer(1, 12))

            # Best for what type of spam and ham (within this model)
            story.append(Paragraph("Best For (High Accuracy Categories)", heading_style))
            # Choose high-accuracy buckets: >=95%
            best_spam = []
            best_ham = []
            for category_name, category_results in results.items():
                if model_name in category_results:
                    acc = category_results[model_name]['accuracy']
                    if acc >= 95:
                        if category_name.startswith('SPAM_'):
                            best_spam.append(category_name.replace('SPAM_', ''))
                        elif category_name.startswith('HAM_'):
                            best_ham.append(category_name.replace('HAM_', ''))
            # Fallback: if none meet >=95%, pick top 2 by accuracy per type
            if not best_spam or not best_ham:
                # collect per-type with accuracy
                spam_acc = []
                ham_acc = []
                for category_name, category_results in results.items():
                    if model_name in category_results:
                        acc = category_results[model_name]['accuracy']
                        if category_name.startswith('SPAM_'):
                            spam_acc.append((category_name.replace('SPAM_', ''), acc))
                        elif category_name.startswith('HAM_'):
                            ham_acc.append((category_name.replace('HAM_', ''), acc))
                spam_acc.sort(key=lambda x: x[1], reverse=True)
                ham_acc.sort(key=lambda x: x[1], reverse=True)
                if not best_spam:
                    best_spam = [name for name, _ in spam_acc[:2]]
                if not best_ham:
                    best_ham = [name for name, _ in ham_acc[:2]]

            best_data = [
                ['Type', 'Categories'],
                ['SPAM', Paragraph('<br/>'.join(best_spam) if best_spam else 'None', styles['Normal'])],
                ['HAM', Paragraph('<br/>'.join(best_ham) if best_ham else 'None', styles['Normal'])]
            ]
            best_table = Table(best_data, colWidths=[1.0*inch, 5.0*inch])
            best_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            story.append(best_table)
        
        # Build PDF
        doc.build(story)
        return True

def main():
    """Generate direct PDF report"""
    generator = DirectPDFReportGenerator()
    
    if not generator.models:
        print("❌ No models loaded successfully")
        return
    
    print(f"✅ {len(generator.models)} models loaded: {list(generator.models.keys())}")
    
    # Run tests
    results = generator.test_categories()
    
    # Analyze results
    analysis, results = generator.analyze_results(results)
    
    # Generate PDF
    output_file = "Spam_Detection_Model_Performance_Report.pdf"
    
    if generator.create_pdf_report(analysis, results, output_file):
        print(f"\n🎉 PDF report generated successfully: {output_file}")
        print(f"📄 Report contains performance analysis across {len(generator.spam_examples) + len(generator.ham_examples)} categories")
        print(f"📊 Shows which models excel at which spam/ham types")
        print(f"💡 Includes specific recommendations for different use cases")
    else:
        print("❌ Failed to generate PDF report")

if __name__ == "__main__":
    main()