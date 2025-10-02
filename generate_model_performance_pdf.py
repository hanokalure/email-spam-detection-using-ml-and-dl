#!/usr/bin/env python3
"""
PDF Report Generator for Spam Detection Model Performance
Analyzes comprehensive test results and generates detailed PDF report
"""

import json
import os
from collections import defaultdict
from datetime import datetime

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

def install_reportlab():
    """Install ReportLab if not available"""
    print("📦 Installing ReportLab for PDF generation...")
    import subprocess
    import sys
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
        print("✅ ReportLab installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install ReportLab")
        return False

class SpamDetectionPDFReport:
    """Generate comprehensive PDF report for spam detection models"""
    
    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.data = None
        self.load_results()
        
    def load_results(self):
        """Load test results from JSON file"""
        if not os.path.exists(self.json_file_path):
            raise FileNotFoundError(f"Results file not found: {self.json_file_path}")
            
        with open(self.json_file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        print(f"✅ Loaded test results: {len(self.data['detailed_results'])} categories")
    
    def analyze_model_performance(self):
        """Analyze performance of each model across categories"""
        analysis = {}
        
        for model_name in self.data['test_metadata']['models_tested']:
            analysis[model_name] = {
                'perfect_categories': [],
                'excellent_categories': [],  # 95-99%
                'good_categories': [],       # 85-94%
                'okay_categories': [],       # 70-84%
                'poor_categories': [],       # <70%
                'category_scores': {},
                'spam_accuracy': 0,
                'ham_accuracy': 0,
                'overall_accuracy': 0,
                'best_examples': [],
                'problematic_examples': []
            }
        
        # Calculate accuracy for each category and model
        for category_name, category_results in self.data['detailed_results'].items():
            for model_name, predictions in category_results.items():
                if model_name in analysis:
                    correct = sum(1 for p in predictions if p['correct'])
                    total = len(predictions)
                    accuracy = (correct / total) * 100 if total > 0 else 0
                    
                    analysis[model_name]['category_scores'][category_name] = accuracy
                    
                    # Categorize performance
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
                    
                    # Find best and problematic examples
                    for pred in predictions:
                        if pred['correct'] and pred['confidence'] > 0.95:
                            analysis[model_name]['best_examples'].append({
                                'category': category_name,
                                'text': pred['text'][:100] + "..." if len(pred['text']) > 100 else pred['text'],
                                'confidence': pred['confidence']
                            })
                        elif not pred['correct']:
                            analysis[model_name]['problematic_examples'].append({
                                'category': category_name,
                                'text': pred['text'][:100] + "..." if len(pred['text']) > 100 else pred['text'],
                                'predicted': pred['prediction'],
                                'expected': pred['expected'],
                                'confidence': pred['confidence']
                            })
        
        # Calculate overall stats
        for model_name in analysis:
            spam_categories = [cat for cat in analysis[model_name]['category_scores'] if cat.startswith('SPAM_')]
            ham_categories = [cat for cat in analysis[model_name]['category_scores'] if cat.startswith('HAM_')]
            
            if spam_categories:
                analysis[model_name]['spam_accuracy'] = sum(analysis[model_name]['category_scores'][cat] for cat in spam_categories) / len(spam_categories)
            
            if ham_categories:
                analysis[model_name]['ham_accuracy'] = sum(analysis[model_name]['category_scores'][cat] for cat in ham_categories) / len(ham_categories)
            
            all_scores = list(analysis[model_name]['category_scores'].values())
            analysis[model_name]['overall_accuracy'] = sum(all_scores) / len(all_scores) if all_scores else 0
            
            # Limit examples for PDF
            analysis[model_name]['best_examples'] = sorted(
                analysis[model_name]['best_examples'], 
                key=lambda x: x['confidence'], 
                reverse=True
            )[:5]
            
            analysis[model_name]['problematic_examples'] = analysis[model_name]['problematic_examples'][:5]
        
        return analysis
    
    def create_pdf_report(self, output_filename: str = "spam_detection_model_report.pdf"):
        """Create comprehensive PDF report"""
        if not REPORTLAB_AVAILABLE:
            print("❌ ReportLab not available for PDF generation")
            if install_reportlab():
                # Re-import after installation
                global SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
                global getSampleStyleSheet, ParagraphStyle, colors, letter
                from reportlab.lib import colors
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
                from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                from reportlab.lib.units import inch
                from reportlab.lib.enums import TA_CENTER, TA_LEFT
            else:
                return False
        
        # Analyze performance
        analysis = self.analyze_model_performance()
        
        # Create PDF document
        doc = SimpleDocTemplate(output_filename, pagesize=letter, 
                              rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
        
        # Styles
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
        
        subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=styles['Heading3'],
            fontSize=14,
            spaceAfter=10,
            textColor=colors.darkgreen
        )
        
        # Story (content) list
        story = []
        
        # Title Page
        story.append(Paragraph("Spam Detection Model Performance Report", title_style))
        story.append(Spacer(1, 20))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", heading_style))
        
        timestamp = self.data['test_metadata']['timestamp']
        models_tested = ", ".join(self.data['test_metadata']['models_tested'])
        total_tests = self.data['test_metadata']['total_categories'] * self.data['test_metadata']['examples_per_category'] * len(self.data['test_metadata']['models_tested'])
        
        summary_text = f"""
        <b>Test Overview:</b><br/>
        • Date: {timestamp}<br/>
        • Models Tested: {models_tested}<br/>
        • Categories: {self.data['test_metadata']['total_categories']}<br/>
        • Examples per Category: {self.data['test_metadata']['examples_per_category']}<br/>
        • Total Predictions: {total_tests}<br/>
        <br/>
        This report analyzes the performance of three spam detection models across diverse email categories,
        providing insights into which model performs best for different types of content.
        """
        
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Overall Performance Summary Table
        story.append(Paragraph("Overall Performance Summary", heading_style))
        
        summary_data = [['Model', 'Overall Accuracy', 'SPAM Detection', 'HAM Detection', 'Perfect Categories']]
        
        for model_name in analysis:
            summary_data.append([
                model_name,
                f"{analysis[model_name]['overall_accuracy']:.1f}%",
                f"{analysis[model_name]['spam_accuracy']:.1f}%",
                f"{analysis[model_name]['ham_accuracy']:.1f}%",
                str(len(analysis[model_name]['perfect_categories']))
            ])
        
        summary_table = Table(summary_data)
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
        
        # Detailed Model Analysis
        for model_name in analysis:
            story.append(PageBreak())
            story.append(Paragraph(f"{model_name} - Detailed Analysis", heading_style))
            
            # Performance categories
            perf_data = [
                ['Performance Level', 'Categories', 'Count'],
                ['Perfect (100%)', ', '.join([cat.replace('SPAM_', '').replace('HAM_', '') for cat in analysis[model_name]['perfect_categories']]) or 'None', len(analysis[model_name]['perfect_categories'])],
                ['Excellent (95-99%)', ', '.join([cat.replace('SPAM_', '').replace('HAM_', '') for cat in analysis[model_name]['excellent_categories']]) or 'None', len(analysis[model_name]['excellent_categories'])],
                ['Good (85-94%)', ', '.join([cat.replace('SPAM_', '').replace('HAM_', '') for cat in analysis[model_name]['good_categories']]) or 'None', len(analysis[model_name]['good_categories'])],
                ['Okay (70-84%)', ', '.join([cat.replace('SPAM_', '').replace('HAM_', '') for cat in analysis[model_name]['okay_categories']]) or 'None', len(analysis[model_name]['okay_categories'])],
                ['Poor (<70%)', ', '.join([cat.replace('SPAM_', '').replace('HAM_', '') for cat in analysis[model_name]['poor_categories']]) or 'None', len(analysis[model_name]['poor_categories'])]
            ]
            
            perf_table = Table(perf_data, colWidths=[1.5*inch, 3.5*inch, 0.8*inch])
            perf_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP')
            ]))
            
            story.append(perf_table)
            story.append(Spacer(1, 15))
            
            # Category-wise detailed scores
            story.append(Paragraph("Category-wise Accuracy Scores", subheading_style))
            
            # Split into SPAM and HAM categories
            spam_scores = [(cat, score) for cat, score in analysis[model_name]['category_scores'].items() if cat.startswith('SPAM_')]
            ham_scores = [(cat, score) for cat, score in analysis[model_name]['category_scores'].items() if cat.startswith('HAM_')]
            
            # SPAM categories table
            if spam_scores:
                story.append(Paragraph("SPAM Categories:", styles['Normal']))
                spam_data = [['Category', 'Accuracy']]
                for cat, score in sorted(spam_scores, key=lambda x: x[1], reverse=True):
                    clean_cat = cat.replace('SPAM_', '').replace('_', ' ').title()
                    spam_data.append([clean_cat, f"{score:.1f}%"])
                
                spam_table = Table(spam_data, colWidths=[3*inch, 1*inch])
                spam_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.red),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightpink),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(spam_table)
                story.append(Spacer(1, 10))
            
            # HAM categories table
            if ham_scores:
                story.append(Paragraph("HAM (Legitimate) Categories:", styles['Normal']))
                ham_data = [['Category', 'Accuracy']]
                for cat, score in sorted(ham_scores, key=lambda x: x[1], reverse=True):
                    clean_cat = cat.replace('HAM_', '').replace('_', ' ').title()
                    ham_data.append([clean_cat, f"{score:.1f}%"])
                
                ham_table = Table(ham_data, colWidths=[3*inch, 1*inch])
                ham_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.green),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(ham_table)
                story.append(Spacer(1, 15))
            
            # Best performing examples
            if analysis[model_name]['best_examples']:
                story.append(Paragraph("Top 5 High-Confidence Correct Predictions:", subheading_style))
                
                for i, example in enumerate(analysis[model_name]['best_examples'], 1):
                    example_text = f"<b>{i}. {example['category'].replace('SPAM_', '').replace('HAM_', '').replace('_', ' ').title()}:</b><br/>"
                    example_text += f"<i>'{example['text']}'</i><br/>"
                    example_text += f"Confidence: {example['confidence']:.3f}<br/><br/>"
                    
                    story.append(Paragraph(example_text, styles['Normal']))
            
            # Problematic examples
            if analysis[model_name]['problematic_examples']:
                story.append(Paragraph("Examples of Incorrect Predictions:", subheading_style))
                
                for i, example in enumerate(analysis[model_name]['problematic_examples'], 1):
                    example_text = f"<b>{i}. {example['category'].replace('SPAM_', '').replace('HAM_', '').replace('_', ' ').title()}:</b><br/>"
                    example_text += f"<i>'{example['text']}'</i><br/>"
                    example_text += f"Predicted: {example['predicted']} | Expected: {example['expected']} | Confidence: {example['confidence']:.3f}<br/><br/>"
                    
                    story.append(Paragraph(example_text, styles['Normal']))
        
        # Final Recommendations
        story.append(PageBreak())
        story.append(Paragraph("Recommendations & Conclusions", heading_style))
        
        # Find best model for each category type
        best_overall = max(analysis.keys(), key=lambda m: analysis[m]['overall_accuracy'])
        best_spam = max(analysis.keys(), key=lambda m: analysis[m]['spam_accuracy'])
        best_ham = max(analysis.keys(), key=lambda m: analysis[m]['ham_accuracy'])
        most_perfect = max(analysis.keys(), key=lambda m: len(analysis[m]['perfect_categories']))
        
        recommendations = f"""
        <b>Key Findings:</b><br/>
        • Best Overall Performance: <b>{best_overall}</b> ({analysis[best_overall]['overall_accuracy']:.1f}% accuracy)<br/>
        • Best SPAM Detection: <b>{best_spam}</b> ({analysis[best_spam]['spam_accuracy']:.1f}% accuracy)<br/>
        • Best HAM Detection: <b>{best_ham}</b> ({analysis[best_ham]['ham_accuracy']:.1f}% accuracy)<br/>
        • Most Perfect Categories: <b>{most_perfect}</b> ({len(analysis[most_perfect]['perfect_categories'])} categories)<br/>
        <br/>
        <b>Usage Recommendations:</b><br/>
        • For maximum spam detection: Use <b>{best_spam}</b><br/>
        • For protecting legitimate emails: Use <b>{best_ham}</b><br/>
        • For balanced performance: Use <b>{best_overall}</b><br/>
        • For consistent performance across categories: Use <b>{most_perfect}</b><br/>
        """
        
        story.append(Paragraph(recommendations, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        print(f"✅ PDF report generated: {output_filename}")
        return True

def main():
    """Generate the PDF report"""
    results_file = "comprehensive_spam_test_results.json"
    output_file = "Spam_Detection_Model_Performance_Report.pdf"
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        print("Please run 'python comprehensive_model_test.py' first to generate test results.")
        return
    
    try:
        report_generator = SpamDetectionPDFReport(results_file)
        
        if report_generator.create_pdf_report(output_file):
            print(f"\n🎉 PDF report successfully created: {output_file}")
            print(f"📄 The report contains detailed analysis of model performance across categories")
            print(f"📊 Including perfect/excellent/good/okay performance breakdowns")
            print(f"💡 With specific recommendations for each use case")
        else:
            print("❌ Failed to generate PDF report")
            
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()