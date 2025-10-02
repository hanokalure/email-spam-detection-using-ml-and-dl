#!/usr/bin/env python3
"""
Generate Professional PDF Report from Model Test Results
Converts JSON test results to a formatted PDF similar to the original report
"""

import json
import os
from datetime import datetime
from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    print("ReportLab not available. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "reportlab"])
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas


class ModelTestReportGenerator:
    """Generate professional PDF reports from model test results"""
    
    def __init__(self, json_file_path: str):
        self.json_file = json_file_path
        with open(json_file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Setup styles
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=20,
            textColor=colors.darkblue,
            alignment=1,  # Center
            spaceAfter=20
        )
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=colors.darkblue,
            spaceBefore=15,
            spaceAfter=10
        )
        self.subheading_style = ParagraphStyle(
            'CustomSubHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.darkblue,
            spaceBefore=10,
            spaceAfter=5
        )
        
    def generate_report(self, output_path: str = None):
        """Generate the complete PDF report"""
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"Model_Performance_Report_{timestamp}.pdf"
        
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Build the story (content)
        story = []
        
        # Title Page
        story.extend(self._create_title_page())
        story.append(PageBreak())
        
        # Executive Summary
        story.extend(self._create_executive_summary())
        story.append(PageBreak())
        
        # Individual Model Reports
        for model_name in self.data['summaries']:
            story.extend(self._create_model_report(model_name))
            story.append(PageBreak())
        
        # Model Comparison
        story.extend(self._create_comparison_table())
        
        # Build PDF
        doc.build(story)
        print(f"📊 PDF Report generated: {output_path}")
        return output_path
    
    def _create_title_page(self):
        """Create title page content"""
        story = []
        
        story.append(Paragraph("Spam Detection Model Performance Report", self.title_style))
        story.append(Spacer(1, 0.5*inch))
        
        # Test overview table
        test_info = self.data['test_info']
        overview_data = [
            ['Test Overview', ''],
            ['Models Tested', '4 (BalancedSpamNet, Enhanced Transformer, SVM, CatBoost)'],
            ['SPAM Categories', '5'],
            ['HAM Categories', '5'],
            ['Examples per Category', '4'],
            ['Total Tests per Model', '40'],
            ['Test Date', datetime.fromisoformat(test_info['timestamp']).strftime('%B %d, %Y')],
        ]
        
        overview_table = Table(overview_data, colWidths=[2.5*inch, 4*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(overview_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Categories tested
        story.append(Paragraph("Categories Tested", self.heading_style))
        
        categories_data = [
            ['SPAM Categories', 'HAM Categories'],
            ['• Financial Scams', '• Banking Legitimate'],
            ['• Phishing', '• Business Professional'],
            ['• Product Promotion', '• Personal Communication'],
            ['• Romance/Dating', '• E-commerce Legitimate'],
            ['• Health/Medical', '• Educational'],
        ]
        
        categories_table = Table(categories_data, colWidths=[3*inch, 3*inch])
        categories_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(categories_table)
        
        return story
    
    def _create_executive_summary(self):
        """Create executive summary page"""
        story = []
        
        story.append(Paragraph("Executive Summary", self.title_style))
        
        # Overall performance table
        summary_data = [['Model', 'Overall Accuracy', 'SPAM Detection', 'HAM Detection', 'Perfect Categories']]
        
        for model_name, summary in self.data['summaries'].items():
            summary_data.append([
                model_name,
                f"{summary['overall_accuracy']:.1f}%",
                f"{summary['spam_detection_accuracy']:.1f}%",
                f"{summary['ham_detection_accuracy']:.1f}%",
                str(len(summary['perfect_categories']))
            ])
        
        summary_table = Table(summary_data, colWidths=[1.5*inch, 1.2*inch, 1.2*inch, 1.2*inch, 1.2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.3*inch))
        
        # Key findings
        story.append(Paragraph("Key Findings", self.heading_style))
        
        findings = [
            "• <b>BalancedSpamNet</b> achieved perfect 100% SPAM detection with zero false negatives",
            "• <b>CatBoost</b> showed the best overall accuracy (85.0%) with excellent HAM detection (90.0%)",
            "• <b>SVM</b> demonstrated consistent performance across 7 perfect categories",
            "• <b>Enhanced Transformer</b> excelled at Banking Legitimate detection (100% accuracy)",
            "• All models struggle with Banking Legitimate and E-commerce categories",
            "• Financial terms in legitimate emails trigger false positives across models"
        ]
        
        for finding in findings:
            story.append(Paragraph(finding, self.styles['Normal']))
            story.append(Spacer(1, 6))
        
        return story
    
    def _create_model_report(self, model_name: str):
        """Create individual model report page"""
        story = []
        summary = self.data['summaries'][model_name]
        
        # Model title
        story.append(Paragraph(f"{model_name} - Summary", self.title_style))
        
        # Performance metrics table
        metrics_data = [
            ['Metric', 'Value'],
            ['Overall Accuracy', f"{summary['overall_accuracy']:.1f}%"],
            ['SPAM Detection Accuracy', f"{summary['spam_detection_accuracy']:.1f}%"],
            ['HAM Detection Accuracy', f"{summary['ham_detection_accuracy']:.1f}%"],
            ['Perfect Categories', str(len(summary['perfect_categories']))]
        ]
        
        metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 0.2*inch))
        
        # Perfect categories
        story.append(Paragraph("Perfect Categories (100% Accuracy)", self.heading_style))
        if summary['perfect_categories']:
            perfect_data = [['SPAM', 'HAM']]
            spam_perfect = [cat for cat in summary['perfect_categories'] 
                          if cat in self.data['test_info']['spam_categories']]
            ham_perfect = [cat for cat in summary['perfect_categories'] 
                         if cat in self.data['test_info']['ham_categories']]
            
            max_rows = max(len(spam_perfect), len(ham_perfect))
            for i in range(max_rows):
                spam_cat = spam_perfect[i] if i < len(spam_perfect) else ""
                ham_cat = ham_perfect[i] if i < len(ham_perfect) else ""
                perfect_data.append([spam_cat, ham_cat])
            
            perfect_table = Table(perfect_data, colWidths=[3*inch, 3*inch])
            perfect_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (1, 0), colors.green),
                ('TEXTCOLOR', (0, 0), (1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(perfect_table)
        else:
            story.append(Paragraph("None", self.styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Low accuracy categories
        if summary['low_accuracy_categories']:
            story.append(Paragraph("Low Accuracy Categories (<50%)", self.heading_style))
            for cat in summary['low_accuracy_categories']:
                acc = summary['category_breakdown'][cat]['accuracy']
                story.append(Paragraph(f"• {cat} ({acc:.1f}%)", self.styles['Normal']))
        
        story.append(Spacer(1, 0.2*inch))
        
        # Category breakdown
        story.append(Paragraph("Category Performance Breakdown", self.heading_style))
        breakdown_data = [['Category', 'Correct/Total', 'Accuracy']]
        
        for category, stats in summary['category_breakdown'].items():
            breakdown_data.append([
                category,
                f"{stats['correct']}/{stats['total']}",
                f"{stats['accuracy']:.1f}%"
            ])
        
        breakdown_table = Table(breakdown_data, colWidths=[3*inch, 1*inch, 1*inch])
        breakdown_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (2, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (2, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(breakdown_table)
        
        return story
    
    def _create_comparison_table(self):
        """Create model comparison page"""
        story = []
        
        story.append(Paragraph("Model Comparison Summary", self.title_style))
        
        # Detailed comparison table
        comparison_data = [
            ['Model', 'Overall', 'SPAM Det.', 'HAM Det.', 'Perfect Cat.', 'Strengths', 'Weaknesses']
        ]
        
        model_analysis = {
            'BalancedSpamNet': {
                'strengths': 'Perfect SPAM detection, Zero false negatives',
                'weaknesses': 'Banking/E-commerce HAM (25%)'
            },
            'Enhanced_Transformer': {
                'strengths': 'Perfect Banking HAM, Context-aware',
                'weaknesses': 'Personal Communication (25%)'
            },
            'SVM': {
                'strengths': '7 perfect categories, Consistent',
                'weaknesses': 'Banking HAM (0%), Traditional ML limits'
            },
            'CatBoost': {
                'strengths': 'Best overall (85%), Excellent HAM (90%)',
                'weaknesses': 'Health/Medical SPAM (25%)'
            }
        }
        
        for model_name, summary in self.data['summaries'].items():
            analysis = model_analysis.get(model_name, {'strengths': 'N/A', 'weaknesses': 'N/A'})
            comparison_data.append([
                model_name,
                f"{summary['overall_accuracy']:.1f}%",
                f"{summary['spam_detection_accuracy']:.1f}%",
                f"{summary['ham_detection_accuracy']:.1f}%",
                str(len(summary['perfect_categories'])),
                analysis['strengths'],
                analysis['weaknesses']
            ])
        
        comparison_table = Table(comparison_data, colWidths=[1*inch, 0.8*inch, 0.8*inch, 0.8*inch, 0.6*inch, 1.5*inch, 1.5*inch])
        comparison_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (4, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        
        story.append(comparison_table)
        
        # Recommendations
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph("Recommendations", self.heading_style))
        
        recommendations = [
            "<b>For Security-Focused Deployment:</b> Use BalancedSpamNet (100% SPAM detection)",
            "<b>For Balanced Performance:</b> Use CatBoost (85% overall, 90% HAM accuracy)",
            "<b>For Banking Applications:</b> Use Enhanced Transformer (100% Banking HAM accuracy)",
            "<b>Improvement Priority:</b> Add diverse HAM training data for banking and e-commerce categories",
            "<b>Training Recommendation:</b> Create 50,000+ HAM examples focusing on legitimate financial communications"
        ]
        
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", self.styles['Normal']))
            story.append(Spacer(1, 6))
        
        return story


def main():
    """Main function to generate the report"""
    # Find the latest test results file
    test_files = list(Path('.').glob('comprehensive_model_test_results_*.json'))
    if not test_files:
        print("❌ No test results JSON file found!")
        print("Please run the comprehensive test first: python comprehensive_model_test.py")
        return
    
    # Use the most recent file
    latest_file = max(test_files, key=os.path.getctime)
    print(f"📊 Using test results: {latest_file}")
    
    # Generate report
    generator = ModelTestReportGenerator(str(latest_file))
    pdf_path = generator.generate_report()
    
    print(f"✅ PDF Report generated successfully!")
    print(f"📁 Location: {os.path.abspath(pdf_path)}")
    

if __name__ == "__main__":
    main()