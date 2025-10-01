#!/usr/bin/env python3
"""
Comprehensive Spam Dataset Curator
- Combines your existing 14K dataset with diverse spam examples
- Covers all major spam categories: phishing, financial, romance, tech, crypto, work-from-home
- Adds hard examples for edge cases your model is missing
- Balances dataset for maximum recall training
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import random
from datetime import datetime


class ComprehensiveSpamDatasetCurator:
    """Curates comprehensive spam dataset for maximum recall training"""
    
    def __init__(self):
        self.spam_categories = {
            'phishing': [],
            'financial': [],
            'romance': [],
            'tech_support': [],
            'crypto': [],
            'work_from_home': [],
            'product_scam': [],
            'lottery_inheritance': [],
            'adult_content': [],
            'fake_services': []
        }
        
        # Your problematic cases - these will be added to ensure they're caught
        self.hard_examples = [
            ("We noticed unusual login attempts. Reset your password immediately: http://phishingsite.com", "spam"),
            ("Make ₹20,000 daily by working only 1 hour from home. Join now:", "spam"),
            ("Your account security alert. Verify now: http://secure-bank-verify.com", "spam"),
            ("Exclusive work opportunity - earn $5000/month from home. No experience required.", "spam"),
            ("Alert: Suspicious activity detected. Click to secure: https://account-security.net", "spam"),
            ("Limited time - make easy money online. Start today: http://easycash.biz", "spam"),
            ("Account will be suspended. Verify identity: http://verify-account.com", "spam"),
            ("Get your Elite Pass instantly by paying ₹250. Don't miss out!", "spam"),
            ("Pay ₹999 as security deposit and start your dream job today. Daily payouts assured.", "spam"),
        ]
    
    def generate_phishing_samples(self, count: int = 500) -> List[Tuple[str, str]]:
        """Generate diverse phishing email examples"""
        
        # Phishing templates
        templates = [
            # Banking/Financial phishing
            "Urgent security alert for your {bank} account. Verify immediately: {url}",
            "Your {bank} account has been temporarily suspended. Restore access: {url}", 
            "Unusual login detected from {location}. Secure your account: {url}",
            "Action required: Verify your {service} account to prevent suspension: {url}",
            "Your payment was declined. Update billing information: {url}",
            "Security breach detected. Change your password now: {url}",
            
            # Email/Tech service phishing  
            "Your {service} storage is full. Upgrade now: {url}",
            "Your {service} account expires today. Renew immediately: {url}",
            "Important {service} security update required: {url}",
            "Your email will be deactivated unless you verify: {url}",
            
            # Government/Official phishing
            "Tax refund of ${amount} approved. Claim now: {url}",
            "Your visa application requires verification: {url}",
            "Court notice: Respond within 24 hours: {url}",
            "Government benefit waiting for you: {url}",
            
            # Package/Delivery phishing
            "Your package delivery failed. Reschedule: {url}",
            "Customs fee required for your shipment: {url}",
            "Package held - pay delivery fee: {url}",
        ]
        
        # Variables for templates
        banks = ["HDFC", "SBI", "ICICI", "Axis Bank", "Bank of America", "Chase", "Wells Fargo", "PayPal"]
        services = ["Gmail", "Outlook", "Facebook", "Instagram", "Netflix", "Amazon", "Microsoft", "Apple"]
        locations = ["Mumbai, India", "New York, USA", "London, UK", "Singapore", "Dubai, UAE"]
        amounts = ["1,500", "2,850", "5,000", "10,250"]
        
        # Suspicious domains
        domains = [
            "secure-banking.net", "account-verify.com", "security-alert.org", 
            "bank-security.info", "verify-account.net", "secure-login.co",
            "account-protection.com", "security-check.net", "verify-now.org",
            "account-suspended.com", "urgent-verify.net", "secure-access.info"
        ]
        
        samples = []
        for _ in range(count):
            template = random.choice(templates)
            
            # Fill template
            text = template.format(
                bank=random.choice(banks),
                service=random.choice(services), 
                location=random.choice(locations),
                amount=random.choice(amounts),
                url="http://" + random.choice(domains)
            )
            
            samples.append((text, "spam"))
        
        return samples
    
    def generate_financial_scam_samples(self, count: int = 400) -> List[Tuple[str, str]]:
        """Generate financial scam examples"""
        
        templates = [
            "Congratulations! You've won ${amount} in our lottery. Claim: {url}",
            "Your inheritance of ${amount} is ready. Process now: {url}", 
            "Loan approved for ₹{amount}. Get cash today: {url}",
            "Make ₹{amount} daily with our investment plan: {url}",
            "Get instant loan of ${amount}. Apply now: {url}",
            "You've won the mega lottery of €{amount}! Claim immediately: {url}",
            "Cryptocurrency investment - earn ${amount}/day guaranteed: {url}",
            "Your account credited with ${amount}. Verify to withdraw: {url}",
            "Tax refund of ₹{amount} approved. Claim within 24hrs: {url}",
            "Urgent: Unclaimed funds of ${amount} in your name: {url}",
        ]
        
        amounts_inr = ["50,000", "1,00,000", "2,50,000", "5,00,000", "10,00,000"]
        amounts_usd = ["5,000", "15,000", "50,000", "100,000", "500,000"]
        amounts_eur = ["10,000", "25,000", "75,000", "200,000"]
        
        domains = [
            "lottery-winner.com", "claim-prize.net", "instant-loan.org",
            "quick-cash.info", "mega-lottery.com", "crypto-profits.net"
        ]
        
        samples = []
        for _ in range(count):
            template = random.choice(templates)
            
            if "₹" in template:
                amount = random.choice(amounts_inr)
            elif "€" in template:
                amount = random.choice(amounts_eur)
            else:
                amount = random.choice(amounts_usd)
            
            text = template.format(
                amount=amount,
                url="http://" + random.choice(domains)
            )
            
            samples.append((text, "spam"))
        
        return samples
    
    def generate_work_from_home_scams(self, count: int = 300) -> List[Tuple[str, str]]:
        """Generate work-from-home scam examples"""
        
        templates = [
            "Earn ₹{amount} daily working from home. No experience needed: {url}",
            "Make ${amount}/month online. Flexible hours: {url}",
            "Work from home opportunity - earn ₹{amount} weekly: {url}",
            "Data entry jobs - earn ${amount} daily from home: {url}",
            "Online survey jobs - make ₹{amount} per survey: {url}",
            "Copy-paste work - earn ${amount} weekly from home: {url}",
            "Part-time online work - ₹{amount} daily guaranteed: {url}",
            "Students can earn ₹{amount}/day from home easily: {url}",
            "Typing jobs online - make ${amount} monthly: {url}",
            "Join our team and earn ₹{amount} working 2hrs/day: {url}",
        ]
        
        daily_amounts = ["1,000", "2,500", "5,000", "10,000", "15,000"]
        weekly_amounts = ["10,000", "25,000", "50,000", "75,000"]  
        monthly_amounts = ["50,000", "1,00,000", "2,00,000", "3,00,000"]
        
        domains = [
            "work-from-home.com", "online-jobs.net", "home-income.org",
            "easy-money.info", "part-time-work.com", "online-earning.net"
        ]
        
        samples = []
        for _ in range(count):
            template = random.choice(templates)
            
            if "daily" in template:
                amount = random.choice(daily_amounts)
            elif "weekly" in template:
                amount = random.choice(weekly_amounts)
            elif "monthly" in template:
                amount = random.choice(monthly_amounts)
            else:
                amount = random.choice(daily_amounts)
            
            text = template.format(
                amount=amount,
                url="http://" + random.choice(domains)
            )
            
            samples.append((text, "spam"))
        
        return samples
    
    def generate_romance_scam_samples(self, count: int = 200) -> List[Tuple[str, str]]:
        """Generate romance scam examples"""
        
        templates = [
            "Hi dear, I am {name} looking for true love. Please help me with ${amount} for visa.",
            "My beloved, I need your help. Send me ₹{amount} for emergency medical treatment.",
            "Darling, I am stuck in {location}. Please send ${amount} for flight ticket home.",
            "My love, I have inheritance of ${amount} but need ₹{amount} for legal fees.",
            "Dear, I am {profession} in {location}. Need ${amount} for business investment.",
            "Sweetheart, my father left me ${amount} but need your help with ₹{amount} for lawyer.",
            "My dear friend, I am lonely widow with ${amount} inheritance. Need ₹{amount} for transfer.",
            "Hello beautiful, I am {profession}. Will share my ${amount} wealth if you send ₹{amount}.",
        ]
        
        names = ["Sarah", "David", "Jennifer", "Michael", "Lisa", "James", "Maria", "Robert"]
        professions = ["doctor", "engineer", "businessman", "military officer", "lawyer", "teacher"]
        locations = ["London", "New York", "Dubai", "Singapore", "Sydney", "Tokyo"]
        amounts = ["5,000", "10,000", "25,000", "50,000", "100,000"]
        help_amounts = ["2,000", "5,000", "10,000", "15,000", "25,000"]
        
        samples = []
        for _ in range(count):
            template = random.choice(templates)
            text = template.format(
                name=random.choice(names),
                profession=random.choice(professions),
                location=random.choice(locations), 
                amount=random.choice(amounts)
            )
            
            # Replace ₹{amount} with help amounts
            if "₹{amount}" in text:
                text = text.replace("₹{amount}", f"₹{random.choice(help_amounts)}")
            
            samples.append((text, "spam"))
        
        return samples
    
    def generate_tech_support_scams(self, count: int = 200) -> List[Tuple[str, str]]:
        """Generate tech support scam examples"""
        
        templates = [
            "Your computer has {threat}! Call Microsoft support: {phone}",
            "Windows security alert: {threat} detected. Contact: {phone}",
            "Your PC is infected with {threat}. Get help: {phone}",
            "Critical security warning: {threat} found. Call: {phone}",  
            "Your computer will be blocked due to {threat}. Contact: {phone}",
            "Virus alert: {threat} detected on your system. Call: {phone}",
            "Your Windows license expired. Renew now: {phone}",
            "System compromised by {threat}. Immediate action required: {phone}",
        ]
        
        threats = ["virus", "malware", "trojan", "spyware", "ransomware", "adware", "suspicious activity"]
        phones = ["+1-800-SUPPORT", "1-855-123-4567", "+1-888-HELP-NOW", "1-844-MICROSOFT", "+1-877-WINDOWS"]
        
        samples = []
        for _ in range(count):
            template = random.choice(templates)
            text = template.format(
                threat=random.choice(threats),
                phone=random.choice(phones)
            )
            samples.append((text, "spam"))
        
        return samples
    
    def generate_legitimate_samples(self, count: int = 1000) -> List[Tuple[str, str]]:
        """Generate legitimate email examples for balance"""
        
        templates = [
            # Business emails
            "Please find attached the quarterly sales report for your review.",
            "Meeting scheduled for tomorrow at {time} in conference room {room}.",
            "Thank you for your email. I will respond by end of business today.",
            "The project deadline has been extended to {date}.",
            "Please review the attached contract and provide feedback by {date}.",
            "Your presentation was excellent. Great job on the quarterly results.",
            "Can you please send me the latest version of the {document}?",
            "The client meeting went well. They approved our proposal.",
            
            # Personal emails
            "Hi, how are you doing? Hope everything is going well.",
            "Thanks for the birthday wishes! Had a great celebration.",
            "Are we still on for dinner tomorrow at {time}?",
            "Just wanted to check in and see how you're doing.",
            "The photos from the vacation look amazing!",
            "Hope you feel better soon. Take care of yourself.",
            
            # Service emails
            "Your order #{order_id} has been shipped and will arrive by {date}.",
            "Your flight {flight} from {city1} to {city2} is confirmed for {date}.",
            "Your appointment with Dr. {name} is scheduled for {date} at {time}.",
            "Your subscription to {service} expires on {date}. Renew to continue.",
            "Welcome to {service}! Your account has been created successfully.",
            "Your password has been successfully changed for {service} account.",
            
            # Educational/Professional
            "Your application for the {position} role has been received.",
            "Assignment deadline reminder: {assignment} due on {date}.",
            "Registration for {course} is now open. Enroll by {date}.",
            "Your exam results for {subject} are now available online.",
            "Conference registration confirmed for {event} on {date}.",
        ]
        
        # Variables
        times = ["10:00 AM", "2:30 PM", "9:00 AM", "3:15 PM", "11:30 AM"]
        rooms = ["A", "B", "C", "Conference Room 1", "Meeting Room 2"]
        dates = ["Friday", "next Monday", "October 15th", "end of this month", "December 1st"]
        documents = ["budget report", "project proposal", "marketing plan", "user guide", "specification"]
        order_ids = ["12345", "67890", "ABC123", "XYZ789", "ORDER-001"]
        cities = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Hyderabad", "Kolkata"]
        flights = ["AI-203", "6E-123", "SG-456", "UK-789", "9W-101"]
        services = ["Netflix", "Spotify", "Amazon Prime", "Adobe Creative", "Microsoft Office"]
        positions = ["Software Engineer", "Marketing Manager", "Data Analyst", "Product Manager"]
        courses = ["Python Programming", "Digital Marketing", "Data Science", "Machine Learning"]
        
        samples = []
        for _ in range(count):
            template = random.choice(templates)
            
            # Fill placeholders
            text = template
            if "{time}" in text:
                text = text.replace("{time}", random.choice(times))
            if "{room}" in text:
                text = text.replace("{room}", random.choice(rooms))
            if "{date}" in text:
                text = text.replace("{date}", random.choice(dates))
            if "{document}" in text:
                text = text.replace("{document}", random.choice(documents))
            if "{order_id}" in text:
                text = text.replace("{order_id}", random.choice(order_ids))
            if "{city1}" in text:
                text = text.replace("{city1}", random.choice(cities))
            if "{city2}" in text:
                text = text.replace("{city2}", random.choice(cities))
            if "{flight}" in text:
                text = text.replace("{flight}", random.choice(flights))
            if "{service}" in text:
                text = text.replace("{service}", random.choice(services))
            if "{name}" in text:
                text = text.replace("{name}", random.choice(["Smith", "Johnson", "Patel", "Kumar"]))
            if "{position}" in text:
                text = text.replace("{position}", random.choice(positions))
            if "{assignment}" in text:
                text = text.replace("{assignment}", random.choice(["Research paper", "Project report", "Case study"]))
            if "{course}" in text:
                text = text.replace("{course}", random.choice(courses))
            if "{subject}" in text:
                text = text.replace("{subject}", random.choice(["Mathematics", "Computer Science", "Economics"]))
            if "{event}" in text:
                text = text.replace("{event}", random.choice(["Tech Conference 2024", "Marketing Summit", "Data Science Meetup"]))
            
            samples.append((text, "ham"))
        
        return samples
    
    def load_existing_dataset(self, file_path: str) -> pd.DataFrame:
        """Load your existing dataset"""
        try:
            df = pd.read_csv(file_path)
            
            # Normalize columns
            if 'text' in df.columns and 'label' in df.columns:
                df = df[['text', 'label']].copy()
            elif 'v1' in df.columns and 'v2' in df.columns:
                df = df[['v2', 'v1']].copy()
                df.columns = ['text', 'label']
            else:
                df = df.iloc[:, :2].copy()
                df.columns = ['text', 'label']
            
            # Normalize labels
            if df['label'].dtype == 'O':
                df['label'] = df['label'].str.lower().map({'ham': 'ham', 'spam': 'spam'})
            else:
                df['label'] = df['label'].map({0: 'ham', 1: 'spam'})
            
            df = df.dropna()
            print(f"✅ Loaded existing dataset: {len(df)} samples")
            print(f"   Ham: {len(df[df['label'] == 'ham'])}, Spam: {len(df[df['label'] == 'spam'])}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return pd.DataFrame(columns=['text', 'label'])
    
    def create_comprehensive_dataset(self, existing_dataset_path: str, output_path: str):
        """Create comprehensive spam dataset"""
        
        print("🚀 CREATING COMPREHENSIVE SPAM DATASET")
        print("=" * 60)
        
        # Load existing data
        existing_df = self.load_existing_dataset(existing_dataset_path)
        
        # Generate new samples
        print("\n📝 Generating additional spam samples...")
        
        all_samples = []
        
        # Add your hard examples first (critical!)
        print(f"Adding {len(self.hard_examples)} hard examples...")
        all_samples.extend(self.hard_examples)
        
        # Generate category samples
        categories = [
            ("Phishing", self.generate_phishing_samples, 500),
            ("Financial Scams", self.generate_financial_scam_samples, 400), 
            ("Work-from-Home", self.generate_work_from_home_scams, 300),
            ("Romance Scams", self.generate_romance_scam_samples, 200),
            ("Tech Support", self.generate_tech_support_scams, 200),
            ("Legitimate Emails", self.generate_legitimate_samples, 1000)
        ]
        
        for category_name, generator_func, count in categories:
            print(f"Generating {count} {category_name} samples...")
            samples = generator_func(count)
            all_samples.extend(samples)
        
        # Create new dataframe
        new_df = pd.DataFrame(all_samples, columns=['text', 'label'])
        
        # Combine with existing data
        if not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        # Remove duplicates
        print("🔄 Removing duplicates...")
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
        after_dedup = len(combined_df)
        
        # Shuffle
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Final statistics
        ham_count = len(combined_df[combined_df['label'] == 'ham'])
        spam_count = len(combined_df[combined_df['label'] == 'spam'])
        
        print(f"\n📊 COMPREHENSIVE DATASET SUMMARY:")
        print(f"Total samples: {len(combined_df)} (removed {before_dedup - after_dedup} duplicates)")
        print(f"Ham samples: {ham_count} ({ham_count/len(combined_df)*100:.1f}%)")
        print(f"Spam samples: {spam_count} ({spam_count/len(combined_df)*100:.1f}%)")
        print(f"Spam/Ham ratio: {spam_count/ham_count:.2f}")
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        print(f"\n✅ Comprehensive dataset saved: {output_path}")
        
        return output_path


def main():
    """Create comprehensive spam dataset"""
    
    curator = ComprehensiveSpamDatasetCurator()
    
    # Paths
    existing_dataset = "data/mega_spam_dataset.csv"
    output_dataset = "data/comprehensive_spam_dataset.csv"
    
    # Create comprehensive dataset
    dataset_path = curator.create_comprehensive_dataset(existing_dataset, output_dataset)
    
    print(f"\n🎉 Comprehensive spam dataset ready for training!")
    print(f"📁 Dataset location: {dataset_path}")
    

if __name__ == "__main__":
    main()