#!/usr/bin/env python3
"""
Enhanced Spam Preprocessor for Maximum Recall Across All Spam Types
- Preserves URLs, domains, email addresses (critical for phishing detection)
- Handles Unicode currency symbols (₹, €, $, £, ¥, etc.)
- Normalizes phone numbers and account patterns
- Adds special tokens: [CLS], [URL], [EMAIL], [DOMAIN], [PHONE], [MONEY]
- Designed for comprehensive spam detection (phishing, financial, romance, tech, crypto, etc.)
"""

import re
import pandas as pd
from typing import List
from urllib.parse import urlparse


class EnhancedSpamPreprocessor:
    """Advanced preprocessor for comprehensive spam detection"""
    
    def __init__(self):
        # Special tokens for spam detection
        self.special_tokens = {
            'CLS': '[CLS]',
            'PAD': '[PAD]', 
            'UNK': '[UNK]',
            'URL': '[URL]',
            'EMAIL': '[EMAIL]',
            'DOMAIN': '[DOMAIN]',
            'PHONE': '[PHONE]',
            'MONEY': '[MONEY]',
            'NUMBER': '[NUM]',
            'URGENT': '[URGENT]'
        }
        
        # Currency symbols (keep these!)
        self.currency_symbols = {'$', '₹', '€', '£', '¥', '₽', '₩', '¢', '₪', '₦', '₨', '＄'}
        
        # Urgency/spam trigger words
        self.urgency_words = {
            'urgent', 'immediately', 'asap', 'expire', 'expires', 'deadline', 
            'limited', 'hurry', 'quick', 'fast', 'now', 'today', 'tonight',
            'emergency', 'alert', 'warning', 'suspend', 'suspended', 'block', 'blocked'
        }
        
        # Money/financial terms
        self.money_terms = {
            'free', 'win', 'won', 'winner', 'prize', 'lottery', 'jackpot',
            'money', 'cash', 'income', 'earn', 'profit', 'million', 'thousand',
            'loan', 'credit', 'debt', 'investment', 'crypto', 'bitcoin'
        }
        
        # Common domains to preserve
        self.suspicious_domains = {
            'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'short.link',
            'secure-bank', 'account-verify', 'security-alert', 'paypal-secure'
        }
        
        # Stopwords (minimal - keep more words for spam detection)
        self.minimal_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been'
        }
    
    def extract_and_normalize_urls(self, text: str) -> tuple:
        """Extract URLs and normalize them while preserving spam signals"""
        urls = []
        domains = []
        normalized_text = text
        
        # Find URLs
        url_pattern = r'https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9-]+\.[a-zA-Z]{2,}[^\s]*'
        found_urls = re.findall(url_pattern, text)
        
        for url in found_urls:
            urls.append(url)
            
            # Extract domain
            try:
                if not url.startswith('http'):
                    url = 'http://' + url
                domain = urlparse(url).netloc.lower()
                if domain:
                    domains.append(domain)
                    
                    # Check for suspicious patterns
                    domain_tokens = []
                    if any(susp in domain for susp in self.suspicious_domains):
                        domain_tokens.append('[SUSPICIOUS_DOMAIN]')
                    
                    # Add domain-specific tokens
                    if 'secure' in domain or 'verify' in domain or 'account' in domain:
                        domain_tokens.append('[PHISHING_DOMAIN]')
                    
                    # Replace URL with special tokens + domain info
                    replacement = f"{self.special_tokens['URL']} {self.special_tokens['DOMAIN']}={domain}"
                    if domain_tokens:
                        replacement += ' ' + ' '.join(domain_tokens)
                    
                    normalized_text = normalized_text.replace(url, replacement, 1)
                    
            except Exception:
                # Fallback: replace with generic URL token
                normalized_text = normalized_text.replace(url, self.special_tokens['URL'], 1)
        
        return normalized_text, urls, domains
    
    def extract_and_normalize_emails(self, text: str) -> tuple:
        """Extract and normalize email addresses"""
        emails = []
        normalized_text = text
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        found_emails = re.findall(email_pattern, text)
        
        for email in found_emails:
            emails.append(email)
            domain = email.split('@')[1].lower()
            
            # Replace with email token + domain
            replacement = f"{self.special_tokens['EMAIL']} {self.special_tokens['DOMAIN']}={domain}"
            normalized_text = normalized_text.replace(email, replacement, 1)
        
        return normalized_text, emails
    
    def extract_and_normalize_phones(self, text: str) -> str:
        """Extract and normalize phone numbers"""
        # Various phone patterns
        phone_patterns = [
            r'\b\d{10,11}\b',  # 10-11 digit numbers
            r'\+\d{1,3}[\s-]?\d{3,4}[\s-]?\d{3,4}[\s-]?\d{3,4}',  # International
            r'\(\d{3}\)[\s-]?\d{3}[\s-]?\d{4}',  # (123) 456-7890
            r'\d{3}[-.]?\d{3}[-.]?\d{4}'  # 123-456-7890
        ]
        
        normalized_text = text
        for pattern in phone_patterns:
            normalized_text = re.sub(pattern, f'{self.special_tokens["PHONE"]}', normalized_text)
        
        return normalized_text
    
    def extract_and_normalize_money(self, text: str) -> str:
        """Extract and normalize money amounts while preserving currency info"""
        normalized_text = text
        
        # Money patterns with various currencies
        money_patterns = [
            r'[$₹€£¥₽₩¢]\s*[\d,]+(?:\.\d{2})?',  # $1,000.00, ₹50,000
            r'[\d,]+(?:\.\d{2})?\s*[$₹€£¥₽₩¢]',  # 1,000.00$, 50000₹
            r'\b\d+\s*(?:dollars?|rupees?|euros?|pounds?|yen)\b',  # 1000 dollars
            r'\b(?:rs|inr|usd|eur|gbp)\.?\s*[\d,]+(?:\.\d{2})?\b'  # Rs.1000, USD 50
        ]
        
        for pattern in money_patterns:
            matches = re.findall(pattern, normalized_text, re.IGNORECASE)
            for match in matches:
                # Keep currency symbol info
                currency = 'UNKNOWN'
                for symbol in self.currency_symbols:
                    if symbol in match:
                        currency = symbol
                        break
                
                replacement = f'{self.special_tokens["MONEY"]}_{currency}'
                normalized_text = normalized_text.replace(match, replacement, 1)
        
        return normalized_text
    
    def add_semantic_tokens(self, text: str) -> str:
        """Add semantic tokens for spam detection"""
        text_lower = text.lower()
        tokens_to_add = []
        
        # Check for urgency words
        if any(word in text_lower for word in self.urgency_words):
            tokens_to_add.append(self.special_tokens['URGENT'])
        
        # Check for money-related terms
        money_count = sum(1 for term in self.money_terms if term in text_lower)
        if money_count >= 2:  # Multiple money terms = likely financial spam
            tokens_to_add.append('[FINANCIAL_SPAM]')
        
        # Check for romance/relationship scams
        romance_terms = ['love', 'marry', 'relationship', 'lonely', 'partner', 'soulmate']
        if any(term in text_lower for term in romance_terms):
            tokens_to_add.append('[ROMANCE_CONTEXT]')
        
        # Check for tech support scams
        tech_terms = ['virus', 'malware', 'computer', 'microsoft', 'windows', 'support', 'technician']
        if any(term in text_lower for term in tech_terms):
            tokens_to_add.append('[TECH_SUPPORT]')
        
        # Prepend semantic tokens
        if tokens_to_add:
            return ' '.join(tokens_to_add) + ' ' + text
        return text
    
    def clean_and_normalize_text(self, text: str) -> str:
        """Main text cleaning with spam-aware normalization"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase for processing
        text = text.lower()
        
        # Step 1: Extract and normalize URLs (CRITICAL for phishing!)
        text, urls, domains = self.extract_and_normalize_urls(text)
        
        # Step 2: Extract and normalize emails
        text, emails = self.extract_and_normalize_emails(text)
        
        # Step 3: Extract and normalize phone numbers
        text = self.extract_and_normalize_phones(text)
        
        # Step 4: Extract and normalize money amounts
        text = self.extract_and_normalize_money(text)
        
        # Step 5: Preserve important punctuation and symbols
        # Keep: ! ? $ £ % + - . , : = & # @ / _ and Unicode currencies
        # This regex is much more permissive than before
        text = re.sub(r'[^a-z0-9\s!?$£₹€¥₽₩¢%+\-.,:#@/=&_\[\]]', ' ', text)
        
        # Step 6: Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Step 7: Add semantic tokens based on content
        text = self.add_semantic_tokens(text)
        
        return text
    
    def remove_minimal_stopwords(self, text: str) -> str:
        """Remove only essential stopwords (keep more for spam detection)"""
        words = text.split()
        # Keep most words - only remove very common ones that don't help spam detection
        filtered_words = [word for word in words if word not in self.minimal_stopwords or len(word) <= 2]
        return ' '.join(filtered_words)
    
    def preprocess(self, text: str) -> str:
        """Complete preprocessing pipeline optimized for spam detection"""
        # Clean and normalize
        text = self.clean_and_normalize_text(text)
        
        # Remove minimal stopwords
        text = self.remove_minimal_stopwords(text)
        
        # Add CLS token at the beginning for classification
        return f"{self.special_tokens['CLS']} {text}".strip()
    
    def get_special_tokens(self) -> dict:
        """Return special tokens for tokenizer"""
        return self.special_tokens


def test_preprocessor():
    """Test the enhanced preprocessor with spam examples"""
    preprocessor = EnhancedSpamPreprocessor()
    
    test_cases = [
        "We noticed unusual login attempts. Reset your password immediately: http://phishingsite.com",
        "Make ₹20,000 daily by working only 1 hour from home. Join now:",
        "You have won the International Lottery of $1,000,000. Send your bank details to claim immediately.",
        "URGENT! Your account will be suspended unless you verify at secure-bank-verify.com",
        "Free iPhone! Call 555-123-4567 now! Limited time offer!",
        "Your computer has virus! Call Microsoft support: +1-800-SCAMMER",
        "Hi dear, I am lonely and looking for true love. Send me $500 for visa.",
    ]
    
    print("🔍 ENHANCED PREPROCESSOR TEST:")
    print("=" * 80)
    
    for i, text in enumerate(test_cases, 1):
        processed = preprocessor.preprocess(text)
        print(f"{i}. Original: {text}")
        print(f"   Processed: {processed}")
        print()


if __name__ == "__main__":
    test_preprocessor()