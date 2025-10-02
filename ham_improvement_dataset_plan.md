# HAM-Focused Dataset Plan for BalancedSpamNet Improvement

## Current Issues:
- Banking Legitimate: 25% accuracy (too aggressive)
- E-commerce Legitimate: 25% accuracy (flags orders as spam)  
- Business Professional: 75% accuracy (some false positives)

## Recommended HAM Categories to Add (5,000+ examples each):

### 1. **Banking & Financial Legitimate** 
```
- Account balance notifications: "Balance: $1,234.56 as of Jan 15"
- Transaction confirmations: "$500 debited from account ending 1234"
- ATM withdrawal alerts: "Cash withdrawal of $100 at Main St ATM"
- Payment confirmations: "Payment of $1,200 processed successfully"
- Monthly statements: "Statement available for December 2024"
- Customer service: "Dear valued customer, your account..."
```

### 2. **E-commerce Legitimate**
```
- Order confirmations: "Order #123456 confirmed - iPhone 15 Pro"
- Shipping notifications: "Your package has been shipped"
- Delivery updates: "Package delivered to your address"
- Return confirmations: "Return processed - refund in 3-5 days"
- Receipt emails: "Thank you for your purchase at TechStore"
- Review requests: "How was your recent purchase?"
```

### 3. **Business Professional**
```
- Meeting invitations: "Meeting scheduled for 2 PM tomorrow"
- Project updates: "Deadline extended to March 15th"
- Expense reports: "Your expense report has been approved"
- Client communications: "Proposal meeting went well"
- Team notifications: "New project assignment for Q1"
- HR communications: "Benefits enrollment period starts"
```

### 4. **Modern Communication Patterns**
```
- Automated notifications from real services
- Two-factor authentication codes
- Password reset confirmations
- Account verification messages
- Service updates and maintenance notices
- Educational institution communications
```

## Dataset Sources to Consider:

### **Option A: Create Synthetic Dataset** (Recommended)
- Generate 10,000+ realistic HAM examples per category
- Use templates based on real-world patterns
- Include legitimate financial/business language
- Vary sentence structures and terminology

### **Option B: Collect Real-World Data**
- Parse real email inboxes (anonymized)
- Corporate email datasets
- Customer service communication logs
- Banking notification samples

### **Option C: Augment Existing Dataset**
- Take current comprehensive_spam_dataset.csv 
- Filter and balance HAM examples
- Add 50,000+ diverse HAM samples
- Maintain SPAM/HAM ratio at 50:50

## Training Strategy:

### **Fine-tuning Approach** (Faster)
1. Load current BalancedSpamNet model
2. Train only on HAM categories with low accuracy
3. Use lower learning rate (1e-5)
4. Focus on Banking, E-commerce, Business categories
5. 5-10 epochs maximum

### **Full Retraining** (Better Results)
1. Combine original dataset + new HAM examples
2. Retrain from scratch with balanced sampling
3. Use same BalancedSpamNet architecture
4. Focus on balanced accuracy metrics

## Success Metrics:
- Banking Legitimate: 25% → 80%+ accuracy
- E-commerce Legitimate: 25% → 85%+ accuracy  
- Business Professional: 75% → 90%+ accuracy
- **Maintain 100% SPAM detection** (or accept 95%+)
- Overall accuracy: 80% → 90%+

## Implementation Timeline:
- **Week 1:** Create HAM-focused dataset (10,000+ examples)
- **Week 2:** Fine-tune model with new data
- **Week 3:** Test and validate improvements
- **Week 4:** Deploy improved model

Would you like me to create the HAM-focused training dataset or help with the fine-tuning process?