# Quick RAG Testing Scenarios

## 🚀 Quick Start Test Questions

Copy and paste these questions to test different RAG capabilities after uploading the test documents.

---

## 📁 Documents to Upload
1. `company_policies.txt` - HR and company policies
2. `product_specifications.md` - Product technical details
3. `financial_report.txt` - Q3 2024 financial data

---

## 🧪 Test Questions by Category

### 1️⃣ **Single Document Retrieval**
Test that the system can find information from specific documents:

```
Q1: How many days of annual leave do employees get?
Expected: 21 days (from company_policies.txt)

Q2: What is the price of the Enterprise tier?
Expected: $4,999/month (from product_specifications.md)

Q3: What was the Q3 2024 total revenue?
Expected: $45.7 million (from financial_report.txt)

Q4: What is the remote work policy?
Expected: Up to 3 days per week (from company_policies.txt)

Q5: What is the uptime SLA for Enterprise customers?
Expected: 99.99% (from product_specifications.md)
```

### 2️⃣ **Multi-Document Synthesis**
Test combining information from multiple documents:

```
Q1: Given our Q3 revenue and CloudSync Pro pricing, how many Enterprise tier customers would we need to reach $10M monthly recurring revenue?
Expected: Should use financial data + product pricing

Q2: What percentage of our revenue could cover the annual training budget for all employees?
Expected: Should combine employee count (423) with training budget ($2,500) and revenue ($45.7M)

Q3: How does our employee headcount compare to our customer base?
Expected: 423 employees serving 1,247 customers (from financial_report.txt)
```

### 3️⃣ **Complex Analytical Questions**
Test the system's ability to analyze and provide insights:

```
Q1: Based on the financial performance, what are the top 3 strengths of the business?
Expected: Should analyze metrics like 72% gross margin, 94% renewal rate, 112% net revenue retention

Q2: What security certifications does CloudSync Pro have and what is planned?
Expected: Current: SOC2 Type II, GDPR, HIPAA ready; Planned: Complete SOC2 Type II in Q4

Q3: Calculate the average revenue per employee based on Q3 data.
Expected: $45.7M / 423 employees = ~$108,000 per employee
```

### 4️⃣ **Comparison and Context Questions**
Test hybrid mode (document + general knowledge):

```
Q1: How does our 72% gross margin compare to typical SaaS companies?
Expected: Should cite our 72% margin and compare to industry standards (typically 60-80%)

Q2: Is our customer churn rate of 5.2% good for a B2B SaaS company?
Expected: Should reference our rate and compare to industry benchmarks

Q3: Based on our benefits package, how competitive are we in the tech industry?
Expected: Should list our benefits and provide market context
```

### 5️⃣ **Specific Data Extraction**
Test precise information retrieval:

```
Q1: List all the pricing tiers and their features.
Expected: Starter ($499), Professional ($1,999), Enterprise ($4,999), Custom

Q2: What are all the types of leave available to employees?
Expected: Annual (21 days), Sick (10 days), Parental (12/4 weeks), Bereavement (5 days), Personal (3 days)

Q3: Break down the Q3 revenue by product line.
Expected: CloudSync Pro: $27.4M, DataBridge: $13.7M, SecureTransfer: $4.6M
```

### 6️⃣ **Time-Sensitive and Forward-Looking**
Test understanding of temporal information:

```
Q1: What are the key initiatives planned for Q4 2024?
Expected: SOC2 certification, partner program, version 3.0, expand sales by 20%

Q2: When are performance reviews conducted?
Expected: Bi-annually in June and December

Q3: What is the revenue projection for Q4?
Expected: $48-50 million
```

### 7️⃣ **Error Handling and Edge Cases**
Test system behavior with challenging queries:

```
Q1: What is the recipe for chocolate cake?
Expected: No relevant information in documents

Q2: Tell me about vacation policy.
Expected: Should interpret as annual leave policy

Q3: What is the policy?
Expected: Should ask for clarification or list available policies
```

---

## 🎯 System Prompts to Test

### For Strict RAG Mode:
```
You are a document assistant. Only answer based on the uploaded documents. If information is not in the documents, clearly state that. Always cite your sources.
```

### For Hybrid Mode:
```
You are a business analyst with access to company documents. Combine document information with your general expertise to provide comprehensive insights. Clearly distinguish between facts from documents and general knowledge.
```

### For Comparative Analysis:
```
You are a strategic advisor. When answering questions, first provide specific information from the documents, then add industry context and best practices for comparison.
```

---

## 📊 Initial Knowledge Context Examples

### Example 1 - Company Context:
```
Company: ACME Corporation
Industry: B2B SaaS, Data Integration
Founded: 2018
Headquarters: San Francisco, CA
Competitors: Informatica, Talend, MuleSoft
Market Position: Challenger in mid-market segment
```

### Example 2 - Financial Context:
```
Fiscal Year: January to December
Reporting Currency: USD
Accounting Standard: GAAP
Auditor: Deloitte
Stock Symbol: Private (Series C)
Target IPO: 2025-2026
```

### Example 3 - Strategic Context:
```
Current Focus: Enterprise market expansion
Key Differentiator: Real-time sync with 11 nines durability
Target Market: Companies with 500-5000 employees
Go-to-Market: Direct sales + channel partners
Technology Stack: Cloud-native, microservices architecture
```

---

## ✅ Quick Validation Checklist

After running these tests, verify:

- [ ] Accurate retrieval from single documents
- [ ] Successful multi-document synthesis
- [ ] Proper source citations displayed
- [ ] Similarity scores shown for chunks
- [ ] Hybrid responses blend document + general knowledge
- [ ] System handles queries with no relevant documents
- [ ] Token counting is accurate
- [ ] Response time is acceptable (< 5 seconds)
- [ ] Different embedding models work correctly
- [ ] RAG parameters (Top K, threshold) affect results

---

## 💡 Tips for Effective Testing

1. **Start Simple**: Begin with direct questions that have clear answers in documents
2. **Progress to Complex**: Move to analytical and synthesis questions
3. **Test Edge Cases**: Try ambiguous or irrelevant queries
4. **Vary Parameters**: Change Top K (1-10) and Similarity Threshold (0.1-0.8)
5. **Compare Modes**: Ask same question with/without documents
6. **Monitor Performance**: Track tokens used and response time
7. **Document Issues**: Note any retrieval failures or incorrect answers

---

## 🔄 Iterative Testing Flow

1. **Upload one document** → Test single doc retrieval
2. **Add second document** → Test multi-doc capability
3. **Add third document** → Test scaling and performance
4. **Clear all** → Test direct LLM mode
5. **Re-upload all** → Test full RAG capabilities
6. **Adjust parameters** → Find optimal settings
7. **Change prompts** → Test different assistant personalities