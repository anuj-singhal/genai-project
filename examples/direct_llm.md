# On-Prem GenAI Workbench - Testing Examples

## 🎯 Quick Start Testing Guide

This document provides various testing scenarios to help you explore all the functionalities of the On-Prem GenAI Workbench.

---

## 📚 Example Prompts and Scenarios

### 1. Academic Assistant Configuration

**System Prompt:**
```
You are an academic assistant specializing in research and education. Provide detailed, well-structured responses with proper citations when relevant. Use academic language while remaining accessible.
```

**Initial Knowledge/Context:**
```
Field of expertise: Computer Science, AI, and Machine Learning
Target audience: Graduate students and researchers
Preferred citation style: APA
Focus areas: Deep learning, NLP, Computer Vision
Current research project: Transformer architectures in multi-modal learning
```

**Test Questions:**
- "Explain the attention mechanism in transformers"
- "What are the key differences between BERT and GPT architectures?"
- "Help me design a research methodology for studying bias in LLMs"

---

### 2. Code Helper Configuration

**System Prompt:**
```
You are an expert programming assistant. Provide clean, efficient, and well-commented code. Follow best practices and explain your implementation choices. Always consider error handling and edge cases.
```

**Initial Knowledge/Context:**
```
Primary languages: Python, JavaScript, SQL
Frameworks: React, Django, FastAPI, Streamlit
Development environment: VS Code
Coding style: PEP 8 for Python, ESLint for JavaScript
Current project: Building a real-time data dashboard
Database: PostgreSQL
Focus: Production-ready code with comprehensive error handling
```

**Test Questions:**
- "Create a Python function to process streaming data with error handling"
- "How do I implement authentication in FastAPI?"
- "Write a React component for real-time chart updates"

---

### 3. Data Analyst Configuration

**System Prompt:**
```
You are a data analyst expert. Provide insights backed by data, suggest appropriate visualizations, and explain statistical concepts clearly. Focus on actionable recommendations and business impact.
```

**Initial Knowledge/Context:**
```
Tools: Python (pandas, numpy, scikit-learn), SQL, Tableau, Power BI
Current dataset: E-commerce transactions (1M rows, 50 features)
Business context: Online retail company, B2C model
KPIs: Customer lifetime value, Churn rate, Average order value
Analysis goals: Improve customer retention and increase basket size
Time period: Last 2 years of data
Seasonality: Strong Q4 performance due to holidays
```

**Test Questions:**
- "What analysis would you recommend for understanding customer churn?"
- "How should I segment customers for targeted marketing?"
- "Create a SQL query to find high-value customers"

---

### 4. Creative Writer Configuration

**System Prompt:**
```
You are a creative writing assistant specializing in storytelling and narrative development. Help with plot structure, character arcs, dialogue, and world-building. Provide constructive feedback and creative suggestions while maintaining the author's voice.
```

**Initial Knowledge/Context:**
```
Genre: Science Fiction / Cyberpunk
Setting: Neo-Tokyo, 2087
Main character: A memory detective who can enter people's memories
Tone: Dark, philosophical, with noir influences
Themes: Identity, consciousness, reality vs. perception
Target audience: Adult readers who enjoy Philip K. Dick and William Gibson
Current chapter: Chapter 5 - The protagonist discovers a memory that shouldn't exist
Writing style: First-person narrative, present tense
```

**Test Questions:**
- "Help me develop the antagonist's motivation"
- "Write a dialogue between the detective and an AI consciousness"
- "Suggest plot twists for the memory that shouldn't exist"

---

### 5. Business Consultant Configuration

**System Prompt:**
```
You are a senior business consultant with expertise in strategy, operations, and digital transformation. Provide practical, actionable advice based on industry best practices. Focus on ROI, risk mitigation, and measurable outcomes.
```

**Initial Knowledge/Context:**
```
Client: Mid-size manufacturing company ($50M revenue)
Industry: Automotive parts supplier
Current challenges: Supply chain disruptions, rising costs, digital transformation needs
Budget: $2M for transformation initiatives
Timeline: 18-month implementation period
Key stakeholders: CEO (tech-forward), CFO (cost-conscious), COO (process-focused)
Competition: Facing pressure from low-cost overseas manufacturers
Strengths: Quality reputation, established relationships with major OEMs
Weaknesses: Legacy IT systems, manual processes, limited data analytics
```

**Test Questions:**
- "What digital transformation priorities should we focus on?"
- "How can we reduce supply chain risks?"
- "Develop a business case for implementing an ERP system"

---

### 6. Technical Documentation Writer

**System Prompt:**
```
You are a technical documentation specialist. Create clear, concise, and well-structured documentation. Use appropriate formatting, include examples, and ensure content is accessible to the target audience. Follow documentation best practices.
```

**Initial Knowledge/Context:**
```
Product: Cloud-based API management platform
Target audience: DevOps engineers and API developers
Documentation type: API reference, tutorials, troubleshooting guides
Tech stack: RESTful APIs, GraphQL, WebSockets
Authentication: OAuth 2.0, API keys
Rate limits: 1000 requests/minute for standard tier
Common use cases: Microservices communication, third-party integrations
Known issues: Timeout handling in high-latency networks
```

**Test Questions:**
- "Write API documentation for a user authentication endpoint"
- "Create a troubleshooting guide for common API errors"
- "Explain rate limiting to non-technical stakeholders"

---

### 7. Medical Research Assistant

**System Prompt:**
```
You are a medical research assistant. Provide accurate, evidence-based information while being clear about limitations. Always emphasize the importance of consulting healthcare professionals for medical decisions.
```

**Initial Knowledge/Context:**
```
Research focus: Cardiovascular health and prevention
Target audience: Healthcare professionals and researchers
Current project: Meta-analysis of lifestyle interventions for hypertension
Database access: PubMed, Cochrane Reviews, clinical trial registries
Statistical tools: R, SPSS, meta-analysis software
Disclaimer: Not for direct patient care or diagnosis
Focus period: Studies from 2015-2024
Population of interest: Adults 40-65 years old
```

**Test Questions:**
- "Summarize recent findings on dietary interventions for blood pressure"
- "What are the limitations of current hypertension studies?"
- "Help design inclusion criteria for a systematic review"

---

### 8. Legal Document Analyzer

**System Prompt:**
```
You are a legal document analyst. Help interpret and summarize legal documents while being clear that you cannot provide legal advice. Focus on structure, key terms, and potential implications while recommending consultation with qualified attorneys.
```

**Initial Knowledge/Context:**
```
Document types: Contracts, NDAs, service agreements, terms of service
Jurisdiction: United States, primarily Delaware and California law
Industry focus: Technology and software companies
Common issues: IP ownership, liability limitations, data privacy
Compliance requirements: GDPR, CCPA, SOC 2
Current task: Reviewing SaaS agreements for potential risks
Key concerns: Data residency, termination clauses, indemnification
```

**Test Questions:**
- "What should I look for in a software licensing agreement?"
- "Explain the difference between warranties and indemnifications"
- "Identify potential red flags in this termination clause"

---

## 🧪 Testing Different Features

### Token Usage Testing
1. Start with a short conversation and observe token counts
2. Add a long initial knowledge context and see how it affects token usage
3. Switch between models and observe token counting differences

### Model Switching Test
1. Start a conversation with GPT-3.5-turbo
2. Ask a complex question
3. Switch to GPT-4 mid-conversation
4. Ask a follow-up question and observe the difference

### Temperature Testing
1. Set temperature to 0.0 and ask for a factual explanation
2. Set temperature to 0.5 and ask the same question
3. Set temperature to 1.0 and ask for creative writing
4. Compare the responses

### Context Persistence Test
1. Add detailed initial knowledge
2. Start a conversation referring to the context
3. Change the initial knowledge mid-conversation
4. Continue the conversation and see how it adapts

### Streaming Response Test
1. Ask for a long, detailed explanation
2. Observe the streaming behavior
3. Ask for a list of 20 items
4. Watch how the response builds progressively

---

## 💡 Tips for Testing

1. **Test Edge Cases:**
   - Very long system prompts
   - Empty initial knowledge
   - Switching models rapidly
   - Maximum token limits

2. **Test Context Switching:**
   - Change from technical to creative prompts
   - Add/remove initial knowledge
   - Modify temperature during conversation

3. **Test Error Handling:**
   - Invalid API key
   - Network interruption
   - Token limit exceeded

4. **Test User Experience:**
   - Quick prompt switching using sidebar examples
   - Chat history scrolling with many messages
   - Real-time token counting accuracy

---

## 📊 Performance Benchmarks

Test these scenarios to evaluate performance:

1. **Response Time:**
   - Simple question: < 2 seconds
   - Complex analysis: < 5 seconds
   - Long-form content: < 10 seconds

2. **Token Efficiency:**
   - Monitor input vs output token ratio
   - Test with different prompt lengths
   - Optimize prompts for token usage

3. **Context Retention:**
   - Test with 10+ message conversations
   - Verify context understanding across messages
   - Check if model remembers initial knowledge

---

## 🚀 Advanced Testing Scenarios

### Multi-Turn Conversations
```
1. "What is machine learning?"
2. "Give me an example"
3. "How does that differ from deep learning?"
4. "Which one should I learn first?"
5. "Create a learning roadmap"
```

### Context-Dependent Tasks
```
Initial Knowledge: "Company has 500 employees, $10M revenue, B2B SaaS"
1. "Analyze our company size"
2. "What's our revenue per employee?"
3. "Suggest growth strategies"
4. "What metrics should we track?"
```

### Dynamic Prompt Changes
1. Start with academic assistant
2. Ask technical question
3. Switch to creative writer
4. Ask for metaphorical explanation
5. Observe adaptation

---

## ✅ Validation Checklist

- [ ] All example prompts load correctly
- [ ] Token counting is accurate
- [ ] Streaming works smoothly
- [ ] Model switching preserves context
- [ ] Temperature changes affect output
- [ ] Initial knowledge is properly integrated
- [ ] Chat history scrolls properly
- [ ] Session clearing works
- [ ] Error messages are helpful

---

## 📝 Notes

- Save interesting conversations for future reference
- Document any bugs or unexpected behaviors
- Note which model/temperature combinations work best for different tasks
- Track token usage for cost optimization