# LLM Phase — Medicare Fraud Detection
## Fraud Investigator Copilot

This folder contains the complete LLM integration phase of the thesis project,
building on the ML models developed in the `ML/` folder.

---

## Notebook Execution Order

Run notebooks in sequence — each builds on outputs from the previous.

| # | Notebook | Purpose | Key Output |
|---|---|---|---|
| 1 | `LLM_1_RAG_Pipeline.ipynb` | Build the Medicare policy knowledge base | `outputs/vector_db/` |
| 2 | `LLM_2_Provider_Report_Generator.ipynb` | SHAP + GPT-4 investigation reports | `outputs/reports/` |
| 3 | `LLM_3_Fraud_Investigator_Chatbot.ipynb` | Interactive chatbot with tool use | Conversation logs |
| 4 | `LLM_4_Zero_Shot_Detection.ipynb` | LLM-only fraud detection comparison | Performance CSVs |

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements_llm.txt
```

### 2. Set your OpenAI API key
```bash
export OPENAI_API_KEY="sk-..."
```
Or set `OPENAI_API_KEY` directly in each notebook's Section 2.

---

## Architecture Overview

```
Medicare PDFs (RAG Documents)
       │
       ▼
┌─────────────────────────────────┐
│  LLM_1_RAG_Pipeline             │
│  • PDF extraction (PyPDF2)      │
│  • Token-aware chunking         │
│  • OpenAI embeddings            │
│  • FAISS vector index           │
│  • CRAG relevance filtering     │
└──────────────┬──────────────────┘
               │  vector_db/
    ┌──────────┴──────────────────────────────┐
    │                                         │
    ▼                                         ▼
┌──────────────────────┐      ┌──────────────────────────────┐
│  LLM_2_Report        │      │  LLM_3_Chatbot               │
│  Generator           │      │  • Multi-turn conversation   │
│  • SHAP values       │      │  • GPT-4 function calling    │
│  • SHAP → text       │      │  • Tools: lookup_provider,   │
│  • RAG retrieval     │      │    search_policy,            │
│  • GPT-4 reports     │      │    get_shap_explanation      │
└──────────────────────┘      └──────────────────────────────┘
    
    ┌──────────────────────────────────────┐
    │  LLM_4_Zero_Shot_Detection           │
    │  • Provider → natural language       │
    │  • Zero-shot GPT-4 classification    │
    │  • Few-shot classification           │
    │  • Policy-grounded (RAG + zero-shot) │
    │  • Comparison vs. ML baseline        │
    └──────────────────────────────────────┘
```

---

## RAG Knowledge Base

Documents indexed in the vector database:

| Source | Content |
|---|---|
| Medicare Claims Processing Manual | Billing rules, HCPCS codes, payment policies |
| Medicare Program Integrity Manual | Fraud investigation procedures, audit guidelines |
| OIG Fraud Alerts & Reports | Specific fraud patterns, exclusion criteria |

---

## Key Innovations

1. **CRAG (Corrective RAG)** — Relevance-checks retrieved chunks before passing to LLM, discarding off-topic passages (Section 9, Notebook 1)

2. **SHAP-to-Text Bridge** — Converts numerical SHAP values into natural language descriptions linked to Medicare policy terms (Section 7, Notebook 2)

3. **Agentic Tool Use** — GPT-4 autonomously calls `lookup_provider`, `search_policy`, and `get_shap_explanation` to gather evidence before answering (Section 6, Notebook 3)

4. **Policy-Grounded Zero-Shot** — Novel combination of zero-shot LLM classification + RAG policy context, enabling comparison of: LLM knowledge alone vs. LLM + Medicare policy (Section 7, Notebook 4)

---

## API Costs (Estimated)

| Notebook | Estimated Cost |
|---|---|
| LLM_1 (embeddings, ~2K–10K chunks) | $0.01–$0.05 |
| LLM_2 (20 reports, GPT-4o) | $0.50–$2.00 |
| LLM_3 (demo session, ~6 turns) | $0.10–$0.50 |
| LLM_4 (260+ providers × 3 approaches) | $1.00–$3.00 |

---

## References

- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.
- Malingu et al. (2024). Application of LLMs to Fraud Detection.
- Shuster et al. (2024). FAA Framework: LLM-Based Credit Card Fraud Investigations.
- Yan et al. (2024). Corrective Retrieval-Augmented Generation (CRAG).
