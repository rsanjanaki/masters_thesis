# Medicare Fraud Detection — Master's Thesis

Machine-learning and LLM-based Medicare fraud detection and investigation system using CMS Part B and Part D public datasets.

---

## Repository Structure

```
masters_thesis_repo/
├── notebooks/
│   ├── LLMs/                                      # LLM phase — RAG, report generation, chatbot
│   │   ├── LLM_1_RAG_Pipeline.ipynb               # PDF extraction → chunking → FAISS index + CRAG
│   │   ├── LLM_2_Provider_Report_Generator.ipynb  # SHAP + RAG → GPT-4 investigation reports
│   │   ├── LLM_3_Fraud_Investigator_Chatbot.ipynb # Agentic multi-turn chatbot (tool-calling)
│   │   ├── streamlit_chatbot.py                   # Streamlit web UI for the chatbot
│   │   ├── review_flagged_providers_analysis.py   # RFP analysis module
│   │   ├── INVESTIGATIVE_TRIAGE_DESIGN.md         # Triage design document
│   │   ├── requirements_llm.txt                   # LLM-specific dependencies
│   │   └── README.md                              # LLM phase README
│   └── ML/                                        # ML phase — fraud detection models
│       ├── ML_Fraud_Detection_AllFeatures_NoSampling.ipynb
│       ├── ML_Fraud_Detection_AllFeatures_SMOTE.ipynb
│       ├── ML_Fraud_Detection_AllFeatures_SMOTEENN.ipynb
│       ├── ML_Fraud_Detection_NonCorrelatedFeatures_NoSampling.ipynb
│       ├── ML_Fraud_Detection_NonCorrelatedFeatures_SMOTEENN.ipynb
│       └── ML_Fraud_Detection_ NonCorrelatedFeatures_SMOTE.ipynb
├── docs/
│   └── thesis_defense.pptx                        # Defense presentation slides
├── Proposal/                                      # Proposal documents
├── Data Dictionary/                               # CMS Part B & Part D data dictionaries
├── PROJECT_ROADMAP.md
├── requirements.txt
└── README.md
```

> **Note:** Data files (`.csv`, `.xlsx`), trained model weights, and FAISS vector databases are **not tracked** in this repo.  
> Download instructions are in the [Data Download](#data-download) section below.

---

## Project Overview

### Phase 1 — Machine Learning Fraud Detection
Six notebooks covering four experimental configurations:

| Notebook | Features | Sampling |
|---|---|---|
| `ML_Fraud_Detection_AllFeatures_NoSampling` | All features | None |
| `ML_Fraud_Detection_AllFeatures_SMOTE` | All features | SMOTE |
| `ML_Fraud_Detection_AllFeatures_SMOTEENN` | All features | SMOTE-ENN |
| `ML_Fraud_Detection_NonCorrelatedFeatures_NoSampling` | Uncorrelated subset | None |
| `ML_Fraud_Detection_ NonCorrelatedFeatures_SMOTE` | Uncorrelated subset | SMOTE |
| `ML_Fraud_Detection_NonCorrelatedFeatures_SMOTEENN` | Uncorrelated subset | SMOTE-ENN |

**Best model:** CalibratedClassifierCV (LinearSVC, SMOTE-ENN) — Recall = 83.3% on test set (45/54 fraud cases caught, threshold = 0.0702).

### Phase 2 — LLM Investigation System
Three notebooks implementing an end-to-end agentic fraud investigation platform:

| Notebook | Purpose |
|---|---|
| `LLM_1_RAG_Pipeline` | PDF extraction → 512-token chunks → OpenAI embeddings → FAISS + CRAG |
| `LLM_2_Provider_Report_Generator` | Per-provider GPT-4 investigation reports grounded in SHAP + RAG |
| `LLM_3_Fraud_Investigator_Chatbot` | Multi-turn chatbot with four agentic tools (ReAct loop, Claude haiku-4-5) |

**Chatbot tools:** `lookup_provider` · `search_policy` · `get_shap_explanation` · `analyze_review_flagged_providers`

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/rsanjanaki/masters_thesis.git
cd masters_thesis
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
# ML phase
pip install -r requirements.txt

# LLM phase (additional packages)
pip install -r notebooks/LLMs/requirements_llm.txt
```

### 4. Set up API keys
Create a `.env` file in `notebooks/LLMs/`:
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Download datasets (see below), then launch Jupyter
```bash
jupyter notebook
```

### 6. Launch the Streamlit chatbot
```bash
cd notebooks/LLMs
streamlit run streamlit_chatbot.py
```

---

## Data Download

Place downloaded files in a local `Data set/` folder (not tracked by git).

| Dataset | Source | Local path |
|---|---|---|
| OIG Exclusions List (LEIE) | [OIG LEIE](https://oig.hhs.gov/exclusions/exclusions_list.asp) | `Data set/OIG.csv` |
| Medicare Part D Prescribers by Provider & Drug (2023) | [CMS Data](https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-provider-and-drug) | `Data set/Medicare Part D.../MUP_DPR_RY25_P04_V10_DY23_NPIBN.csv` |
| Medicare Part B Physician & Supplier (2023) | [CMS Data](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider-and-service) | `Data set/Part B/2023/MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv` |
| CMS/OIG Policy PDFs (for RAG) | CMS Medicare Manuals + OIG Fraud Alerts | `RAG documents/` |

---

## Requirements

- Python 3.12+
- See `requirements.txt` for ML packages
- See `notebooks/LLMs/requirements_llm.txt` for LLM packages
