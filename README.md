# Medicare Fraud Detection — Thesis Project

Machine-learning and LLM-based Medicare fraud detection system using CMS Part B and Part D public datasets.

---

## Repository Structure

```
Thesis_Project/
├── ML/                  # ML notebooks, models, figures, reports
│   ├── ML_Fraud_Detection.ipynb
│   ├── ML_Fraud_Detection_1.ipynb
│   ├── ML_Fraud_Detection_2.ipynb
│   ├── ML_Fraud_Detection_3.ipynb
│   ├── models/          # Saved trained models
│   ├── figures/         # Output plots
│   └── reports/         # Model comparison CSVs
├── LLMs/                # LLM-related code
├── Data Dictionary/
└── PROJECT_ROADMAP.md
Proposal/
└── Chatbot.py
requirements.txt
```

> **Note:** The `Data set/` folder (~6.5 GB) and `NPPES dataset/` are excluded from this repo.  
> Download instructions are in the [Data Download](#data-download) section below.

---

## Setup on a New Machine

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download datasets (see below), then launch Jupyter
```bash
jupyter notebook
```

---

## Data Download

Place all downloaded files inside a `Data set/` folder at the project root.

| Dataset | Source | Local path |
|---------|--------|-----------|
| OIG Exclusions List | [OIG LEIE](https://oig.hhs.gov/exclusions/exclusions_list.asp) | `Data set/OIG.csv` |
| Medicare Part D Prescribers by Provider & Drug (2023) | [CMS Data](https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-provider-and-drug) | `Data set/Medicare Part D Prescribers - by Provider and Drug/2023/MUP_DPR_RY25_P04_V10_DY23_NPIBN.csv` |
| Medicare Part B Physician & Supplier (2023) | [CMS Data](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners/medicare-physician-other-practitioners-by-provider-and-service) | `Data set/Part B/2023/MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv` |

---

## Pre-trained Models

The `Thesis_Project/ML/models/` folder contains saved models that can be loaded directly without retraining:

- `random_forest_model.pkl` — Random Forest classifier
- `logistic_regression_model.pkl` — Logistic Regression baseline
- `catboost_model.cbm` — CatBoost classifier
- `robust_scaler.pkl` — Fitted feature scaler

---

## Requirements

- Python 3.12+
- See `requirements.txt` for full package list
