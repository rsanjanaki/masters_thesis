"""
AI-Powered Medicare Fraud Investigation Platform
Professional healthcare fraud analytics system powered by machine learning and LLMs.

Run:  streamlit run streamlit_chatbot.py
"""

import os
import io
import json
import pickle
import warnings
import sys
import re
import contextlib
from pathlib import Path
from typing import List, Dict

# Suppress tqdm/transformers progress bars (prevents BrokenPipeError in Streamlit)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import pandas as pd
import faiss
import shap
import joblib
import streamlit as st
from dotenv import load_dotenv
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import RobustScaler

# Import RFP analysis module
sys.path.append(str(Path(__file__).parent))
from review_flagged_providers_analysis import ReviewFlaggedProvidersAnalyzer

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fraud Investigation Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for professional dark theme ────────────────────────────────────
st.markdown("""
<style>
    /* Main theme */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1419 100%);
        border-right: 1px solid #2d3748;
    }
    
    /* KPI Card styling */
    .kpi-card {
        background: linear-gradient(135deg, #1e2433 0%, #2d3748 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #3d4a5c;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3), 0 0 20px rgba(99, 102, 241, 0.1);
        transition: all 0.3s ease;
        margin-bottom: 10px;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4), 0 0 30px rgba(99, 102, 241, 0.2);
        border-color: #6366f1;
    }
    
    .kpi-number {
        font-size: 32px;
        font-weight: 700;
        color: #f8fafc;
        margin: 5px 0;
    }
    
    .kpi-label {
        font-size: 14px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    
    .kpi-icon {
        font-size: 24px;
        margin-bottom: 10px;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background: linear-gradient(135deg, #1e2433 0%, #2d3748 100%);
        border: 1px solid #3d4a5c;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(99, 102, 241, 0.3);
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 100%);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.5);
        transform: translateY(-1px);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a1f2e;
        border-radius: 8px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        color: #94a3b8;
        font-weight: 500;
        padding: 8px 16px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 24px;
        color: #f8fafc;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-weight: 500;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1e2433;
        border-radius: 8px;
        border: 1px solid #3d4a5c;
    }
    
    /* Alert styling */
    .stAlert {
        background-color: #1e2433;
        border: 1px solid #3d4a5c;
        border-radius: 8px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-weight: 700;
    }
    
    /* Divider */
    hr {
        border-color: #3d4a5c;
    }
</style>
""", unsafe_allow_html=True)

# ── Paths (identical to notebook) ────────────────────────────────────────────
ENV_FILE       = Path("/Users/roya/Desktop/Thesis/Thesis_Project/LLMs/ATT18160.env")
ML_DIR         = Path("/Users/roya/Desktop/Thesis/Thesis_Project/ML")
LLM_OUTPUT_DIR = Path("/Users/roya/Desktop/Thesis/Thesis_Project/LLMs/outputs")
VECTOR_DB_DIR  = LLM_OUTPUT_DIR / "vector_db"
MODELS_DIR     = ML_DIR / "models"
DATASETS_DIR   = ML_DIR / "datasets"

CLAUDE_MODEL     = "claude-sonnet-4-6"
EMBED_MODEL      = "all-MiniLM-L6-v2"
POLICY_MIN_SCORE = 0.30

# ── Feature descriptions ──────────────────────────────────────────────────────
FEATURE_DESCRIPTIONS = {
    "Total_Services_B":           "total number of Medicare Part B services billed",
    "Total_Beneficiaries_B":      "total unique Medicare beneficiaries served (Part B)",
    "Avg_Submitted_Charge_B":     "average submitted charge per Part B service",
    "Max_Submitted_Charge_B":     "maximum submitted charge on a single Part B claim",
    "Avg_Medicare_Allowed_B":     "average Medicare-allowed payment per Part B service",
    "Max_Medicare_Allowed_B":     "highest Medicare-allowed payment on a single Part B service",
    "Weighted_Avg_Payment_B":     "weighted average Medicare payment per Part B service",
    "Unique_HCPCS_B":             "number of distinct HCPCS procedure codes billed (Part B)",
    "Service_Type_Diversity_B":   "variety of service categories billed (Part B)",
    "Unique_Service_Places_B":    "number of distinct places of service where claims were filed",
    "Total_Claims_D":             "total Medicare Part D prescription drug claims",
    "Total_Drug_Cost_D":          "total cost of all Part D drugs prescribed",
    "Avg_Drug_Cost_Per_Claim_D":  "average cost per Part D prescription claim",
    "Unique_Brand_Drugs_D":       "number of unique brand-name drugs prescribed (Part D)",
    "Charge_to_Allowed_Ratio":    "ratio of submitted charges to Medicare-allowed amounts (markup ratio)",
    "Payment_to_Allowed_Ratio":   "ratio of actual payment to allowed amount",
    "MaxCharge_Ratio":            "ratio of maximum charge to the average charge",
    "Payment_Efficiency":         "ratio of Medicare payment to submitted charges",
    "Services_per_Beneficiary_B": "average number of Part B services performed per beneficiary",
    "BeneDay_Per_Beneficiary_B":  "average Medicare beneficiary-days per patient",
    "Services_Per_HCPCS":         "average number of services billed per procedure code",
    "Has_PartD":                  "indicator: provider also prescribes Part D drugs",
    "Has_Both":                   "indicator: provider bills both Part B and Part D",
    "Drug_Cost_Per_Day":          "average daily drug cost per beneficiary",
    "Service_to_Drug_Ratio":      "ratio of Part B services to Part D drug claims",
    "High_Service_Volume_Flag":   "flag: provider is in the top 10% for service volume",
    "High_Drug_Cost_Flag":        "flag: provider is in the top 10% for drug costs",
    "High_Payment_Flag":          "flag: provider is in the top 10% for Medicare payments",
    "High_Service_High_Payment":  "flag: high volume AND high payment (joint risk indicator)",
    "Charge_x_Volume":            "product of charge ratio and service volume (compounded risk)",
    "Z_Total_Services_B":         "standardized z-score for total Part B services vs. peers",
    "Z_Total_Drug_Cost_D":        "standardized z-score for total Part D drug costs vs. peers",
    "Z_Avg_Medicare_Payment_B":   "standardized z-score for average Medicare payment vs. peers",
    "RUCA_Code":                  "Rural-Urban Commuting Area (RUCA) code for provider location",
    "Total_BeneDay_Services_B":   "total Medicare beneficiary-day services (Part B)",
    "log_Total_Services_B":       "log-transformed total Part B services",
    "log_Total_Claims_D":         "log-transformed total Part D claims",
    "Fills_Per_Claim_D":          "average drug fills per Part D claim",
    "Drug_Cost_Per_Claim_D":      "average drug cost per Part D claim",
}

SYSTEM_PROMPT = """You are an AI Fraud Analyst for Medicare fraud investigators.

You help investigators:
1. Understand why specific providers were flagged as high-risk by the machine learning model
2. Find relevant Medicare policy and OIG guidance for billing behaviors of concern
3. Interpret SHAP explanations in plain language tied to regulatory context
4. Identify what evidence would confirm or rule out fraud
5. Suggest investigation steps grounded in official Medicare guidelines
6. Analyze review-flagged providers (high model scores but non-fraud labels) for investigation prioritization

When answering:
- ALWAYS use your tools to look up specific provider data and policy evidence before drawing conclusions
- Ground every claim in retrieved policy excerpts — cite the source (manual name and file)
- Be concise and objective; do not speculate beyond what the data shows
- Acknowledge uncertainty when the evidence is incomplete
- Use numbered lists for recommendations and action items

Available tools: lookup_provider, search_policy, get_shap_explanation, analyze_review_flagged_providers
"""

# ── Helper function for KPI cards ─────────────────────────────────────────────
def render_kpi_card(icon, number, label):
    """Render a professional KPI card with icon, number, and label."""
    return f"""
    <div class="kpi-card">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-number">{number}</div>
        <div class="kpi-label">{label}</div>
    </div>
    """

TOOLS = [
    {
        "name": "lookup_provider",
        "description": (
            "Look up a Medicare provider by their numeric ID. "
            "Returns fraud probability, flag status, and top SHAP risk factors."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "provider_id": {
                    "type": "integer",
                    "description": "The numeric provider index from the Medicare test dataset",
                }
            },
            "required": ["provider_id"],
        },
    },
    {
        "name": "search_policy",
        "description": (
            "Search the Medicare policy knowledge base (Medicare Claims Processing Manual, "
            "Program Integrity Manual, and OIG Fraud Alerts) for guidance "
            "relevant to a specific fraud indicator or billing behavior."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query about Medicare fraud or billing rules",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of policy passages to retrieve (default: 4)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_shap_explanation",
        "description": (
            "Get a detailed SHAP (Shapley Additive Explanations) breakdown for a provider. "
            "Shows which features most strongly drive the fraud risk prediction and in what direction."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "provider_id": {
                    "type": "integer",
                    "description": "The numeric provider index",
                }
            },
            "required": ["provider_id"],
        },
    },
    {
        "name": "analyze_review_flagged_providers",
        "description": (
            "Analyze providers flagged at very high fraud risk (≥90%) but NOT labeled as fraud. "
            "These may represent undetected fraud, emerging patterns, or high-risk edge cases requiring "
            "human judgment. Returns investigation priorities, pattern analysis, and comparison to confirmed fraud."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "enum": ["summary", "top_providers", "pattern_comparison", "specific_provider"],
                    "description": (
                        "Type of analysis: summary=group statistics | top_providers=ranked list | "
                        "pattern_comparison=RFP vs fraud features | specific_provider=detailed analysis"
                    ),
                },
                "top_n": {
                    "type": "integer",
                    "description": "Number of top providers to return (for top_providers mode, default: 20)",
                },
                "provider_id": {
                    "type": "integer",
                    "description": "Specific provider ID to analyze (for specific_provider mode)",
                },
            },
            "required": ["analysis_type"],
        },
    },
]


# ── Cache expensive resources ─────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base and ML model…")
def load_resources():
    load_dotenv(ENV_FILE)
    client   = Anthropic(api_key=os.environ.get("OPENAI_API_KEY"))

    # Silence ALL stdout/stderr during heavy loading to prevent BrokenPipeError
    # (Streamlit replaces sys.stdout with an internal pipe that breaks during cache loading)
    import logging, contextlib, io
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    import transformers
    transformers.utils.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()

    _null = io.StringIO()
    with contextlib.redirect_stdout(_null), contextlib.redirect_stderr(_null):
        st_model = SentenceTransformer(EMBED_MODEL)

    # FAISS + chunks
    rag_index = faiss.read_index(str(VECTOR_DB_DIR / "medicare_policy.faiss"))
    with open(VECTOR_DB_DIR / "chunk_texts.pkl", "rb") as f:
        chunk_texts = pickle.load(f)
    with open(VECTOR_DB_DIR / "chunk_metadata.pkl", "rb") as f:
        chunk_metadata = pickle.load(f)

    # ML datasets
    X_test  = pd.read_csv(DATASETS_DIR / "X_test_reduced.csv",  index_col="original_index")
    y_test  = pd.read_csv(DATASETS_DIR / "y_test.csv",          index_col="original_index").squeeze()
    X_train = pd.read_csv(DATASETS_DIR / "X_train_reduced.csv", index_col="original_index")
    y_train = pd.read_csv(DATASETS_DIR / "y_train.csv",         index_col="original_index").squeeze()

    # Balanced data
    bal_path = DATASETS_DIR / "X_train_balanced.csv"
    if bal_path.exists():
        X_train_balanced = pd.read_csv(bal_path, index_col="balanced_index")
        y_train_balanced = pd.read_csv(
            DATASETS_DIR / "y_train_balanced.csv", index_col="balanced_index"
        ).squeeze()
    else:
        X_train_balanced = None
        y_train_balanced = None

    # Model + scaler + params
    model      = joblib.load(MODELS_DIR / "svm_tuned_model.joblib")
    svm_params = joblib.load(MODELS_DIR / "svm_tuned_params.joblib")
    scaler_path = MODELS_DIR / "svm_scaler.joblib"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    else:
        scaler = RobustScaler()
        scaler.fit(X_train)

    X_test_scaled = scaler.transform(X_test)
    y_proba       = model.predict_proba(X_test_scaled)[:, 1]

    # SHAP
    if X_train_balanced is not None:
        X_shap_train = scaler.transform(X_train_balanced)
        y_shap_train = y_train_balanced
    else:
        X_shap_train = scaler.transform(X_train)
        y_shap_train = y_train

    svm_for_shap = LinearSVC(
        C=svm_params["C"],
        loss="squared_hinge",
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )
    svm_for_shap.fit(X_shap_train, y_shap_train)
    explainer   = shap.LinearExplainer(svm_for_shap, X_shap_train)
    shap_values = explainer.shap_values(X_test_scaled)
    shap_fraud  = shap_values

    detection_threshold = svm_params["threshold"]
    provider_db = X_test.copy()
    provider_db["fraud_prob"] = y_proba
    provider_db["true_label"] = y_test.values
    provider_db["flagged"]    = (y_proba >= detection_threshold).astype(int)


    # Initialize RFP analyzer (wrapped to suppress print() calls hitting Streamlit's pipe)
    _null2 = io.StringIO()
    with contextlib.redirect_stdout(_null2), contextlib.redirect_stderr(_null2):
        analyzer = ReviewFlaggedProvidersAnalyzer(
            X=pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index),
            y_true=y_test,
            y_proba=y_proba,
            shap_values=shap_fraud,
            feature_names=X_test.columns.tolist(),
            high_risk_threshold=0.90,
            true_positive_threshold=0.70
        )
    return dict(
        client=client,
        st_model=st_model,
        rag_index=rag_index,
        chunk_texts=chunk_texts,
        chunk_metadata=chunk_metadata,
        X_test=X_test,
        feature_names=X_test.columns.tolist(),
        provider_db=provider_db,
        shap_fraud=shap_fraud,
        explainer=explainer,
        detection_threshold=detection_threshold,
        scaler=scaler,
        model=model,
        analyzer=analyzer,
    )


# ── On-the-fly scorer for providers not in the dataset ───────────────────────
def score_new_provider(provider_id: int, row_series: pd.Series, res: dict) -> dict:
    """Score a brand-new provider using the cached model, scaler, and SHAP explainer."""
    feature_names = res["feature_names"]
    scaler        = res["scaler"]
    model         = res["model"]
    explainer     = res["explainer"]
    threshold     = res["detection_threshold"]
    scaled     = scaler.transform(row_series[feature_names].values.reshape(1, -1))
    fraud_prob = float(model.predict_proba(scaled)[0, 1])
    shap_vals  = explainer.shap_values(scaled)
    shap_arr   = np.asarray(shap_vals)
    shap_row   = shap_arr[0] if shap_arr.ndim == 2 else shap_arr
    return {
        "row":        row_series,
        "fraud_prob": fraud_prob,
        "flagged":    bool(fraud_prob >= threshold),
        "shap_row":   shap_row,
    }


# ── Tool implementations ──────────────────────────────────────────────────────
def lookup_provider(provider_id: int, res: dict, extra: dict = None) -> dict:
    extra         = extra or {}
    feature_names = res["feature_names"]

    # ── New provider scored on-the-fly ────────────────────────────────────────
    if provider_id in extra:
        ep       = extra[provider_id]
        row_data = ep["row"]
        shap_row = ep["shap_row"]
        top_idx  = np.argsort(np.abs(shap_row))[::-1][:5]
        profile  = {
            "provider_id":  provider_id,
            "fraud_prob":   f"{ep['fraud_prob']:.1%}",
            "flagged":      ep["flagged"],
            "true_label":   "unknown (new provider — not in test set)",
            "top_features": {},
            "note":         "Scored on-the-fly from uploaded feature data.",
        }
        for fi in top_idx:
            fname = feature_names[fi]
            profile["top_features"][fname] = {
                "value": round(float(row_data[fname]), 4),
                "shap":  round(float(shap_row[fi]), 4),
                "desc":  FEATURE_DESCRIPTIONS.get(fname, fname),
            }
        return profile

    # ── Dataset provider ──────────────────────────────────────────────────────
    provider_db = res["provider_db"]
    X_test      = res["X_test"]
    shap_fraud  = res["shap_fraud"]
    if provider_id not in provider_db.index:
        return {"error": f"Provider {provider_id} not found. If this is a new provider, upload their feature CSV in the sidebar first."}
    row      = provider_db.loc[provider_id]
    test_pos = X_test.index.get_loc(provider_id)
    shap_row = shap_fraud[test_pos]
    top_idx  = np.argsort(np.abs(shap_row))[::-1][:5]
    profile  = {
        "provider_id": provider_id,
        "fraud_prob":  f"{row['fraud_prob']:.1%}",
        "flagged":     bool(row["flagged"]),
        "true_label":  int(row["true_label"]),
        "top_features": {},
    }
    for fi in top_idx:
        fname = feature_names[fi]
        profile["top_features"][fname] = {
            "value": round(float(X_test.iloc[test_pos][fname]), 4),
            "shap":  round(float(shap_row[fi]), 4),
            "desc":  FEATURE_DESCRIPTIONS.get(fname, fname),
        }
    return profile


def embed_query(query: str, st_model) -> np.ndarray:
    vec = st_model.encode([query], convert_to_numpy=True)[0].astype(np.float32)
    return vec / (np.linalg.norm(vec) + 1e-10)


def search_policy(query: str, res: dict, top_k: int = 4, min_score: float = POLICY_MIN_SCORE) -> list:
    rag_index      = res["rag_index"]
    chunk_texts    = res["chunk_texts"]
    chunk_metadata = res["chunk_metadata"]
    st_model       = res["st_model"]

    q = embed_query(query, st_model).reshape(1, -1)
    scores, indices = rag_index.search(q, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1 or float(score) < min_score:
            continue
        meta = chunk_metadata[idx]
        sp, ep = meta.get("start_page"), meta.get("end_page")
        if sp and ep and sp != ep:
            pages_str = f"pp. {sp}–{ep}"
        elif sp:
            pages_str = f"p. {sp}"
        else:
            pages_str = "page unknown"
        results.append({
            "score":      round(float(score), 4),
            "category":   meta["source_category"],
            "source":     meta["source_file"],
            "pages":      pages_str,
            "chapter":    (meta.get("chapter")    or "n/a")[:80],
            "subsection": (meta.get("subsection") or "n/a")[:80],
            "chunk_id":   meta.get("chunk_id", "n/a"),
            "excerpt":    chunk_texts[idx][:400],
        })
    if not results:
        return [{"message": "No highly relevant policy matches found."}]
    return results


def get_shap_explanation(provider_id: int, res: dict, extra: dict = None) -> dict:
    extra         = extra or {}
    explainer     = res["explainer"]
    feature_names = res["feature_names"]

    base_val = (
        float(explainer.expected_value)
        if np.isscalar(explainer.expected_value)
        else float(explainer.expected_value[0])
    )

    # ── New provider scored on-the-fly ────────────────────────────────────────
    if provider_id in extra:
        ep       = extra[provider_id]
        shap_row = ep["shap_row"]
        top10    = []
        for fi, sv in sorted(enumerate(shap_row), key=lambda x: abs(x[1]), reverse=True)[:10]:
            top10.append({
                "feature":     feature_names[fi],
                "shap_value":  round(float(sv), 5),
                "direction":   "increases fraud risk" if sv > 0 else "decreases fraud risk",
                "description": FEATURE_DESCRIPTIONS.get(feature_names[fi], feature_names[fi]),
            })
        return {
            "provider_id":  provider_id,
            "fraud_prob":   f"{ep['fraud_prob']:.1%}",
            "base_value":   round(base_val, 5),
            "top_features": top10,
            "note":         "Scored on-the-fly from uploaded feature data.",
        }

    # ── Dataset provider ──────────────────────────────────────────────────────
    provider_db = res["provider_db"]
    X_test      = res["X_test"]
    shap_fraud  = res["shap_fraud"]
    if provider_id not in provider_db.index:
        return {"error": f"Provider {provider_id} not found. If this is a new provider, upload their feature CSV in the sidebar first."}
    test_pos = X_test.index.get_loc(provider_id)
    shap_row = shap_fraud[test_pos]
    top10    = []
    for fi, sv in sorted(enumerate(shap_row), key=lambda x: abs(x[1]), reverse=True)[:10]:
        top10.append({
            "feature":     feature_names[fi],
            "shap_value":  round(float(sv), 5),
            "direction":   "increases fraud risk" if sv > 0 else "decreases fraud risk",
            "description": FEATURE_DESCRIPTIONS.get(feature_names[fi], feature_names[fi]),
        })
    return {
        "provider_id":  provider_id,
        "fraud_prob":   f"{provider_db.loc[provider_id, 'fraud_prob']:.1%}",
        "base_value":   round(base_val, 5),
        "top_features": top10,
    }


def analyze_review_flagged_providers(
    analysis_type: str, res: dict, top_n: int = 20, provider_id: int = None
) -> dict:
    """
    Analyze review-flagged providers.
    
    Parameters
    ----------
    analysis_type : str
        Type of analysis: 'summary', 'top_providers', 'pattern_comparison', or 'specific_provider'
    res : dict
        Resources dictionary containing analyzer
    top_n : int, optional
        Number of top providers to return (for 'top_providers' mode)
    provider_id : int, optional
        Specific provider ID to analyze (for 'specific_provider' mode)
    
    Returns
    -------
    dict
        Analysis results
    """
    analyzer = res["analyzer"]
    
    if analysis_type == "summary":
        summary = analyzer.generate_group_summary()
        return {
            "total_rfp_providers": summary['group_sizes']['review_flagged_providers'],
            "total_confirmed_fraud": summary['group_sizes']['true_positive'],
            "avg_fraud_probability_rfp": f"{summary['avg_fraud_probability']['review_flagged_providers']:.1%}",
            "avg_fraud_probability_fraud": f"{summary['avg_fraud_probability']['true_positive']:.1%}",
            "common_risk_drivers": summary['feature_overlap']['common_features'][:7],
            "unique_to_rfp": summary['feature_overlap']['rfp_unique'][:5],
            "unique_to_fraud": summary['feature_overlap']['tp_unique'][:5],
            "pattern_similarity": f"{summary['feature_overlap']['jaccard_similarity']:.1%}",
            "avg_similarity_to_fraud": f"{summary['similarity_stats']['mean']:.1%}"
        }
    
    elif analysis_type == "top_providers":
        queue = analyzer.calculate_investigation_priority()
        results = []
        for _, row in queue.head(top_n).iterrows():
            results.append({
                'provider_id': int(row['provider_id']),
                'fraud_probability': f"{row['fraud_probability']:.1%}",
                'priority_score': round(row['priority_score'], 1),
                'priority_rank': int(row['priority_rank']),
                'similarity_to_fraud': f"{row['similarity_to_fraud']:.1%}"
            })
        return {"top_priority_providers": results}
    
    elif analysis_type == "pattern_comparison":
        rfp_feats = analyzer.get_top_shap_features(analyzer.rfp_mask, top_k=10)
        tp_feats = analyzer.get_top_shap_features(analyzer.tp_mask, top_k=10)
        return {
            "rfp_top_features": rfp_feats[['feature_name', 'mean_shap_value', 'pct_providers_positive']].to_dict('records'),
            "fraud_top_features": tp_feats[['feature_name', 'mean_shap_value', 'pct_providers_positive']].to_dict('records'),
            "interpretation": (
                f"Identified {len(analyzer.rfp_indices)} review-flagged providers. "
                f"Comparing SHAP feature patterns to {len(analyzer.tp_indices)} confirmed fraud cases."
            )
        }
    
    elif analysis_type == "specific_provider":
        if not provider_id:
            return {"error": "provider_id required for specific_provider mode"}
        try:
            details = analyzer.get_provider_details(provider_id)
            return {
                'provider_id': details['provider_id'],
                'fraud_probability': f"{details['fraud_probability']:.1%}",
                'priority_score': round(details['priority_score'], 1),
                'priority_rank': details['priority_rank'],
                'similarity_to_fraud': f"{details['similarity_to_fraud']:.1%}",
                'top_risk_drivers': [
                    {
                        'feature': f['feature_name'],
                        'shap_value': round(f['shap_value'], 4),
                        'feature_value': round(f['feature_value'], 4)
                    }
                    for f in details['top_shap_features'][:5]
                ]
            }
        except ValueError as e:
            return {"error": str(e)}
    
    else:
        return {"error": f"Unknown analysis_type: {analysis_type}"}


def execute_tool_call(tool_name: str, tool_args: dict, res: dict, extra: dict = None) -> str:
    extra = extra or {}
    if tool_name == "lookup_provider":
        result = lookup_provider(**tool_args, res=res, extra=extra)
    elif tool_name == "search_policy":
        result = search_policy(**tool_args, res=res)
    elif tool_name == "get_shap_explanation":
        result = get_shap_explanation(**tool_args, res=res, extra=extra)
    elif tool_name == "analyze_review_flagged_providers":
        result = analyze_review_flagged_providers(**tool_args, res=res)
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    return json.dumps(result, indent=2)


# ── Agentic chat loop ─────────────────────────────────────────────────────────
def run_chat(user_message: str, res: dict):
    """Run one turn of the agentic loop. Returns (answer, tool_log)."""
    client   = res["client"]
    messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    messages.append({"role": "user", "content": user_message})

    extra    = st.session_state.get("extra_providers", {})
    tool_log = []  # list of (tool_name, args, result_str)
    for _ in range(6):
        response = client.messages.create(
            model=CLAUDE_MODEL,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=TOOLS,
            temperature=0.2,
            max_tokens=1500,
        )
        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result_str = execute_tool_call(block.name, block.input, res, extra)
                    tool_log.append((block.name, block.input, result_str))
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     result_str,
                    })
            messages.append({"role": "user", "content": tool_results})
        else:
            return response.content[0].text, tool_log

    return "Maximum tool iterations reached. Please rephrase your question.", tool_log


# ── Session state init ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []   # {"role": ..., "content": ..., "tool_log": [...]}


# ── Load resources ────────────────────────────────────────────────────────────
res = load_resources()
provider_db = res["provider_db"]
flagged_ids = provider_db[provider_db["flagged"] == 1].sort_values(
    "fraud_prob", ascending=False
).index.tolist()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛡️ AI Fraud Investigation Platform")
    st.caption(f"Powered by {CLAUDE_MODEL}")
    st.divider()

    st.subheader("📊 Dataset Statistics")
    st.metric("Total Providers", f"{len(provider_db):,}")
    st.metric("Flagged High-Risk", len(flagged_ids))
    fraud_count = int((provider_db["true_label"] == 1).sum())
    st.metric("Confirmed Fraud", f"{fraud_count:,}")
    st.metric("RAG Knowledge Base", f"{res['rag_index'].ntotal:,} vectors")
    st.divider()

    st.subheader("⚡ Quick Start")
    selected = st.selectbox(
        "Select a flagged provider",
        options=flagged_ids,
        format_func=lambda pid: (
            f"Provider {pid}  —  {provider_db.loc[pid, 'fraud_prob']:.1%} risk"
        ),
    )
    if st.button("Investigate this provider", use_container_width=True):
        starter = (
            f"Tell me about provider {selected}. "
            "Why did the model flag them for fraud? "
            "What Medicare rules might they be violating?"
        )
        st.session_state._prefill = starter

    st.divider()
    st.subheader("🆕 New Provider")
    st.caption("Score any provider not yet in the dataset.")
    new_pid = st.number_input("Provider ID", min_value=0, step=1, key="new_pid_input")
    template_csv = pd.DataFrame(columns=res["feature_names"]).to_csv(index=False)
    st.download_button(
        "📥 Download feature template",
        data=template_csv,
        file_name="new_provider_template.csv",
        mime="text/csv",
        use_container_width=True,
    )
    uploaded = st.file_uploader("Upload filled CSV", type="csv", key="new_csv")
    if st.button("Score & Add Provider", use_container_width=True):
        if uploaded is None:
            st.warning("Please upload a filled CSV first.")
        else:
            try:
                df_new  = pd.read_csv(uploaded)
                missing = [c for c in res["feature_names"] if c not in df_new.columns]
                if missing:
                    st.error(f"Missing {len(missing)} column(s): {', '.join(missing[:4])}{'…' if len(missing) > 4 else ''}")
                else:
                    row = df_new[res["feature_names"]].iloc[0]
                    ep  = score_new_provider(int(new_pid), row, res)
                    if "extra_providers" not in st.session_state:
                        st.session_state.extra_providers = {}
                    st.session_state.extra_providers[int(new_pid)] = ep
                    st.success(f"Provider {int(new_pid)} scored: {ep['fraud_prob']:.1%} fraud risk")
            except Exception as e:
                st.error(f"Error scoring provider: {e}")
    _extra = st.session_state.get("extra_providers", {})
    if _extra:
        st.caption(f"{len(_extra)} new provider(s) added this session")
        for _pid, _ep in _extra.items():
            _flag = "🚨" if _ep["flagged"] else "✅"
            st.text(f"{_flag} Provider {_pid}: {_ep['fraud_prob']:.1%}")
    
    st.divider()
    st.subheader("🔬 Review-Flagged Providers")
    st.caption("Analyze providers with high fraud scores but no fraud label — potential undetected fraud.")
    
    rfp_analysis_type = st.selectbox(
        "Analysis Type",
        options=["summary", "top_providers", "pattern_comparison"],
        format_func=lambda x: {
            "summary": "📊 Group Summary",
            "top_providers": "🎯 Priority Queue",
            "pattern_comparison": "🔍 Pattern vs Fraud"
        }[x],
        key="rfp_analysis_type"
    )
    
    if rfp_analysis_type == "top_providers":
        top_n = st.slider("Number of providers", min_value=5, max_value=50, value=20, step=5, key="rfp_top_n")
    else:
        top_n = 20
    
    if st.button("Run RFP Analysis", use_container_width=True):
        if rfp_analysis_type == "summary":
            starter = "Analyze the review-flagged provider group. Give me a summary of how many there are, their average fraud probability, and how they compare to confirmed fraud cases."
        elif rfp_analysis_type == "top_providers":
            starter = f"Show me the top {top_n} review-flagged providers ranked by investigation priority. Include their fraud probabilities and priority scores."
        else:  # pattern_comparison
            starter = "Compare the risk patterns of review-flagged providers versus confirmed fraud cases. What features are most important for each group?"
        st.session_state._prefill = starter
    
    st.divider()
    if st.button("🗑️  Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main header ──────────────────────────────────────────────────────────────
st.markdown("# 🛡️ AI-Powered Fraud Investigation Platform")
st.caption(
    "Medicare fraud detection and investigation assistant powered by machine learning and large language models. "
    "Ask questions about flagged providers, analyze billing patterns, or search policy guidance."
)

# ── KPI Cards ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

total_providers = len(provider_db)
fraud_providers = int((provider_db["true_label"] == 1).sum())
rfp_count = int(((provider_db["fraud_prob"] >= 0.90) & (provider_db["true_label"] == 0)).sum())
avg_fraud_prob = provider_db["fraud_prob"].mean()
high_risk_count = int((provider_db["flagged"] == 1).sum())

with col1:
    st.markdown(render_kpi_card("👥", f"{total_providers:,}", "Total Providers"), unsafe_allow_html=True)

with col2:
    st.markdown(render_kpi_card("🚨", f"{fraud_providers:,}", "Confirmed Fraud"), unsafe_allow_html=True)

with col3:
    st.markdown(render_kpi_card("⚠️", f"{high_risk_count:,}", "Flagged High-Risk"), unsafe_allow_html=True)

with col4:
    st.markdown(render_kpi_card("🔬", f"{rfp_count:,}", "Review-Flagged Providers"), unsafe_allow_html=True)

with col5:
    st.markdown(render_kpi_card("📊", f"{avg_fraud_prob:.1%}", "Avg Fraud Probability"), unsafe_allow_html=True)

st.divider()

# ── Render conversation history ───────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            # Create tabs for AI responses
            tab_labels = ["💬 Response"]
            if msg.get("tool_log"):
                tab_labels.append("🔧 Tool Calls")
            
            tabs = st.tabs(tab_labels)
            
            # Response tab
            with tabs[0]:
                st.markdown(msg["content"])
            
            # Tool calls tab
            if msg.get("tool_log") and len(tab_labels) > 1:
                with tabs[1]:
                    for tool_name, tool_args, tool_result in msg["tool_log"]:
                        with st.expander(f"🔧 {tool_name}", expanded=False):
                            st.json(tool_args)
                            st.code(tool_result, language="json")
        else:
            st.markdown(msg["content"])

# ── Handle quick-start prefill ────────────────────────────────────────────────
prefill = st.session_state.pop("_prefill", None)

# ── Chat input ────────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask the AI Fraud Analyst about a provider, policy, or investigation strategy...") or prefill

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("🔍 Analyzing data and searching policy guidance..."):
            answer, tool_log = run_chat(prompt, res)

        # Create tabs for response
        tab_labels = ["💬 Response"]
        if tool_log:
            tab_labels.append("🔧 Tool Calls")
        
        tabs = st.tabs(tab_labels)
        
        with tabs[0]:
            st.markdown(answer)
        
        if tool_log and len(tab_labels) > 1:
            with tabs[1]:
                for tool_name, tool_args, tool_result in tool_log:
                    with st.expander(f"🔧 {tool_name}", expanded=False):
                        st.json(tool_args)
                        st.code(tool_result, language="json")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "tool_log": tool_log}
    )
