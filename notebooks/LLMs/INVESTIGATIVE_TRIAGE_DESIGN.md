# Review-Flagged Providers Provider Analysis System
## Design Document for PhD Thesis Extension

---

## 1. Academic Framing & Motivation

### 1.1 Theoretical Foundation

**Traditional View:**
- False positives = model errors to minimize
- Binary classification: fraud vs. non-fraud
- Performance metrics: precision, recall, F1

**Investigative Triage View (Your Contribution):**
- False positives may contain **high-value investigative leads**
- **Label uncertainty**: "non-fraud" ≠ confirmed legitimate
- **Resource-constrained decision support**: prioritize limited investigative capacity
- **Human-AI collaboration**: model assists, humans decide

### 1.2 Research Questions

1. **RQ1**: Do high-confidence false positives share feature patterns with confirmed fraud cases?
2. **RQ2**: Can SHAP explanations identify common risk drivers among unconfirmed high-risk providers?
3. **RQ3**: What provider characteristics distinguish "worth investigating" false positives from true negatives?
4. **RQ4**: How can XAI techniques support investigative resource allocation beyond binary classification?

### 1.3 Literature Positioning

**Cite & Position Against:**
- **Adversarial ML**: Providers may game detection systems (false negatives)
- **Label Noise**: Ground truth may have errors (undetected fraud labeled as non-fraud)
- **Cost-Sensitive Learning**: Investigation costs vs. fraud recovery value
- **Human-AI Collaboration**: Shared decision-making in high-stakes domains (medical diagnosis, loan approval, fraud detection)
- **Active Learning**: High-confidence disagreements → informative samples
- **Explainable AI**: Using explanations for discovery, not just justification

---

## 2. System Architecture

### 2.1 Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│  INVESTIGATIVE TRIAGE SYSTEM                                 │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │  1. REVIEW-FLAGGED PROVIDER IDENTIFICATION  │     │
│  │     - Probability threshold (e.g., ≥ 0.90)         │     │
│  │     - Ground truth: y_true == 0                    │     │
│  │     - Comparison group: true positives             │     │
│  └────────────────────────────────────────────────────┘     │
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │  2. SHAP-BASED PATTERN MINING                      │     │
│  │     - Top risk drivers (SHAP value aggregation)    │     │
│  │     - Feature distribution comparison              │     │
│  │     - Cluster analysis (FP vs TP)                  │     │
│  └────────────────────────────────────────────────────┘     │
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │  3. INVESTIGATION PRIORITY SCORING                 │     │
│  │     - Fraud probability                            │     │
│  │     - SHAP explanation strength                    │     │
│  │     - Similarity to confirmed fraud                │     │
│  │     - Policy violation severity                    │     │
│  └────────────────────────────────────────────────────┘     │
│                                                               │
│  ┌────────────────────────────────────────────────────┐     │
│  │  4. INVESTIGATOR-FACING INTERFACE                  │     │
│  │     - Ranked investigation queue                   │     │
│  │     - Comparative visualizations                   │     │
│  │     - Natural language summaries                   │     │
│  │     - Chatbot integration                          │     │
│  └────────────────────────────────────────────────────┘     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
ML Model Predictions
         ↓
Review-Flagged Providers Filter (P(fraud) ≥ 0.90 & y_true == 0)
         ↓
SHAP Explanation Extraction
         ↓
Feature Pattern Analysis ← Comparison → True Positive Group
         ↓
Priority Score Calculation
         ↓
Investigation Queue Generation
         ↓
Chatbot / Dashboard Display
```

---

## 3. Implementation Plan

### 3.1 Core Analysis Module

**File: `review_flagged_providers_analysis.py`**

```python
"""
Review-Flagged Providers Provider Analysis
Identifies and analyzes providers flagged at high confidence but not labeled as fraud.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity


class ReviewFlaggedProvidersAnalyzer:
    """
    Analyzes providers with high fraud probability but non-fraud labels.
    
    Supports investigative triage by identifying patterns in high-confidence
    false positives that may represent:
    - Unlabeled fraud cases
    - Emerging fraud patterns
    - High-risk edge cases requiring human judgment
    """
    
    def __init__(
        self,
        X: pd.DataFrame,
        y_true: pd.Series,
        y_proba: np.ndarray,
        shap_values: np.ndarray,
        feature_names: List[str],
        high_risk_threshold: float = 0.90,
        true_positive_threshold: float = 0.70
    ):
        """
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (scaled)
        y_true : pd.Series
            Ground truth labels (0=non-fraud, 1=fraud)
        y_proba : np.ndarray
            Model fraud probabilities
        shap_values : np.ndarray
            SHAP explanation matrix (n_samples x n_features)
        feature_names : List[str]
            Feature names
        high_risk_threshold : float
            Minimum probability to be considered "review-flagged"
        true_positive_threshold : float
            Minimum probability for confirmed fraud comparison group
        """
        self.X = X
        self.y_true = y_true
        self.y_proba = y_proba
        self.shap_values = shap_values
        self.feature_names = feature_names
        self.high_risk_threshold = high_risk_threshold
        self.true_positive_threshold = true_positive_threshold
        
        # Identify groups
        self.rfp_mask = (y_proba >= high_risk_threshold) & (y_true == 0)
        self.tp_mask = (y_proba >= true_positive_threshold) & (y_true == 1)
        
        self.rfp_indices = np.where(self.rfp_mask)[0]
        self.tp_indices = np.where(self.tp_mask)[0]
        
        print(f"Review-Flagged Providers (RFP): {len(self.rfp_indices)} providers")
        print(f"True Positives (TP):         {len(self.tp_indices)} providers")
    
    
    def get_top_shap_features(
        self,
        group_mask: np.ndarray,
        top_k: int = 10
    ) -> pd.DataFrame:
        """
        Get top SHAP features for a provider group.
        
        Returns DataFrame with:
        - feature_name
        - mean_shap_value (average SHAP contribution)
        - median_shap_value
        - pct_providers_positive (% of providers where SHAP > 0)
        """
        group_shap = self.shap_values[group_mask]
        
        results = []
        for i, fname in enumerate(self.feature_names):
            shap_col = group_shap[:, i]
            results.append({
                'feature_name': fname,
                'mean_shap_value': shap_col.mean(),
                'median_shap_value': np.median(shap_col),
                'std_shap_value': shap_col.std(),
                'pct_providers_positive': (shap_col > 0).mean() * 100
            })
        
        df = pd.DataFrame(results).sort_values('mean_shap_value', ascending=False)
        return df.head(top_k)
    
    
    def compare_feature_distributions(
        self,
        feature_name: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare feature distributions: RFP vs TP vs TN.
        
        Returns statistics for each group:
        - mean, median, std, min, max, p25, p75
        """
        feature_idx = self.feature_names.index(feature_name)
        
        rfp_vals = self.X.iloc[self.rfp_indices, feature_idx]
        tp_vals = self.X.iloc[self.tp_indices, feature_idx]
        tn_vals = self.X.iloc[(self.y_true == 0) & (self.y_proba < 0.30), feature_idx]
        
        def stats(vals):
            return {
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'p25': float(np.percentile(vals, 25)),
                'p75': float(np.percentile(vals, 75))
            }
        
        return {
            'review_flagged_providers': stats(rfp_vals),
            'true_positive': stats(tp_vals),
            'true_negative': stats(tn_vals)
        }
    
    
    def calculate_similarity_to_fraud(self) -> pd.Series:
        """
        For each review-flagged provider, calculate average cosine similarity
        to confirmed fraud cases (based on SHAP patterns).
        
        Returns Series indexed by review-flagged provider IDs.
        """
        rfp_shap = self.shap_values[self.rfp_mask]
        tp_shap = self.shap_values[self.tp_mask]
        
        # Compute pairwise cosine similarity
        similarities = cosine_similarity(rfp_shap, tp_shap)
        
        # Average similarity to all TPs
        avg_sim = similarities.mean(axis=1)
        
        return pd.Series(
            avg_sim,
            index=self.X.index[self.rfp_mask],
            name='avg_similarity_to_fraud'
        )
    
    
    def calculate_investigation_priority(self) -> pd.DataFrame:
        """
        Calculate investigation priority score for review-flagged providers.
        
        Priority Score = weighted combination of:
        1. Fraud probability (40%)
        2. SHAP explanation strength (30%)
        3. Similarity to confirmed fraud (30%)
        
        Returns ranked DataFrame with:
        - provider_id
        - fraud_probability
        - shap_strength (sum of absolute SHAP values)
        - similarity_to_fraud
        - priority_score (0-100)
        - priority_rank
        """
        rfp_ids = self.X.index[self.rfp_mask]
        rfp_probas = self.y_proba[self.rfp_mask]
        rfp_shap = self.shap_values[self.rfp_mask]
        
        # SHAP strength = total absolute contribution
        shap_strength = np.abs(rfp_shap).sum(axis=1)
        
        # Similarity to fraud
        similarity = self.calculate_similarity_to_fraud().values
        
        # Normalize to 0-1
        def normalize(x):
            if x.max() == x.min():
                return np.ones_like(x)
            return (x - x.min()) / (x.max() - x.min())
        
        norm_proba = normalize(rfp_probas)
        norm_shap = normalize(shap_strength)
        norm_sim = normalize(similarity)
        
        # Weighted priority score
        priority = (
            0.40 * norm_proba +
            0.30 * norm_shap +
            0.30 * norm_sim
        ) * 100
        
        df = pd.DataFrame({
            'provider_id': rfp_ids,
            'fraud_probability': rfp_probas,
            'shap_strength': shap_strength,
            'similarity_to_fraud': similarity,
            'priority_score': priority
        }).sort_values('priority_score', ascending=False)
        
        df['priority_rank'] = range(1, len(df) + 1)
        
        return df
    
    
    def generate_group_summary(self) -> Dict:
        """
        Generate comprehensive summary comparing RFP vs TP groups.
        
        Returns dict with:
        - group_sizes
        - top_shap_features (for each group)
        - common_features (overlap in top 10)
        - distinct_features (unique to each group)
        - avg_fraud_probability
        - feature_distribution_divergence
        """
        rfp_top_features = self.get_top_shap_features(self.rfp_mask, top_k=10)
        tp_top_features = self.get_top_shap_features(self.tp_mask, top_k=10)
        
        rfp_set = set(rfp_top_features['feature_name'].values)
        tp_set = set(tp_top_features['feature_name'].values)
        
        return {
            'group_sizes': {
                'review_flagged_providers': int(self.rfp_mask.sum()),
                'true_positive': int(self.tp_mask.sum())
            },
            'avg_fraud_probability': {
                'review_flagged_providers': float(self.y_proba[self.rfp_mask].mean()),
                'true_positive': float(self.y_proba[self.tp_mask].mean())
            },
            'top_shap_features': {
                'review_flagged_providers': rfp_top_features.to_dict('records'),
                'true_positive': tp_top_features.to_dict('records')
            },
            'feature_overlap': {
                'common_features': list(rfp_set & tp_set),
                'rfp_unique': list(rfp_set - tp_set),
                'tp_unique': list(tp_set - rfp_set)
            }
        }


def generate_investigator_summary(
    provider_id: int,
    fraud_prob: float,
    priority_score: float,
    priority_rank: int,
    top_shap_features: List[Tuple[str, float]],
    similarity_to_fraud: float
) -> str:
    """
    Generate human-readable investigation summary for review-flagged provider.
    
    Parameters
    ----------
    provider_id : int
    fraud_prob : float
    priority_score : float (0-100)
    priority_rank : int
    top_shap_features : List[Tuple[feature_name, shap_value]]
    similarity_to_fraud : float (0-1)
    
    Returns
    -------
    str : Investigation summary
    """
    
    summary = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  INVESTIGATION PRIORITY ALERT — Provider {provider_id}                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

CLASSIFICATION STATUS:
  • Ground Truth Label:     NON-FRAUD (per available data)
  • Model Risk Score:        {fraud_prob:.1%} fraud probability
  • Investigation Priority:  {priority_score:.0f}/100 (Rank #{priority_rank})

⚠️  INVESTIGATIVE NOTE:
This provider is NOT confirmed as fraudulent in the current dataset. However,
the ML model has assigned an extremely high fraud risk score ({fraud_prob:.1%}),
placing it in the top tier of model confidence.

High-confidence model flags on non-fraud-labeled providers may indicate:
  1. Undetected fraud (not yet investigated or proven)
  2. Emerging fraud patterns not captured in training labels
  3. Legitimate edge cases requiring human expert judgment
  4. Data labeling uncertainty or incomplete ground truth

SIMILARITY TO CONFIRMED FRAUD CASES: {similarity_to_fraud:.1%}
The provider's billing patterns (as measured by SHAP explanations) show
{similarity_to_fraud:.1%} similarity to confirmed fraud cases, suggesting
structural similarities in billing behavior.

TOP RISK DRIVERS (SHAP Analysis):
"""
    
    for i, (feat, shap_val) in enumerate(top_shap_features[:5], 1):
        summary += f"  {i}. {feat}\n"
        summary += f"     Risk contribution: {shap_val:+.4f}\n"
    
    summary += f"""
RECOMMENDED INVESTIGATIVE ACTIONS:
  1. Manual review of high-dollar claims and billing patterns
  2. Cross-reference with OIG exclusion lists and prior investigations
  3. Compare billing rates to peer providers (same specialty/region)
  4. Assess for policy violations related to top risk drivers
  5. If warranted, escalate to UPIC for formal audit

IMPORTANT: This alert is a decision-support tool. It does NOT constitute
evidence of fraud. All investigative conclusions must be based on documentary
evidence and established legal standards.
"""
    
    return summary
```

### 3.2 Chatbot Integration

**Add to `LLM_3_Fraud_Investigator_Chatbot.ipynb`:**

```python
# ── New Tool: analyze_review_flagged_providers ───────────────────────────────────

TOOL_ANALYZE_HRU = {
    "name": "analyze_review_flagged_providers",
    "description": (
        "Analyze providers flagged at very high fraud risk by the model but NOT "
        "labeled as fraud in the dataset. These may represent undetected fraud, "
        "emerging patterns, or high-risk edge cases. Returns summary statistics, "
        "top risk drivers, and investigation priorities."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "min_probability": {
                "type": "number",
                "description": "Minimum fraud probability to be considered review-flagged (default: 0.90)"
            },
            "top_n": {
                "type": "integer",
                "description": "Number of top providers to return (default: 20)"
            },
            "comparison_mode": {
                "type": "string",
                "enum": ["summary", "detailed", "comparison"],
                "description": "summary=group stats | detailed=individual providers | comparison=RFP vs TP"
            }
        },
        "required": []
    }
}


def analyze_review_flagged_providers(
    min_probability: float = 0.90,
    top_n: int = 20,
    comparison_mode: str = "summary"
) -> Dict:
    """
    Analyze review-flagged providers.
    """
    
    # Initialize analyzer
    analyzer = ReviewFlaggedProvidersAnalyzer(
        X=pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index),
        y_true=y_test,
        y_proba=y_proba,
        shap_values=shap_values,
        feature_names=feature_names,
        high_risk_threshold=min_probability
    )
    
    if comparison_mode == "summary":
        # Group-level summary
        return analyzer.generate_group_summary()
    
    elif comparison_mode == "detailed":
        # Top N individual providers
        priority_df = analyzer.calculate_investigation_priority()
        return priority_df.head(top_n).to_dict('records')
    
    elif comparison_mode == "comparison":
        # Feature-by-feature comparison
        rfp_features = analyzer.get_top_shap_features(analyzer.rfp_mask, top_k=10)
        tp_features = analyzer.get_top_shap_features(analyzer.tp_mask, top_k=10)
        
        return {
            "review_flagged_providers_top_features": rfp_features.to_dict('records'),
            "true_positive_top_features": tp_features.to_dict('records'),
            "interpretation": (
                f"Identified {len(analyzer.rfp_indices)} review-flagged providers "
                f"with ≥{min_probability:.0%} fraud probability but non-fraud labels. "
                f"Comparing feature patterns to {len(analyzer.tp_indices)} confirmed fraud cases."
            )
        }
```

### 3.3 Dashboard Additions (Streamlit)

**Add to `streamlit_chatbot.py`:**

```python
# ── New tab: Investigation Queue ──────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["💬 Chatbot", "📊 Provider Lookup", "🚨 Investigation Queue"])

with tab3:
    st.header("🚨 Review-Flagged Providers Provider Investigation Queue")
    
    st.info(
        "**What is this?**  \n"
        "Providers with very high fraud risk scores (≥90%) but NOT labeled as fraud "
        "in the dataset. These may represent undetected fraud, emerging patterns, or "
        "high-priority edge cases requiring human judgment."
    )
    
    # Initialize analyzer
    analyzer = ReviewFlaggedProvidersAnalyzer(
        X=pd.DataFrame(X_test_scaled, columns=feature_names, index=X_test.index),
        y_true=y_test,
        y_proba=y_proba,
        shap_values=shap_values,
        feature_names=feature_names,
        high_risk_threshold=0.90
    )
    
    priority_df = analyzer.calculate_investigation_priority()
    
    st.metric("Review-Flagged Providers Providers", len(priority_df))
    
    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        min_priority = st.slider("Min Priority Score", 0, 100, 70)
    with col2:
        top_n = st.number_input("Show Top N", 1, len(priority_df), 20)
    
    # Filtered table
    filtered = priority_df[priority_df['priority_score'] >= min_priority].head(top_n)
    
    st.dataframe(
        filtered.style.background_gradient(subset=['priority_score'], cmap='Reds'),
        use_container_width=True
    )
    
    # Pattern analysis
    st.subheader("Common Risk Patterns")
    summary = analyzer.generate_group_summary()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top Risk Drivers (Review-Flagged Providers)**")
        rfp_feats = pd.DataFrame(summary['top_shap_features']['review_flagged_providers'])
        st.dataframe(rfp_feats[['feature_name', 'mean_shap_value']])
    
    with col2:
        st.write("**Top Risk Drivers (Confirmed Fraud)**")
        tp_feats = pd.DataFrame(summary['top_shap_features']['true_positive'])
        st.dataframe(tp_feats[['feature_name', 'mean_shap_value']])
    
    # Common vs unique features
    st.subheader("Feature Pattern Comparison")
    common = summary['feature_overlap']['common_features']
    hru_unique = summary['feature_overlap']['rfp_unique']
    tp_unique = summary['feature_overlap']['tp_unique']
    
    st.write(f"**Shared Risk Drivers ({len(common)}):** {', '.join(common)}")
    st.write(f"**Unique to RFP ({len(hru_unique)}):** {', '.join(hru_unique)}")
    st.write(f"**Unique to Fraud ({len(tp_unique)}):** {', '.join(tp_unique)}")
```

---

## 4. Visualization Recommendations

### 4.1 Priority Queue Heatmap

```python
import plotly.express as px

priority_df = analyzer.calculate_investigation_priority()

fig = px.scatter(
    priority_df.head(50),
    x='fraud_probability',
    y='similarity_to_fraud',
    size='shap_strength',
    color='priority_score',
    hover_data=['provider_id', 'priority_rank'],
    title='Review-Flagged Providers Providers: Risk Landscape',
    labels={
        'fraud_probability': 'Model Fraud Probability',
        'similarity_to_fraud': 'Similarity to Confirmed Fraud',
        'priority_score': 'Investigation Priority'
    },
    color_continuous_scale='Reds'
)
fig.show()
```

### 4.2 SHAP Comparison (RFP vs TP)

```python
import matplotlib.pyplot as plt

rfp_shap = analyzer.get_top_shap_features(analyzer.rfp_mask, top_k=15)
tp_shap = analyzer.get_top_shap_features(analyzer.tp_mask, top_k=15)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

ax1.barh(rfp_shap['feature_name'], rfp_shap['mean_shap_value'], color='orange')
ax1.set_title('Review-Flagged Providers\n(Top SHAP Features)')
ax1.set_xlabel('Mean SHAP Value')

ax2.barh(tp_shap['feature_name'], tp_shap['mean_shap_value'], color='red')
ax2.set_title('Confirmed Fraud\n(Top SHAP Features)')
ax2.set_xlabel('Mean SHAP Value')

plt.tight_layout()
plt.show()
```

### 4.3 Feature Distribution Comparison

```python
import seaborn as sns

feature = 'Max_Medicare_Allowed_B'
comparison = analyzer.compare_feature_distributions(feature)

data = []
for group, stats in comparison.items():
    data.append({
        'Group': group.replace('_', ' ').title(),
        'Value': stats['median']
    })

df_plot = pd.DataFrame(data)

fig = px.box(
    # Actual data here would need to be expanded
    title=f'{feature} Distribution Across Groups',
    labels={'value': feature, 'variable': 'Group'}
)
```

---

## 5. Thesis Integration

### 5.1 Chapter Structure Recommendation

**Chapter 5: Explainable AI for Investigative Triage**

5.1 Introduction: Beyond Binary Classification  
5.2 Review-Flagged Providers Provider Identification  
  5.2.1 Motivation: Label Uncertainty in Fraud Detection  
  5.2.2 Methodology: Threshold-Based Risk Stratification  
  5.2.3 Comparison Group: True Positive Fraud Cases  
5.3 SHAP-Based Pattern Discovery  
  5.3.1 Feature Attribution Aggregation  
  5.3.2 Group-Level Risk Driver Analysis  
  5.3.3 Similarity Metrics for Fraud Pattern Matching  
5.4 Investigation Priority Scoring  
  5.4.1 Multi-Criteria Ranking Framework  
  5.4.2 Weighted Priority Score Calculation  
  5.4.3 Validation Against Investigator Feedback (if available)  
5.5 System Implementation  
  5.5.1 Chatbot Integration for Investigative Queries  
  5.5.2 Dashboard Design for Risk Visualization  
  5.5.3 User Interface Evaluation  
5.6 Results & Discussion  
  5.6.1 Pattern Analysis: RFP vs TP Comparison  
  5.6.2 Investigative Value of High-Confidence False Positives  
  5.6.3 Limitations & Ethical Considerations  

### 5.2 Key Thesis Claims

1. **"High-confidence false positives in fraud detection systems may represent high-value investigative leads rather than pure model errors."**

2. **"SHAP-based pattern analysis can identify common risk drivers among unconfirmed high-risk providers, enabling evidence-based investigation prioritization."**

3. **"Explainable AI techniques can support human-AI collaboration in resource-constrained investigative workflows by providing interpretable risk stratification."**

### 5.3 Academic Contributions

1. **Methodological**: Novel use of XAI for investigative triage (not just model explanation)
2. **Practical**: Decision-support framework for fraud investigation resource allocation
3. **Theoretical**: Reframing false positives in high-stakes domains (from errors to opportunities)

---

## 6. Example Chatbot Prompts

Add these to the chatbot system prompt or example queries:

```
INVESTIGATOR-ORIENTED QUERIES:

1. "Show me the top 10 review-flagged providers"
2. "Why did the model flag these providers if they're not confirmed fraud?"
3. "What are the common characteristics of high-risk false positives?"
4. "Compare feature patterns: review-flagged vs confirmed fraud"
5. "Which providers should I investigate first and why?"
6. "Are there any providers with >95% fraud probability but non-fraud labels?"
7. "What makes provider [ID] suspicious even though they're not labeled as fraud?"
8. "Show me providers similar to confirmed fraud cases but not yet proven"
9. "What are the unique risk drivers in unconfirmed high-risk providers?"
10. "Generate an investigation summary for the top-priority provider"
```

---

## 7. Evaluation Metrics

### 7.1 Quantitative Metrics

- **Pattern Similarity**: Cosine similarity between RFP and TP SHAP vectors
- **Feature Overlap**: Jaccard index of top-10 features (RFP vs TP)
- **Risk Stratification**: Spearman correlation between priority scores and actual investigation outcomes (if available)
- **Explanation Consistency**: Inter-rater reliability of SHAP interpretations

### 7.2 Qualitative Evaluation

- **Expert Review**: Have domain experts (fraud investigators) review top-priority RFP cases
- **Usability Study**: Measure time-to-decision with vs without the system
- **Trust Calibration**: Survey investigator confidence in system recommendations

---

## 8. Ethical & Legal Considerations

### 8.1 Critical Disclaimers

**Always include:**

1. **Non-Fraud Label ≠ Innocence**  
   "The absence of a fraud label does not imply the provider is innocent; it may reflect incomplete investigation or data."

2. **Decision Support, Not Automation**  
   "This system provides investigative leads. All final determinations must be based on legal evidence and due process."

3. **Transparency**  
   "All risk scores and explanations must be documented and auditable."

4. **Bias Monitoring**  
   "Regularly assess whether the system disproportionately flags providers based on protected characteristics."

### 8.2 Thesis Discussion Points

- **Label Noise**: How do you validate that review-flagged providers are "truly" worth investigating?
- **Confirmation Bias**: Risk of investigators selectively pursuing model-flagged cases
- **Feedback Loops**: If review-flagged providers are investigated and confirmed as fraud, does that validate the model or create circularity?

---

## 9. Implementation Timeline

**Week 1-2**: Code implementation (analyzer class, chatbot integration)  
**Week 3**: Dashboard design & visualization  
**Week 4**: Analysis & pattern discovery  
**Week 5-6**: Thesis writing (Chapter 5)  
**Week 7**: Expert review & refinement  

---

## 10. Next Steps

1. **Implement `ReviewFlaggedProvidersAnalyzer` class** (Section 3.1)
2. **Add chatbot tool** (`analyze_review_flagged_providers`)
3. **Create dashboard tab** for investigation queue
4. **Run analysis** on test set
5. **Generate visualizations** (SHAP comparison, priority scatter)
6. **Write thesis section** (Chapter 5 draft)
7. **(Optional)** Expert review with fraud investigators

---

## Contact & Questions

This design document provides a comprehensive roadmap. Implementation questions to resolve:

1. **Ground truth validation**: How to verify if review-flagged providers are "truly" suspicious?
2. **Priority score weights**: Should 40/30/30 be data-driven or domain-expert-specified?
3. **Threshold selection**: Is 0.90 the right cutoff, or should it be adaptive?
4. **Cluster analysis**: Should we use k-means/DBSCAN to find RFP sub-groups?

---

**Document Version**: 1.0  
**Author**: AI Assistant (based on professor feedback)  
**Date**: May 21, 2026  
**Purpose**: PhD Thesis Extension — Investigative Triage System
