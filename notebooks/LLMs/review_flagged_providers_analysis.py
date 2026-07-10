"""
Review-Flagged Provider Analysis Module
===============================================

Analyzes providers flagged at very high fraud probability but not labeled as fraud.
Supports investigative triage by identifying patterns that may represent:
  - Undetected fraud cases
  - Emerging fraud patterns
  - High-risk edge cases requiring human judgment

Author: [Your Name]
Date: May 21, 2026
Purpose: PhD Thesis - Explainable AI for Medicare Fraud Detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ks_2samp


class ReviewFlaggedProvidersAnalyzer:
    """
    Analyzes providers with high fraud probability but non-fraud labels.
    
    This class supports investigative triage by identifying patterns in high-confidence
    false positives that may represent valuable investigative leads.
    
    Attributes
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
        Minimum probability for "review-flagged" classification
    true_positive_threshold : float
        Minimum probability for confirmed fraud comparison group
    
    Examples
    --------
    >>> analyzer = ReviewFlaggedProvidersAnalyzer(
    ...     X=X_test_scaled_df,
    ...     y_true=y_test,
    ...     y_proba=fraud_probabilities,
    ...     shap_values=shap_matrix,
    ...     feature_names=feature_list,
    ...     high_risk_threshold=0.90
    ... )
    >>> priority_queue = analyzer.calculate_investigation_priority()
    >>> summary = analyzer.generate_group_summary()
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
        Initialize the analyzer.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (scaled), with provider IDs as index
        y_true : pd.Series
            Ground truth labels (0=non-fraud, 1=fraud), same index as X
        y_proba : np.ndarray
            Model fraud probabilities, aligned with X
        shap_values : np.ndarray
            SHAP explanation matrix (n_samples x n_features)
        feature_names : List[str]
            Feature names (must match X.columns)
        high_risk_threshold : float, default=0.90
            Minimum probability to be considered "review-flagged"
        true_positive_threshold : float, default=0.70
            Minimum probability for confirmed fraud comparison group
        """
        self.X = X
        self.y_true = y_true
        self.y_proba = y_proba
        self.shap_values = shap_values
        self.feature_names = feature_names
        self.high_risk_threshold = high_risk_threshold
        self.true_positive_threshold = true_positive_threshold
        
        # Validate inputs
        assert len(X) == len(y_true) == len(y_proba) == len(shap_values), \
            "X, y_true, y_proba, and shap_values must have the same length"
        assert list(X.columns) == feature_names, \
            "X.columns must match feature_names"
        
        # Identify groups
        self.rfp_mask = (y_proba >= high_risk_threshold) & (y_true == 0)
        self.tp_mask = (y_proba >= true_positive_threshold) & (y_true == 1)
        self.tn_mask = (y_proba < 0.30) & (y_true == 0)
        
        self.rfp_indices = np.where(self.rfp_mask)[0]
        self.tp_indices = np.where(self.tp_mask)[0]
        self.tn_indices = np.where(self.tn_mask)[0]
        
        try:
            print(f"╔════════════════════════════════════════════════════════════════╗")
            print(f"║  REVIEW-FLAGGED PROVIDER ANALYSIS                       ║")
            print(f"╚════════════════════════════════════════════════════════════════╝")
            print(f"Total providers analyzed:     {len(X):,}")
            print(f"Review-Flagged Providers (RFP):  {len(self.rfp_indices):,}  "
                  f"(P(fraud) ≥ {high_risk_threshold:.0%}, label=0)")
            print(f"True Positives (TP):          {len(self.tp_indices):,}  "
                  f"(P(fraud) ≥ {true_positive_threshold:.0%}, label=1)")
            print(f"True Negatives (TN):          {len(self.tn_indices):,}  "
                  f"(P(fraud) < 30%, label=0)")
            print()
        except (BrokenPipeError, OSError):
            pass  # stdout unavailable (e.g. Streamlit context)
    
    
    def get_top_shap_features(
        self,
        group_mask: np.ndarray,
        top_k: int = 10,
        min_shap_threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Get top SHAP features for a provider group.
        
        Parameters
        ----------
        group_mask : np.ndarray
            Boolean mask selecting providers in the group
        top_k : int, default=10
            Number of top features to return
        min_shap_threshold : float, default=0.0
            Minimum mean SHAP value to be included
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - feature_name: Feature name
            - mean_shap_value: Average SHAP contribution across group
            - median_shap_value: Median SHAP contribution
            - std_shap_value: Standard deviation of SHAP contributions
            - pct_providers_positive: % of providers where SHAP > 0
            - rank: Feature rank (1 = highest mean SHAP)
        """
        group_shap = self.shap_values[group_mask]
        
        results = []
        for i, fname in enumerate(self.feature_names):
            shap_col = group_shap[:, i]
            mean_shap = shap_col.mean()
            
            if mean_shap >= min_shap_threshold:
                results.append({
                    'feature_name': fname,
                    'mean_shap_value': mean_shap,
                    'median_shap_value': np.median(shap_col),
                    'std_shap_value': shap_col.std(),
                    'pct_providers_positive': (shap_col > 0).mean() * 100
                })
        
        df = pd.DataFrame(results).sort_values('mean_shap_value', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        return df.head(top_k)
    
    
    def compare_feature_distributions(
        self,
        feature_name: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare feature distributions: RFP vs TP vs TN.
        
        Parameters
        ----------
        feature_name : str
            Name of the feature to compare
        
        Returns
        -------
        Dict[str, Dict[str, float]]
            Nested dictionary with statistics for each group:
            {
                'review_flagged_providers': {'mean': ..., 'median': ..., ...},
                'true_positive': {...},
                'true_negative': {...},
                'statistical_tests': {
                    'rfp_vs_tp_ks_statistic': ...,
                    'rfp_vs_tp_ks_pvalue': ...,
                    'rfp_vs_tn_ks_statistic': ...,
                    'rfp_vs_tn_ks_pvalue': ...
                }
            }
        """
        feature_idx = self.feature_names.index(feature_name)
        
        rfp_vals = self.X.iloc[self.rfp_indices, feature_idx].values
        tp_vals = self.X.iloc[self.tp_indices, feature_idx].values
        tn_vals = self.X.iloc[self.tn_indices, feature_idx].values
        
        def stats(vals):
            return {
                'mean': float(np.mean(vals)),
                'median': float(np.median(vals)),
                'std': float(np.std(vals)),
                'min': float(np.min(vals)),
                'max': float(np.max(vals)),
                'p25': float(np.percentile(vals, 25)),
                'p50': float(np.percentile(vals, 50)),
                'p75': float(np.percentile(vals, 75)),
                'p90': float(np.percentile(vals, 90)),
                'p95': float(np.percentile(vals, 95))
            }
        
        # Kolmogorov-Smirnov tests
        ks_rfp_tp = ks_2samp(rfp_vals, tp_vals)
        ks_rfp_tn = ks_2samp(rfp_vals, tn_vals)
        
        return {
            'review_flagged_providers': stats(rfp_vals),
            'true_positive': stats(tp_vals),
            'true_negative': stats(tn_vals),
            'statistical_tests': {
                'rfp_vs_tp_ks_statistic': float(ks_rfp_tp.statistic),
                'rfp_vs_tp_ks_pvalue': float(ks_rfp_tp.pvalue),
                'rfp_vs_tn_ks_statistic': float(ks_rfp_tn.statistic),
                'rfp_vs_tn_ks_pvalue': float(ks_rfp_tn.pvalue),
                'interpretation': (
                    f"RFP vs TP: {'Similar' if ks_rfp_tp.pvalue > 0.05 else 'Different'} distribution "
                    f"(p={ks_rfp_tp.pvalue:.4f}) | "
                    f"RFP vs TN: {'Similar' if ks_rfp_tn.pvalue > 0.05 else 'Different'} distribution "
                    f"(p={ks_rfp_tn.pvalue:.4f})"
                )
            }
        }
    
    
    def calculate_similarity_to_fraud(self) -> pd.Series:
        """
        For each review-flagged provider, calculate average cosine similarity
        to confirmed fraud cases (based on SHAP patterns).
        
        Returns
        -------
        pd.Series
            Series indexed by review-flagged provider IDs, values = avg similarity (0-1)
        """
        if len(self.tp_indices) == 0:
            print("WARNING: No true positives found for similarity calculation")
            return pd.Series(
                np.zeros(len(self.rfp_indices)),
                index=self.X.index[self.rfp_mask],
                name='avg_similarity_to_fraud'
            )
        
        rfp_shap = self.shap_values[self.rfp_mask]
        tp_shap = self.shap_values[self.tp_mask]
        
        # Compute pairwise cosine similarity
        similarities = cosine_similarity(rfp_shap, tp_shap)
        
        # Average similarity to all confirmed fraud cases
        avg_sim = similarities.mean(axis=1)
        
        return pd.Series(
            avg_sim,
            index=self.X.index[self.rfp_mask],
            name='avg_similarity_to_fraud'
        )
    
    
    def calculate_investigation_priority(
        self,
        weight_proba: float = 0.40,
        weight_shap: float = 0.30,
        weight_similarity: float = 0.30
    ) -> pd.DataFrame:
        """
        Calculate investigation priority score for review-flagged providers.
        
        Priority Score = weighted combination of:
        1. Fraud probability (default: 40%)
        2. SHAP explanation strength (default: 30%)
        3. Similarity to confirmed fraud (default: 30%)
        
        Parameters
        ----------
        weight_proba : float, default=0.40
            Weight for fraud probability
        weight_shap : float, default=0.30
            Weight for SHAP strength
        weight_similarity : float, default=0.30
            Weight for similarity to fraud
        
        Returns
        -------
        pd.DataFrame
            Ranked DataFrame with columns:
            - provider_id: Provider identifier
            - fraud_probability: Model fraud probability
            - shap_strength: Total absolute SHAP contribution
            - similarity_to_fraud: Avg similarity to confirmed fraud (0-1)
            - priority_score: Weighted priority score (0-100)
            - priority_rank: Rank (1 = highest priority)
        """
        assert abs(weight_proba + weight_shap + weight_similarity - 1.0) < 0.01, \
            "Weights must sum to 1.0"
        
        rfp_ids = self.X.index[self.rfp_mask]
        rfp_probas = self.y_proba[self.rfp_mask]
        rfp_shap = self.shap_values[self.rfp_mask]
        
        # SHAP strength = total absolute contribution
        shap_strength = np.abs(rfp_shap).sum(axis=1)
        
        # Similarity to fraud
        similarity = self.calculate_similarity_to_fraud().values
        
        # Normalize to 0-1 range
        def normalize(x):
            if x.max() == x.min():
                return np.ones_like(x)
            return (x - x.min()) / (x.max() - x.min())
        
        norm_proba = normalize(rfp_probas)
        norm_shap = normalize(shap_strength)
        norm_sim = normalize(similarity)
        
        # Weighted priority score (scaled to 0-100)
        priority = (
            weight_proba * norm_proba +
            weight_shap * norm_shap +
            weight_similarity * norm_sim
        ) * 100
        
        df = pd.DataFrame({
            'provider_id': rfp_ids,
            'fraud_probability': rfp_probas,
            'shap_strength': shap_strength,
            'similarity_to_fraud': similarity,
            'priority_score': priority
        }).sort_values('priority_score', ascending=False).reset_index(drop=True)
        
        df['priority_rank'] = range(1, len(df) + 1)
        
        return df
    
    
    def generate_group_summary(self) -> Dict:
        """
        Generate comprehensive summary comparing RFP vs TP groups.
        
        Returns
        -------
        Dict
            Summary dictionary with keys:
            - group_sizes: Count of providers in each group
            - avg_fraud_probability: Mean fraud probability by group
            - top_shap_features: Top 10 SHAP features for each group
            - feature_overlap: Common vs unique top features
            - similarity_stats: Distribution of similarity scores
        """
        rfp_top_features = self.get_top_shap_features(self.rfp_mask, top_k=10)
        tp_top_features = self.get_top_shap_features(self.tp_mask, top_k=10)
        
        rfp_set = set(rfp_top_features['feature_name'].values)
        tp_set = set(tp_top_features['feature_name'].values)
        
        similarity_scores = self.calculate_similarity_to_fraud()
        
        return {
            'group_sizes': {
                'review_flagged_providers': int(self.rfp_mask.sum()),
                'true_positive': int(self.tp_mask.sum()),
                'true_negative': int(self.tn_mask.sum())
            },
            'avg_fraud_probability': {
                'review_flagged_providers': float(self.y_proba[self.rfp_mask].mean()),
                'true_positive': float(self.y_proba[self.tp_mask].mean()),
                'true_negative': float(self.y_proba[self.tn_mask].mean())
            },
            'top_shap_features': {
                'review_flagged_providers': rfp_top_features.to_dict('records'),
                'true_positive': tp_top_features.to_dict('records')
            },
            'feature_overlap': {
                'common_features': sorted(list(rfp_set & tp_set)),
                'rfp_unique': sorted(list(rfp_set - tp_set)),
                'tp_unique': sorted(list(tp_set - rfp_set)),
                'jaccard_similarity': len(rfp_set & tp_set) / len(rfp_set | tp_set)
            },
            'similarity_stats': {
                'mean': float(similarity_scores.mean()),
                'median': float(similarity_scores.median()),
                'std': float(similarity_scores.std()),
                'min': float(similarity_scores.min()),
                'max': float(similarity_scores.max()),
                'p25': float(similarity_scores.quantile(0.25)),
                'p75': float(similarity_scores.quantile(0.75))
            }
        }
    
    
    def get_provider_details(self, provider_id: int) -> Dict:
        """
        Get detailed information for a specific review-flagged provider.
        
        Parameters
        ----------
        provider_id : int
            Provider identifier
        
        Returns
        -------
        Dict
            Provider details including:
            - fraud_probability
            - priority_score
            - priority_rank
            - top_shap_features (top 10)
            - similarity_to_fraud
            - feature_values (all features)
        """
        if provider_id not in self.X.index[self.rfp_mask]:
            raise ValueError(f"Provider {provider_id} is not in the review-flagged group")
        
        priority_df = self.calculate_investigation_priority()
        provider_row = priority_df[priority_df['provider_id'] == provider_id].iloc[0]
        
        provider_idx = self.X.index.get_loc(provider_id)
        provider_shap = self.shap_values[provider_idx]
        
        # Top SHAP features for this provider
        shap_df = pd.DataFrame({
            'feature_name': self.feature_names,
            'shap_value': provider_shap,
            'feature_value': self.X.loc[provider_id].values
        }).sort_values('shap_value', ascending=False)
        
        return {
            'provider_id': int(provider_id),
            'fraud_probability': float(provider_row['fraud_probability']),
            'priority_score': float(provider_row['priority_score']),
            'priority_rank': int(provider_row['priority_rank']),
            'similarity_to_fraud': float(provider_row['similarity_to_fraud']),
            'top_shap_features': shap_df.head(10).to_dict('records'),
            'all_features': self.X.loc[provider_id].to_dict()
        }


def generate_investigator_summary(
    provider_id: int,
    fraud_prob: float,
    priority_score: float,
    priority_rank: int,
    top_shap_features: List[Dict],
    similarity_to_fraud: float,
    total_rfp: int
) -> str:
    """
    Generate human-readable investigation summary for review-flagged provider.
    
    Parameters
    ----------
    provider_id : int
        Provider identifier
    fraud_prob : float
        Fraud probability (0-1)
    priority_score : float
        Investigation priority score (0-100)
    priority_rank : int
        Rank in investigation queue
    top_shap_features : List[Dict]
        Top SHAP features with 'feature_name' and 'shap_value' keys
    similarity_to_fraud : float
        Similarity to confirmed fraud (0-1)
    total_rfp : int
        Total number of review-flagged providers
    
    Returns
    -------
    str
        Multi-line investigation summary
    """
    
    priority_tier = (
        "🔴 CRITICAL" if priority_score >= 90 else
        "🟠 HIGH" if priority_score >= 70 else
        "🟡 MEDIUM" if priority_score >= 50 else
        "🟢 LOW"
    )
    
    summary = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  INVESTIGATION PRIORITY ALERT — Provider {provider_id}                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

CLASSIFICATION STATUS:
  • Ground Truth Label:       NON-FRAUD (per available data)
  • Model Fraud Probability:  {fraud_prob:.1%}
  • Investigation Priority:   {priority_score:.1f}/100  {priority_tier}
  • Queue Rank:               #{priority_rank} of {total_rfp} review-flagged providers

⚠️  INVESTIGATIVE NOTE:
This provider is NOT confirmed as fraudulent in the current dataset. However,
the ML model has assigned an extremely high fraud risk score ({fraud_prob:.1%}),
placing it in the top {100 * priority_rank / total_rfp:.0f}% of model confidence.

High-confidence model flags on non-fraud-labeled providers may indicate:
  1. Undetected fraud (not yet investigated or proven)
  2. Emerging fraud patterns not captured in training labels
  3. Legitimate edge cases requiring human expert judgment
  4. Data labeling uncertainty or incomplete ground truth

SIMILARITY TO CONFIRMED FRAUD CASES: {similarity_to_fraud:.1%}
The provider's billing pattern (as measured by SHAP explanations) shows
{similarity_to_fraud:.1%} similarity to confirmed fraud cases, suggesting
{"strong" if similarity_to_fraud >= 0.70 else "moderate" if similarity_to_fraud >= 0.50 else "weak"}
structural similarities in risk drivers.

TOP RISK DRIVERS (SHAP Analysis):
"""
    
    for i, feat in enumerate(top_shap_features[:5], 1):
        feat_name = feat.get('feature_name', 'Unknown')
        shap_val = feat.get('shap_value', 0)
        feat_val = feat.get('feature_value', 'N/A')
        
        summary += f"  {i}. {feat_name}\n"
        summary += f"     SHAP contribution: {shap_val:+.4f}  |  Value: {feat_val}\n"
    
    summary += f"""
RECOMMENDED INVESTIGATIVE ACTIONS:
  1. Manual review of high-dollar claims and unusual billing patterns
  2. Cross-reference with OIG exclusion lists and prior investigations
  3. Compare billing rates to peer providers (same specialty/region)
  4. Assess for Medicare policy violations related to top risk drivers
  5. If evidence warrants, escalate to UPIC for formal audit

IMPORTANT LEGAL & ETHICAL CONSIDERATIONS:
  • This alert is a DECISION-SUPPORT TOOL, not evidence of fraud
  • All investigative conclusions must be based on documentary evidence
  • Providers have due process rights and presumption of legitimacy
  • High model scores alone do NOT constitute proof of wrongdoing
  • Human expert review is REQUIRED before any enforcement action

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
System: Review-Flagged Provider Analysis v1.0
"""
    
    return summary


# ── Utility Functions ─────────────────────────────────────────────────────────

def export_investigation_queue(
    analyzer: ReviewFlaggedProvidersAnalyzer,
    output_path: str,
    top_n: Optional[int] = None
) -> None:
    """
    Export investigation priority queue to CSV.
    
    Parameters
    ----------
    analyzer : ReviewFlaggedProvidersAnalyzer
        Initialized analyzer object
    output_path : str
        Path to save CSV file
    top_n : int, optional
        Number of top providers to export (None = all)
    """
    priority_df = analyzer.calculate_investigation_priority()
    
    if top_n:
        priority_df = priority_df.head(top_n)
    
    priority_df.to_csv(output_path, index=False)
    print(f"✓ Investigation queue exported: {output_path}")
    print(f"  {len(priority_df)} providers included")


def create_comparison_report(
    analyzer: ReviewFlaggedProvidersAnalyzer,
    output_path: str
) -> None:
    """
    Create detailed comparison report (RFP vs TP).
    
    Parameters
    ----------
    analyzer : ReviewFlaggedProvidersAnalyzer
        Initialized analyzer object
    output_path : str
        Path to save report (text file)
    """
    summary = analyzer.generate_group_summary()
    
    report = f"""
REVIEW-FLAGGED PROVIDER ANALYSIS REPORT
{'=' * 80}

1. GROUP SIZES
   Review-Flagged Providers (RFP):  {summary['group_sizes']['review_flagged_providers']:,}
   True Positives (TP):          {summary['group_sizes']['true_positive']:,}
   True Negatives (TN):          {summary['group_sizes']['true_negative']:,}

2. AVERAGE FRAUD PROBABILITY
   RFP:  {summary['avg_fraud_probability']['review_flagged_providers']:.1%}
   TP:   {summary['avg_fraud_probability']['true_positive']:.1%}
   TN:   {summary['avg_fraud_probability']['true_negative']:.1%}

3. TOP SHAP FEATURES — REVIEW-FLAGGED
"""
    
    for feat in summary['top_shap_features']['review_flagged_providers'][:10]:
        report += f"   {feat['rank']:2d}. {feat['feature_name']:40s}  {feat['mean_shap_value']:+.4f}\n"
    
    report += f"""
4. TOP SHAP FEATURES — TRUE POSITIVES
"""
    
    for feat in summary['top_shap_features']['true_positive'][:10]:
        report += f"   {feat['rank']:2d}. {feat['feature_name']:40s}  {feat['mean_shap_value']:+.4f}\n"
    
    report += f"""
5. FEATURE OVERLAP ANALYSIS
   Common features (both RFP & TP):  {len(summary['feature_overlap']['common_features'])}
     {', '.join(summary['feature_overlap']['common_features'])}
   
   Unique to RFP:  {len(summary['feature_overlap']['rfp_unique'])}
     {', '.join(summary['feature_overlap']['rfp_unique']) if summary['feature_overlap']['rfp_unique'] else 'None'}
   
   Unique to TP:   {len(summary['feature_overlap']['tp_unique'])}
     {', '.join(summary['feature_overlap']['tp_unique']) if summary['feature_overlap']['tp_unique'] else 'None'}
   
   Jaccard Similarity:  {summary['feature_overlap']['jaccard_similarity']:.2%}

6. SIMILARITY TO FRAUD DISTRIBUTION
   Mean:    {summary['similarity_stats']['mean']:.4f}
   Median:  {summary['similarity_stats']['median']:.4f}
   Std:     {summary['similarity_stats']['std']:.4f}
   Range:   [{summary['similarity_stats']['min']:.4f}, {summary['similarity_stats']['max']:.4f}]

{'=' * 80}
Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Comparison report saved: {output_path}")


if __name__ == "__main__":
    print("Review-Flagged Provider Analysis Module")
    print("Usage: Import this module in your Jupyter notebook or Python script")
    print("\nExample:")
    print("  from review_flagged_providers_analysis import ReviewFlaggedProvidersAnalyzer")
    print("  analyzer = ReviewFlaggedProvidersAnalyzer(X, y_true, y_proba, shap_values, feature_names)")
    print("  priority_queue = analyzer.calculate_investigation_priority()")
