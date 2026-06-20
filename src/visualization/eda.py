"""
src/visualization/eda.py
────────────────────────
Exploratory data analysis (EDA) and plots.
Pipeline.md section 5.
"""

import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from configs.config import (
    FIGURES_DIR, CARDIO_17, THYROID_RAW9,
    CONTINUOUS_FEATURES, FEATURE_SETS,
)

# Global style
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150


# ─── Cohort Flow ───────────────────────────────────────────────────────

def cohort_flow(df_full: pd.DataFrame, df_strict: pd.DataFrame,
                df_competing: pd.DataFrame, save: bool = True) -> dict:
    """
    Produce cohort flow diagram numbers.
    Returns a dictionary with counts.
    """
    from configs.config import HORIZON_DAYS

    n_total = len(df_full)

    cvd_within = ((df_full["event_cvd"] == 1) &
                  (df_full["time_days"] <= HORIZON_DAYS)).sum()
    event_free = ((df_full["time_days"] >= HORIZON_DAYS) &
                  ~((df_full["event_cvd"] == 1) &
                    (df_full["time_days"] <= HORIZON_DAYS))).sum()

    # Censored: alive with FU < 7y
    alive_short = ((df_full["event_death"] == 0) &
                   (df_full["time_days"] < HORIZON_DAYS)).sum()
    # Censored: non-CVD death < 7y
    noncvd_early = ((df_full["event_noncvd"] == 1) &
                    (df_full["time_days"] < HORIZON_DAYS)).sum()
    censored = alive_short + noncvd_early

    flow = {
        "Total patients": n_total,
        "CVD death <=7y (y7=1)": int(cvd_within),
        "Event-free >=7y (y7=0)": int(event_free),
        "Censored (total)": int(censored),
        "  Alive, FU <7y": int(alive_short),
        "  Non-CVD death <7y": int(noncvd_early),
        "Strict cohort": len(df_strict),
        "Competing cohort": len(df_competing),
    }

    print("\n" + "=" * 50)
    print("COHORT FLOW")
    print("=" * 50)
    for k, v in flow.items():
        print(f"  {k:30s}: {v:>6d}")

    if save:
        # Save as text-based figure
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis("off")
        text = "\n".join(f"{k}: {v}" for k, v in flow.items())
        ax.text(0.1, 0.5, text, fontsize=12, fontfamily="monospace",
                verticalalignment="center",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax.set_title("Cohort Flow", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "cohort_flow.png", bbox_inches="tight")
        plt.close(fig)

    return flow


# ─── Missingness ───────────────────────────────────────────────────────

def check_missingness(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """Check missingness in main features."""
    all_feats = sorted(set(CARDIO_17 + THYROID_RAW9))
    available = [f for f in all_feats if f in df.columns]

    miss = df[available].isnull().sum()
    miss_pct = 100 * miss / len(df)
    report = pd.DataFrame({"missing": miss, "pct": miss_pct})
    report = report[report["missing"] > 0].sort_values("missing", ascending=False)

    print(f"\n[Missingness] {label}: {len(df)} rows")
    if len(report) == 0:
        print("  No missing values in the 26 main features.")
    else:
        print(report.to_string())

    return report


# ─── Univariate association with target ────────────────────────────────

def univariate_association(df: pd.DataFrame, target: str = "y7",
                           cohort_name: str = "strict",
                           save: bool = True) -> pd.DataFrame:
    """
    Univariate association with target (strict cohort).
    Continuous: Mann-Whitney U + rank-biserial r
    Binary: odds ratio + chi-squared
    """
    cont_feats = ["Age", "Vessels", "fe", "TSH", "fT3", "fT4"]
    bin_feats = [f for f in CARDIO_17 + THYROID_RAW9
                 if f not in cont_feats and f in df.columns]

    results = []

    # Continuous features
    for feat in cont_feats:
        if feat not in df.columns:
            continue
        grp0 = df.loc[df[target] == 0, feat].dropna()
        grp1 = df.loc[df[target] == 1, feat].dropna()
        if len(grp0) == 0 or len(grp1) == 0:
            continue
        stat, pval = stats.mannwhitneyu(grp0, grp1, alternative="two-sided")
        # Rank-biserial r
        n0, n1 = len(grp0), len(grp1)
        r = 1 - (2 * stat) / (n0 * n1)
        results.append({
            "feature": feat, "type": "continuous",
            "test": "Mann-Whitney U", "statistic": stat,
            "p_value": pval, "effect_size": abs(r),
            "effect_name": "rank-biserial |r|",
        })

    # Binary features
    for feat in bin_feats:
        if feat not in df.columns:
            continue
        ct = pd.crosstab(df[feat], df[target])
        if ct.shape != (2, 2):
            continue
        chi2, pval, _, _ = stats.chi2_contingency(ct)
        # Odds ratio
        a, b = ct.iloc[1, 1], ct.iloc[1, 0]
        c, d = ct.iloc[0, 1], ct.iloc[0, 0]
        odds_r = (a * d) / (b * c) if (b * c) > 0 else np.inf
        results.append({
            "feature": feat, "type": "binary",
            "test": "chi2", "statistic": chi2,
            "p_value": pval, "effect_size": odds_r,
            "effect_name": "OR",
        })

    res_df = pd.DataFrame(results).sort_values("effect_size", ascending=False)
    res_df.insert(0, "cohort", cohort_name)

    if save:
        res_df.to_csv(FIGURES_DIR.parent / "univariate_association.csv",
                       index=False)
    print(f"\n[Univariate] {len(res_df)} features analysed.")
    return res_df


# ─── Violin plots of continuous features by class ─────────────────────

def plot_continuous_by_class(df: pd.DataFrame, target: str = "y7",
                             cohort_name: str = "strict",
                             save: bool = True):
    """Violin / histogram plots of continuous features by class."""
    cont_feats = [f for f in ["Age", "Vessels", "fe", "TSH", "fT3", "fT4"]
                  if f in df.columns]
    n = len(cont_feats)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, feat in enumerate(cont_feats):
        ax = axes[i]
        data = df[[feat, target]].dropna()
        sns.violinplot(x=target, y=feat, data=data, ax=ax,
                       palette=["#4CAF50", "#F44336"],
                       inner="quartile", cut=0)
        ax.set_xlabel("CVD Death <=7y", fontsize=10)
        ax.set_ylabel(feat, fontsize=11)
        ax.set_title(feat, fontsize=12, fontweight="bold")
        ax.set_xticklabels(["Survived", "CVD Death"])

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Continuous Feature Distributions by Class",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / f"violins_continuous_{cohort_name}.png",
                    bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] Continuous feature violins saved.")


# ─── Dendrogram + Spearman Heatmap ────────────────────────────────────

def plot_correlation_cluster(df: pd.DataFrame, cohort_name: str = "strict",
                             save: bool = True):
    """
    Hierarchical clustering dendrogram on 1-|Spearman| distance
    and Spearman correlation heatmap ordered by cluster.
    """
    all_feats = sorted(set(CARDIO_17 + THYROID_RAW9))
    available = [f for f in all_feats if f in df.columns]
    corr = df[available].corr(method="spearman")

    # Distance = 1 - |r|
    dist = 1 - corr.abs()
    # Condensed distance matrix
    from scipy.spatial.distance import squareform
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method="average")

    # Dendrogram
    fig, ax = plt.subplots(figsize=(14, 6))
    dendrogram(Z, labels=available, ax=ax, leaf_rotation=90,
               leaf_font_size=9, color_threshold=0.7)
    ax.set_ylabel("1 - |Spearman rho|", fontsize=11)
    ax.set_title("Hierarchical Clustering of Features (1-|Spearman| distance)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / f"dendrogram_features_{cohort_name}.png",
                    bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] Feature dendrogram saved.")

    # Cluster ordering
    from scipy.cluster.hierarchy import leaves_list
    order = leaves_list(Z)
    ordered_feats = [available[i] for i in order]

    # Spearman heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr.loc[ordered_feats, ordered_feats],
                annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
                annot_kws={"fontsize": 7})
    ax.set_title("Spearman Correlation (cluster-ordered)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / f"heatmap_spearman_{cohort_name}.png",
                    bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] Spearman heatmap saved.")


# ─── KMeans + PCA ─────────────────────────────────────────────────────

def plot_patient_clustering(df: pd.DataFrame, target: str = "y7",
                            cohort_name: str = "strict",
                            save: bool = True) -> dict:
    """
    KMeans on patients (standardised features), silhouette for k=2..6.
    PCA 2D projection coloured by cluster and by target.
    """
    all_feats = sorted(set(CARDIO_17 + THYROID_RAW9))
    available = [f for f in all_feats if f in df.columns]
    data = df[available + [target]].dropna()

    X = data[available].values
    y = data[target].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    # Silhouette for k=2..6
    silhouettes = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        labels = km.fit_predict(X_sc)
        sil = silhouette_score(X_sc, labels)
        silhouettes[k] = sil
        print(f"  KMeans k={k}: silhouette={sil:.3f}")

    best_k = max(silhouettes, key=silhouettes.get)
    print(f"  -> Best k={best_k} (silhouette={silhouettes[best_k]:.3f})")

    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    clusters = km_best.fit_predict(X_sc)

    # PCA 2D
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # By cluster
    scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters,
                               cmap="Set2", alpha=0.4, s=10)
    axes[0].set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
    axes[0].set_title(f"PCA -- KMeans (k={best_k})", fontweight="bold")
    plt.colorbar(scatter, ax=axes[0], label="Cluster")

    # By target
    colors = ["#4CAF50" if v == 0 else "#F44336" for v in y]
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=colors, alpha=0.4, s=10)
    axes[1].set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
    axes[1].set_title("PCA -- Target (CVD Death <=7y)", fontweight="bold")
    # Manual legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50',
               markersize=8, label='Survived'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336',
               markersize=8, label='CVD Death'),
    ]
    axes[1].legend(handles=legend_elements, loc="upper right")

    fig.suptitle("Patient Clustering", fontsize=14, fontweight="bold")
    fig.tight_layout()

    if save:
        fig.savefig(FIGURES_DIR / f"pca_clustering_{cohort_name}.png",
                    bbox_inches="tight")
    plt.close(fig)
    print("[PLOT] PCA + clustering saved.")

    return silhouettes


# ─── Entry point ────────────────────────────────────────────────────────

def run(df_full: pd.DataFrame, df_strict: pd.DataFrame,
        df_competing: pd.DataFrame, force: bool = False):
    """Run all EDA analyses."""

    print("\n" + "=" * 60)
    print("[EDA] Exploratory Data Analysis")
    print("=" * 60)

    # 1. Cohort flow
    flow = cohort_flow(df_full, df_strict, df_competing)

    univ_tables = []
    silhouettes = {}
    for cohort_name, cohort_df in [
            ("strict", df_strict), ("competing", df_competing)]:
        print(f"\n[EDA] Cohort: {cohort_name}")

        # 2. Missingness
        check_missingness(cohort_df, label=f"{cohort_name} cohort")

        # 3. Univariate association
        univ_tables.append(univariate_association(
            cohort_df, cohort_name=cohort_name, save=False))

        # 4. Violin plots
        plot_continuous_by_class(cohort_df, cohort_name=cohort_name)

        # 5-6. Dendrogram + heatmap
        plot_correlation_cluster(cohort_df, cohort_name=cohort_name)

        # 7. KMeans + PCA
        silhouettes[cohort_name] = plot_patient_clustering(
            cohort_df, cohort_name=cohort_name)

    univ = pd.concat(univ_tables, ignore_index=True)
    univ.to_csv(FIGURES_DIR.parent / "univariate_association.csv",
                index=False)

    print("\n[EDA] Complete.")
    return {"flow": flow, "univariate": univ, "silhouettes": silhouettes}
