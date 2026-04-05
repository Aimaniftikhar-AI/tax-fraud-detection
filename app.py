"""
Tax Fraud Detection System
==========================
Production-level Streamlit application for detecting tax fraud using ML.
Authors: Aiman Iftikhar & Memoona Sheikh
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
)
from imblearn.over_sampling import SMOTE


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tax Fraud Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background: #080c14;
    color: #ffffff;
}
.stApp { background: #080c14; color: #ffffff; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #0d1120; }
::-webkit-scrollbar-thumb { background: #1e4a7a; border-radius: 4px; }

/* Global white text */
p, span, div, label, li, td, th { color: #ffffff; }
.stMarkdown, .stMarkdown p, .stMarkdown li { color: #ffffff !important; }
.stText, .stCaption { color: #ccddee !important; }
[data-testid="stDataFrame"] * { color: #ffffff !important; }
[data-testid="stExpander"] p,
[data-testid="stExpander"] li,
[data-testid="stExpander"] span { color: #ffffff !important; }
.stSelectbox label, .stNumberInput label,
.stSlider label, .stFileUploader label { color: #ffffff !important; }

/* HERO */
.hero {
    background: linear-gradient(135deg, #0b1525 0%, #0d1f35 60%, #091528 100%);
    border: 1px solid #1a3a5c;
    border-radius: 20px;
    padding: 3rem 3.5rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    bottom: -80px; right: -80px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,180,255,0.08) 0%, transparent 65%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #2e8ecf;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #e8f4ff;
    line-height: 1.15;
    margin: 0 0 0.5rem;
    letter-spacing: -0.5px;
}
.hero-title span { color: #38bdf8; }
.hero-sub { color: #7aaac8; font-size: 0.9rem; margin-bottom: 1.2rem; }
.pill {
    display: inline-block;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.25);
    color: #38bdf8;
    border-radius: 30px;
    padding: 3px 13px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 3px 4px 3px 0;
}

/* SECTION HEADER */
.sec-header { display: flex; align-items: center; gap: 12px; margin: 2rem 0 1.2rem; }
.sec-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 600;
    color: #38bdf8; background: #0a1e30;
    border: 1px solid #1a3d5c; border-radius: 6px;
    padding: 2px 8px; letter-spacing: 1px;
}
.sec-title { font-size: 1.05rem; font-weight: 600; color: #ffffff; }
.sec-line { flex: 1; height: 1px; background: linear-gradient(to right, #1a3d5c, transparent); }

/* STAT CARDS */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 12px; margin-bottom: 1.5rem;
}
.stat-card {
    background: #0c1828; border: 1px solid #162a40;
    border-radius: 14px; padding: 1.2rem 1.4rem;
    position: relative; overflow: hidden;
}
.stat-card::before {
    content: ""; position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #1a5078, #38bdf8, transparent);
}
.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1.2px;
    color: #7ab8d8; margin-bottom: 6px;
}
.stat-value { font-size: 1.8rem; font-weight: 700; color: #38bdf8; line-height: 1; }
.stat-sub { font-size: 0.72rem; color: #7ab0cc; margin-top: 4px; }

/* INFO / ALERT */
.info-block {
    background: #0c1828; border: 1px solid #162a40;
    border-radius: 12px; padding: 1rem 1.3rem;
    margin-bottom: 0.8rem; font-size: 0.88rem;
    color: #c8e0f4; line-height: 1.6;
}
.info-block strong { color: #ffffff; }
.alert-warn {
    background: #180e05; border: 1px solid #7c4a10;
    border-left: 3px solid #f59e0b; border-radius: 10px;
    padding: 0.9rem 1.2rem; color: #fde68a;
    font-size: 0.88rem; margin: 0.5rem 0;
}
.alert-warn strong { color: #ffffff; font-weight: 700; }
.alert-ok {
    background: #031a0f; border: 1px solid #0d5c30;
    border-left: 3px solid #10b981; border-radius: 10px;
    padding: 0.9rem 1.2rem; color: #6ee7b7;
    font-size: 0.88rem; margin: 0.5rem 0;
}
.alert-ok strong { color: #ffffff; font-weight: 700; }

/* BEST MODEL */
.best-card {
    background: linear-gradient(135deg, #031a0f 0%, #052e1a 100%);
    border: 1px solid #0d5c30; border-radius: 18px;
    padding: 2.5rem; text-align: center;
    margin: 1rem auto; max-width: 600px;
}
.best-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; letter-spacing: 2.5px;
    text-transform: uppercase; color: #10b981; margin-bottom: 0.6rem;
}
.best-name { font-size: 2.2rem; font-weight: 700; color: #34d399; margin-bottom: 1.5rem; }
.best-metrics { display: flex; justify-content: center; gap: 2.5rem; flex-wrap: wrap; }
.best-metric-item { text-align: center; }
.best-metric-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem; letter-spacing: 1.5px;
    text-transform: uppercase; color: #6ee7b7; margin-bottom: 4px;
}
.best-metric-val { font-size: 1.3rem; font-weight: 700; color: #ffffff; }

/* PREDICTION */
.pred-fraud {
    background: #180e05; border: 1px solid #b45309;
    border-radius: 16px; padding: 2rem; text-align: center;
}
.pred-ok {
    background: #031a0f; border: 1px solid #059669;
    border-radius: 16px; padding: 2rem; text-align: center;
}
.pred-icon { font-size: 2.5rem; }
.pred-label { font-size: 1.5rem; font-weight: 700; margin: 0.4rem 0; }
.pred-meta { font-size: 0.82rem; color: #aaccdd; }

/* STREAMLIT OVERRIDES */
[data-testid="stMetric"] {
    background: #0c1828 !important; border: 1px solid #162a40 !important;
    border-radius: 12px !important; padding: 1rem 1.3rem !important;
}
[data-testid="stMetricLabel"] { color: #aad0e8 !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 700 !important; }

.stButton > button {
    background: linear-gradient(135deg, #0e4272, #1264a3) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 10px !important; font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important; padding: 0.6rem 2rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1264a3, #1e80cc) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(30,128,204,0.3) !important;
}
[data-testid="stFileUploader"] {
    border: 1px dashed #1a3d5c !important;
    border-radius: 14px !important; background: #080c14 !important;
}
[data-testid="stDataFrame"] { border-radius: 10px !important; }
[data-testid="stExpander"] {
    background: #0c1828 !important; border: 1px solid #162a40 !important;
    border-radius: 10px !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #0d1526;
    border: 1px solid #162035; border-radius: 14px; padding: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px; color: #aac8e0 !important;
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 500; font-size: 0.85rem; padding: 8px 20px; border: none;
}
.stTabs [aria-selected="true"] {
    background: #0d2d4a !important; color: #38bdf8 !important;
    font-weight: 600 !important; border: 1px solid #1a5078 !important;
}
hr { border: none; border-top: 1px solid #111e33; margin: 1.5rem 0; }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CHART THEME
# ─────────────────────────────────────────────────────────────────────────────
BG   = "#0c1828"
GRID = "#162a40"
TEXT = "#aac8e0"
ACC  = "#38bdf8"
GRN  = "#10b981"
RED  = "#ef4444"
PRP  = "#818cf8"
ORG  = "#f472b6"

def _ax_style(ax, title=""):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.set_title(title, color=ACC, fontsize=10, fontweight="bold", pad=10)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)

def _fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(BG)
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────

def load_data(uploaded_file):
    df = pd.read_csv(uploaded_file)
    if "taxpayer_id" in df.columns:
        df.drop(columns=["taxpayer_id"], inplace=True)
    return df


def preprocess_data(df):
    summary = {}

    summary["missing_before"] = int(df.isnull().sum().sum())
    df.fillna(df.median(numeric_only=True), inplace=True)
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    summary["missing_after"] = int(df.isnull().sum().sum())

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if "fraud_flag" in cat_cols:
        cat_cols.remove("fraud_flag")
    summary["encoded_cols"] = cat_cols
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    summary["shape_after_encoding"] = df.shape

    target      = df["fraud_flag"]
    df_features = df.drop(columns=["fraud_flag"])
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    rows_before  = len(df_features)

    mask = pd.Series(True, index=df_features.index)
    outlier_counts = {}
    for col in numeric_cols:
        Q1, Q3 = df_features[col].quantile(0.25), df_features[col].quantile(0.75)
        IQR    = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        col_mask = (df_features[col] >= lo) & (df_features[col] <= hi)
        n_out    = int((~col_mask).sum())
        if n_out:
            outlier_counts[col] = n_out
        mask &= col_mask

    df_features = df_features[mask]
    target      = target[mask]
    summary["outliers_removed"]       = rows_before - len(df_features)
    summary["outlier_counts_per_col"] = outlier_counts

    scaler     = StandardScaler()
    X_scaled   = scaler.fit_transform(df_features)
    feat_names = df_features.columns.tolist()
    summary["n_features"] = len(feat_names)
    summary["n_samples"]  = len(df_features)

    return X_scaled, target.values, scaler, feat_names, summary, df_features


def apply_smote(X, y):
    sm = SMOTE(random_state=42)
    return sm.fit_resample(X, y)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LAYER
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree":       lambda: DecisionTreeClassifier(random_state=42,  max_depth=6,
    min_samples_split=10,
    min_samples_leaf=5),
    "Random Forest":       lambda: RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM":                 lambda: SVC(kernel="rbf", probability=True, random_state=42),
    "KNN":                 lambda: KNeighborsClassifier(n_neighbors=5),
}


def train_all(X_train, y_train):
    fitted = {}
    for name, factory in MODEL_REGISTRY.items():
        model = factory()
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def evaluate_all(fitted, X_train, X_test, y_train, y_test):
    records, preds_dict, overfit_dict = [], {}, {}
    for name, model in fitted.items():
        y_pred    = model.predict(X_test)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc  = accuracy_score(y_test, y_pred)
        records.append({
            "Model":     name,
            "Accuracy":  round(test_acc, 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        })
        preds_dict[name]   = y_pred
        overfit_dict[name] = (round(train_acc, 4), round(test_acc, 4))

    results_df = (
        pd.DataFrame(records)
        .sort_values("F1 Score", ascending=False)
        .reset_index(drop=True)
    )
    return results_df, preds_dict, overfit_dict

def select_best_model(results_df, overfit_dict):

    valid_models = []

    for _, row in results_df.iterrows():
        name = row["Model"]
        train_acc, test_acc = overfit_dict[name]

        if not is_overfitting(train_acc, test_acc):
            valid_models.append(row)

    # Agar koi non-overfitting model exist karta hai
    if len(valid_models) > 0:
        df = pd.DataFrame(valid_models)
        best = df.sort_values(by="F1 Score", ascending=False).iloc[0]
        return best

    # fallback (agar sab overfitting hon)
    return results_df.iloc[0]

def is_overfitting(train_acc, test_acc):
    """
    FIX: Improved overfitting detection.

    Decision Tree bina max_depth ke training data ko 100% memorize kar leta hai.
    Pehle sirf gap > 0.05 check hota tha — agar SMOTE se test bhi high ho
    toh gap chhota lagta tha aur DT ko "Good Fit" bol deta tha. WRONG.

    Naya logic:
      Rule 1 → Gap (train - test) > 0.05          → Overfitting
      Rule 2 → Train == 1.0 AND test < 0.98        → Overfitting (perfect memorization)
      Dono mein se ek bhi true → Overfitting flag
    """
    gap = train_acc - test_acc
    if gap > 0.05:
        return True
    if train_acc >= 1.0 and test_acc < 0.98:
        return True
    return False


def overfit_reason(train_acc, test_acc):
    """Return human-readable reason for overfitting."""
    gap = round(train_acc - test_acc, 4)
    if train_acc >= 1.0 and test_acc < 0.98:
        return f"Train=1.0 (memorization), Test={test_acc}"
    if gap > 0.05:
        return f"Gap={gap} > threshold 0.05"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def chart_class_dist(before, after):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.2))
    fig.patch.set_facecolor(BG)
    labels = ["Not Fraud", "Fraud"]
    ax1.bar(labels, [before.get("0", 0), before.get("1", 0)],
            color=[ACC, RED], edgecolor=GRID, width=0.4)
    _ax_style(ax1, "Before SMOTE")
    ax2.bar(labels, [after.get("0", 0), after.get("1", 0)],
            color=[ACC, GRN], edgecolor=GRID, width=0.4)
    _ax_style(ax2, "After SMOTE")
    plt.tight_layout(pad=2)
    return fig


def chart_model_comparison(results_df):
    fig, ax = _fig(11, 4)
    x = np.arange(len(results_df))
    w = 0.18
    for i, (m, c) in enumerate(zip(
        ["Accuracy", "Precision", "Recall", "F1 Score"],
        [ACC, PRP, ORG, GRN]
    )):
        ax.bar(x + i * w, results_df[m], width=w, label=m, color=c, edgecolor=GRID, alpha=0.9)
    ax.set_xticks(x + 1.5 * w)
    ax.set_xticklabels(results_df["Model"], rotation=15, ha="right", color=TEXT, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.legend(facecolor="#0d1526", edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    _ax_style(ax, "Model Performance Comparison")
    plt.tight_layout()
    return fig


def chart_overfit(overfit_dict):
    names  = list(overfit_dict.keys())
    trains = [v[0] for v in overfit_dict.values()]
    tests  = [v[1] for v in overfit_dict.values()]
    x      = np.arange(len(names))

    # Red bars = overfitting, blue/green = good
    train_colors = [RED if is_overfitting(tr, te) else ACC for tr, te in zip(trains, tests)]
    test_colors  = [ORG if is_overfitting(tr, te) else GRN for tr, te in zip(trains, tests)]

    fig, ax = _fig(10, 3.8)
    b1 = ax.bar(x - 0.2, trains, 0.35, color=train_colors, edgecolor=GRID, label="Train Acc")
    b2 = ax.bar(x + 0.2, tests,  0.35, color=test_colors,  edgecolor=GRID, label="Test Acc")

    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha='center', va='bottom', color=TEXT, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right", color=TEXT, fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.legend(facecolor="#0d1526", edgecolor=GRID, labelcolor=TEXT, fontsize=8)
    _ax_style(ax, "Train vs Test Accuracy  (Red = Overfitting)")
    plt.tight_layout()
    return fig


def chart_confusion(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                linewidths=0.5, linecolor=GRID, cbar=False,
                xticklabels=["Not Fraud", "Fraud"],
                yticklabels=["Not Fraud", "Fraud"],
                annot_kws={"size": 13, "weight": "bold", "color": "white"})
    ax.set_xlabel("Predicted", color=TEXT, fontsize=8)
    ax.set_ylabel("Actual",    color=TEXT, fontsize=8)
    _ax_style(ax, model_name)
    plt.tight_layout()
    return fig


def chart_outliers(outlier_counts):
    cols_o = list(outlier_counts.keys())
    vals_o = list(outlier_counts.values())
    fig, ax = plt.subplots(figsize=(8, max(3, len(cols_o) * 0.38)))
    fig.patch.set_facecolor(BG)
    ax.barh(cols_o, vals_o, color=ACC, edgecolor=GRID, height=0.55)
    ax.set_xlabel("Count", color=TEXT, fontsize=8)
    _ax_style(ax, "Outliers per Feature")
    plt.tight_layout()
    return fig


def chart_corr(df_clean, n_feats=10):
    top = df_clean.iloc[:, :min(n_feats, df_clean.shape[1])]
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(BG)
    sns.heatmap(top.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.3, linecolor=GRID, ax=ax,
                annot_kws={"size": 7}, cbar_kws={"shrink": 0.7})
    _ax_style(ax, "Feature Correlation Matrix")
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    return fig


def chart_prob_bar(not_fraud_pct, fraud_pct):
    fig, ax = plt.subplots(figsize=(6, 1.4))
    fig.patch.set_facecolor(BG)
    ax.barh(["Not Fraud"], [not_fraud_pct], color=GRN, edgecolor=GRID, height=0.4)
    ax.barh(["Fraud"],     [fraud_pct],     color=RED, edgecolor=GRID, height=0.4)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", color=TEXT, fontsize=8)
    _ax_style(ax)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def sec(num, title):
    st.markdown(
        f"<div class='sec-header'>"
        f"<span class='sec-num'>{num}</span>"
        f"<span class='sec-title'>{title}</span>"
        f"<div class='sec-line'></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def stat_cards(items):
    html = "<div class='stat-grid'>"
    for label, value, sub in items:
        html += (
            f"<div class='stat-card'>"
            f"<div class='stat-label'>{label}</div>"
            f"<div class='stat-value'>{value}</div>"
            f"<div class='stat-sub'>{sub}</div>"
            f"</div>"
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB RENDERERS
# ─────────────────────────────────────────────────────────────────────────────

def tab_upload():
    sec("01", "Upload Dataset")
    uploaded = st.file_uploader(
        "Drop your CSV file here · Expected: tax_fraud_detection_dataset.csv",
        type=["csv"],
    )
    if uploaded is None:
        st.markdown(
            "<div class='info-block'>⬆ Upload a CSV to begin the pipeline. "
            "The file must contain a <strong>fraud_flag</strong> column as the target.</div>",
            unsafe_allow_html=True,
        )
        return

    df = load_data(uploaded)
    st.session_state.df_raw   = df
    st.session_state.uploaded = True

    sec("02", "Dataset Preview")
    fraud_count = int(df["fraud_flag"].sum()) if "fraud_flag" in df.columns else "—"
    stat_cards([
        ("Total Rows",     f"{df.shape[0]:,}",             "samples"),
        ("Columns",        str(df.shape[1]),               "features"),
        ("Fraud Cases",    f"{fraud_count:,}",             "positive labels"),
        ("Missing Values", f"{df.isnull().sum().sum():,}", "before imputation"),
    ])

    st.markdown("**First 10 rows**")
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("📋 Column Types & Null Counts"):
        st.dataframe(pd.DataFrame({
            "Column":   df.columns,
            "dtype":    df.dtypes.astype(str).values,
            "Non-Null": df.notnull().sum().values,
            "Nulls":    df.isnull().sum().values,
            "Unique":   df.nunique().values,
        }), use_container_width=True)

    with st.expander("📊 Statistical Summary"):
        st.dataframe(df.describe().T.round(3), use_container_width=True)

    st.success("✅ Dataset loaded. Proceed to **Preprocessing** tab.")


def tab_preprocess():
    if "df_raw" not in st.session_state:
        st.info("Please upload a dataset first (Tab 1).")
        return

    sec("03", "Preprocessing Pipeline")

    with st.spinner("Running preprocessing…"):
        X_scaled, y, scaler, feat_names, summary, df_clean = preprocess_data(
            st.session_state.df_raw.copy()
        )

    st.session_state.X_scaled   = X_scaled
    st.session_state.y          = y
    st.session_state.scaler     = scaler
    st.session_state.feat_names = feat_names
    st.session_state.df_clean   = df_clean

    stat_cards([
        ("Nulls Before",   str(summary["missing_before"]),   "imputed"),
        ("Nulls After",    str(summary["missing_after"]),    "remaining"),
        ("Rows Removed",   str(summary["outliers_removed"]), "IQR outliers"),
        ("Final Features", str(summary["n_features"]),       f"{summary['n_samples']:,} samples"),
    ])

    for title, body in [
        ("Step 1 — Missing Value Imputation",
         f"Numeric → <strong>median</strong> &nbsp;·&nbsp; Categorical → <strong>mode</strong>.<br>"
         f"Nulls before: <strong>{summary['missing_before']}</strong> → after: <strong>{summary['missing_after']}</strong>"),
        ("Step 2 — One-Hot Encoding",
         f"Columns encoded: <strong>{summary['encoded_cols'] or 'None'}</strong><br>"
         f"Shape after encoding: <strong>{summary['shape_after_encoding']}</strong>"),
        ("Step 3 — IQR Outlier Removal",
         f"Fence = <strong>1.5 × IQR</strong>. Rows removed: <strong>{summary['outliers_removed']}</strong>"
         f" &nbsp;·&nbsp; Remaining: <strong>{summary['n_samples']}</strong>"),
        ("Step 4 — StandardScaler",
         f"All features scaled to <strong>mean=0, std=1</strong>. Features: <strong>{summary['n_features']}</strong>"),
    ]:
        with st.expander(f"🔧 {title}"):
            st.markdown(f"<div class='info-block'>{body}</div>", unsafe_allow_html=True)

    if summary["outlier_counts_per_col"]:
        with st.expander("🔎 Outliers per Column"):
            oc = (pd.DataFrame(summary["outlier_counts_per_col"].items(),
                               columns=["Column", "Outlier Rows"])
                  .sort_values("Outlier Rows", ascending=False))
            c1, c2 = st.columns([1, 2])
            with c1: st.dataframe(oc, use_container_width=True)
            with c2: st.pyplot(chart_outliers(summary["outlier_counts_per_col"]))

    with st.expander("🗺 Feature Correlation Heatmap"):
        st.pyplot(chart_corr(df_clean))

    st.success("✅ Preprocessing complete. Proceed to **SMOTE Balancing** tab.")


def tab_smote():
    if "X_scaled" not in st.session_state:
        st.info("Please run preprocessing first (Tab 2).")
        return

    sec("04", "Class Balancing — SMOTE")
    X, y = st.session_state.X_scaled, st.session_state.y

    unique_b, counts_b = np.unique(y, return_counts=True)
    before_dict = dict(zip(unique_b.astype(str), counts_b))

    with st.spinner("Applying SMOTE…"):
        X_bal, y_bal = apply_smote(X, y)

    unique_a, counts_a = np.unique(y_bal, return_counts=True)
    after_dict = dict(zip(unique_a.astype(str), counts_a))

    st.session_state.X_balanced = X_bal
    st.session_state.y_balanced = y_bal

    stat_cards([
        ("Samples Before", f"{len(y):,}",                   "original"),
        ("Samples After",  f"{len(y_bal):,}",               "after SMOTE"),
        ("Majority Class", f"{before_dict.get('0', 0):,}",  "Not Fraud"),
        ("Minority Class", f"{before_dict.get('1', 0):,}",  "Fraud (original)"),
    ])

    st.pyplot(chart_class_dist(before_dict, after_dict))
    st.markdown(
        f"<div class='alert-ok'>✅ SMOTE applied. Dataset expanded from "
        f"<strong>{len(y):,}</strong> → <strong>{len(y_bal):,}</strong> samples.</div>",
        unsafe_allow_html=True,
    )


def tab_train():
    if "X_balanced" not in st.session_state:
        st.info("Please complete SMOTE balancing first (Tab 3).")
        return

    sec("05", "Model Training")

    split_pct = st.slider("Train / Test split (train %)", 60, 90, 80, 5)
    st.caption(f"Training on **{split_pct}%** · Testing on **{100-split_pct}%** "
               f"of {len(st.session_state.y_balanced):,} samples.")

    if st.button("⚡ Train All 5 Models"):
        X_bal, y_bal = st.session_state.X_balanced, st.session_state.y_balanced
        X_train, X_test, y_train, y_test = train_test_split(
            X_bal, y_bal, test_size=(100-split_pct)/100,
            random_state=42, stratify=y_bal,
        )
        with st.spinner("Training models…"):
            fitted = train_all(X_train, y_train)
            results_df, preds_dict, overfit_dict = evaluate_all(
                fitted, X_train, X_test, y_train, y_test
            )
        st.session_state.update(dict(
            X_train=X_train, X_test=X_test,
            y_train=y_train, y_test=y_test,
            fitted=fitted, results_df=results_df,
            preds_dict=preds_dict, overfit_dict=overfit_dict,
            trained=True,
        ))
        st.success("✅ All models trained successfully!")

    if not st.session_state.get("trained"):
        st.info("Click **Train All 5 Models** to launch the pipeline.")
        return

    # ── Model Comparison ──────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    sec("06", "Model Comparison")

    rdf = st.session_state.results_df.copy()
    rdf["Overfitting"] = rdf["Model"].apply(
        lambda m: is_overfitting(*st.session_state.overfit_dict[m])
    )

    rdf = rdf.sort_values(by=["Overfitting", "F1 Score"], ascending=[True, False])
    rdf.insert(0, "Rank", [f"#{i+1}" for i in range(len(rdf))])
    st.dataframe(
        rdf.style
           .background_gradient(subset=["Accuracy","Precision","Recall","F1 Score"], cmap="Blues")
           .format({c: "{:.4f}" for c in ["Accuracy","Precision","Recall","F1 Score"]}),
        use_container_width=True, height=230,
    )
    st.pyplot(chart_model_comparison(st.session_state.results_df))

    # ── Overfitting Analysis ──────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    sec("07", "Overfitting Analysis")

    st.markdown(
        "<div class='info-block'>"
        "📌 <strong>Overfitting Detection — 2 Rules:</strong><br>"
        "&nbsp;&nbsp;① Gap (Train − Test) &gt; 0.05 → Overfitting<br>"
        "&nbsp;&nbsp;② Train Accuracy = 1.0 (model ne data memorize kiya) AND Test &lt; 0.98 → Overfitting<br>"
        "<strong>Decision Tree</strong> aksar Train=1.0 deta hai jo clear memorization hai — "
        "chahe test accuracy high ho, ye overfitting hai."
        "</div>",
        unsafe_allow_html=True,
    )

    rows = []
    for name, (tr, te) in st.session_state.overfit_dict.items():
        gap     = round(tr - te, 4)
        overfit = is_overfitting(tr, te)
        reason  = overfit_reason(tr, te)
        rows.append({
            "Model":     name,
            "Train Acc": tr,
            "Test Acc":  te,
            "Gap (Δ)":   gap,
            "Status":    f"⚠ Overfitting" if overfit else "✓ Good Fit",
            "Reason":    reason,
        })

    overfit_df = pd.DataFrame(rows)
    st.dataframe(overfit_df, use_container_width=True, hide_index=True)

    for _, row in overfit_df.iterrows():
        if "Overfitting" in row["Status"]:
            st.markdown(
                f"<div class='alert-warn'>⚠ <strong>{row['Model']}</strong> — "
                f"Train: <strong>{row['Train Acc']}</strong> · "
                f"Test: <strong>{row['Test Acc']}</strong> · "
                f"<em>{row['Reason']}</em></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='alert-ok'>✓ <strong>{row['Model']}</strong> — "
                f"Generalising well. Train: {row['Train Acc']} · Test: {row['Test Acc']}</div>",
                unsafe_allow_html=True,
            )

    st.pyplot(chart_overfit(st.session_state.overfit_dict))

def tab_results():
    if not st.session_state.get("trained"):
        st.info("Please train the models first (Tab 4).")
        return

    sec("08", "Best Model")

    # ── Select best model ─────────────────────────────
    best = select_best_model(
        st.session_state.results_df,
        st.session_state.overfit_dict
    )

    best_model_name = best["Model"]

    # Get full metrics row of that model
    best_model_row = st.session_state.results_df[
        st.session_state.results_df["Model"] == best_model_name
    ].iloc[0]

    # Map metrics
    best_value_map = {
        "Accuracy": best_model_row["Accuracy"],
        "F1 Score": best_model_row["F1 Score"],
        "Precision": best_model_row["Precision"],
        "Recall": best_model_row["Recall"],
    }

    # ── Center layout ─────────────────────────────
    _, mid, _ = st.columns([1, 2, 1])

    with mid:
        st.markdown(
            f"<div class='best-card'>"
            f"<div class='best-badge'>🏆 Best Performing Model</div>"
            f"<div class='best-name'>{best_model_name}</div>"
            f"<div class='best-metrics'>"
            + "".join(
                f"<div class='best-metric-item'>"
                f"<div class='best-metric-label'>{m}</div>"
                f"<div class='best-metric-val'>{best_value_map[m]:.4f}</div>"
                f"</div>"
                for m in ["Accuracy", "F1 Score", "Precision", "Recall"]
            )
            + "</div></div>",
            unsafe_allow_html=True,
        )

    # ── Classification Report ─────────────────────
    with st.expander(f"📄 Full Classification Report — {best_model_name}"):
        rpt = classification_report(
            st.session_state.y_test,
            st.session_state.preds_dict[best_model_name],
            target_names=["Not Fraud", "Fraud"],
        )
        st.code(rpt, language="text")

    # ── Confusion Matrices ────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    sec("09", "Confusion Matrices")
    st.caption("Row = actual class · Column = predicted class")

    names = list(st.session_state.preds_dict.keys())

    # First row (3 models)
    r1 = st.columns(3)
    for i, name in enumerate(names[:3]):
        with r1[i]:
            st.pyplot(
                chart_confusion(
                    st.session_state.y_test,
                    st.session_state.preds_dict[name],
                    name
                )
            )

    # Remaining models
    if len(names) > 3:
        _, c1, c2, _ = st.columns([0.5, 1, 1, 0.5])
        for col, name in zip([c1, c2], names[3:]):
            with col:
                st.pyplot(
                    chart_confusion(
                        st.session_state.y_test,
                        st.session_state.preds_dict[name],
                        name
                    )
                )

def tab_predict():
    if not st.session_state.get("trained"):
        st.info("Please train the models first (Tab 4).")
        return

    sec("10", "Real-Time Fraud Prediction")

    fitted     = st.session_state.fitted
    feat_names = st.session_state.feat_names
    df_clean   = st.session_state.df_clean
    scaler     = st.session_state.scaler
    df_raw     = st.session_state.df_raw
    best_name  = st.session_state.results_df.iloc[0]["Model"]

    col_sel, col_hint = st.columns([2, 1])
    with col_sel:
        chosen = st.selectbox("Select model for prediction", list(fitted.keys()))
    with col_hint:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption(f"🏆 Best model: **{best_name}**")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("Enter feature values")

    # ── FIX: No min/max restriction on inputs ──
    # Pehle min_value=col_min, max_value=col_max tha
    # jo user ko dataset range ke bahar value type nahi karne deta tha.
    # Ab sirf default (median) set hai, step aur format hai — baaki free.

    user_inputs   = {}
    original_cols = df_raw.columns.tolist()

    for i in range(0, len(feat_names), 4):
        chunk = feat_names[i : i + 4]
        cols  = st.columns(len(chunk))
        for col, feat in zip(cols, chunk):
            with col:
                is_ohe = (feat not in original_cols) and ("_" in feat)
                if is_ohe:
                    user_inputs[feat] = st.selectbox(feat, [0, 1], key=f"p_{feat}")
                else:
                    default_val = float(df_clean[feat].median()) if feat in df_clean.columns else 0.0
                    rng = ""
                    if feat in df_clean.columns:
                        rng = f"Range: {df_clean[feat].min():.1f} – {df_clean[feat].max():.1f}"

                    # No min_value / max_value — completely free input
                    user_inputs[feat] = st.number_input(
                        feat,
                        value=default_val,
                        step=1.0,
                        format="%.2f",
                        key=f"p_{feat}",
                        help=rng,
                    )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍 Run Prediction"):
        input_df     = pd.DataFrame([user_inputs])[feat_names]
        input_scaled = scaler.transform(input_df)
        model        = fitted[chosen]
        prediction   = model.predict(input_scaled)[0]

        has_proba = hasattr(model, "predict_proba")
        fraud_prob = not_fraud_prob = None
        if has_proba:
            proba          = model.predict_proba(input_scaled)[0]
            fraud_prob     = round(float(proba[1]) * 100, 2)
            not_fraud_prob = round(float(proba[0]) * 100, 2)

        st.markdown("<br>", unsafe_allow_html=True)
        _, r_col, _ = st.columns([1, 2, 1])
        with r_col:
            if prediction == 1:
                st.markdown(
                    f"<div class='pred-fraud'>"
                    f"<div class='pred-icon'>🚨</div>"
                    f"<div class='pred-label' style='color:#f87171;'>FRAUD DETECTED</div>"
                    f"<div class='pred-meta'>Model: <strong>{chosen}</strong> · Class 1</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='pred-ok'>"
                    f"<div class='pred-icon'>✅</div>"
                    f"<div class='pred-label' style='color:#34d399;'>NOT FRAUD</div>"
                    f"<div class='pred-meta'>Model: <strong>{chosen}</strong> · Class 0</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        if has_proba and fraud_prob is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            p1, p2 = st.columns(2)
            p1.metric("✅ Not Fraud Probability", f"{not_fraud_prob}%")
            p2.metric("🚨 Fraud Probability",     f"{fraud_prob}%")
            st.pyplot(chart_prob_bar(not_fraud_prob, fraud_prob))

        with st.expander("🔎 Input values used"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}).round(4),
                         use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.markdown("""
    <div class="hero">
      <div class="hero-eyebrow">Machine Learning · Classification</div>
      <div class="hero-title">Tax <span>Fraud</span> Detection System</div>
      <div class="hero-sub">Aiman Iftikhar &nbsp;·&nbsp; Memoona Sheikh &nbsp;·&nbsp; Final Project</div>
      <div>
        <span class="pill">SMOTE Balancing</span>
        <span class="pill">5 ML Models</span>
        <span class="pill">IQR Outlier Removal</span>
        <span class="pill">StandardScaler</span>
        <span class="pill">Auto Best Model</span>
        <span class="pill">Real-Time Prediction</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    tabs = st.tabs([
        "📁  Upload & Preview",
        "⚙️  Preprocessing",
        "⚖️  SMOTE",
        "🤖  Train Models",
        "🏆  Results",
        "🔍  Predict",
    ])

    with tabs[0]: tab_upload()
    with tabs[1]: tab_preprocess()
    with tabs[2]: tab_smote()
    with tabs[3]: tab_train()
    with tabs[4]: tab_results()
    with tabs[5]: tab_predict()

    st.markdown(
        "<div style='text-align:center;padding:3rem 0 1rem;color:#2a5a7c;"
        "font-size:0.78rem;font-family:JetBrains Mono,monospace;'>"
        "Tax Fraud Detection System &nbsp;·&nbsp; Aiman Iftikhar & Memoona Sheikh"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()