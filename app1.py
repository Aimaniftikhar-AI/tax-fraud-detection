import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# Machine Learning 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

st.set_page_config(
    page_title="Tax Fraud Detection System",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* ── background ── */
.stApp {
    background: #0d0f14;
    color: #e2e8f0;
}

/* ── header banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f1923 0%, #0d1f2d 50%, #0a1628 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(0,180,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.1rem;
    font-weight: 600;
    color: #38bdf8;
    letter-spacing: -0.5px;
    margin: 0 0 0.4rem 0;
}
.hero-sub {
    color: #64748b;
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.3px;
}
.badge {
    display: inline-block;
    background: rgba(56,189,248,0.12);
    border: 1px solid rgba(56,189,248,0.3);
    color: #38bdf8;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-family: 'IBM Plex Mono', monospace;
    margin-right: 8px;
    margin-top: 10px;
}

/* ── section headers ── */
.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.05rem;
    font-weight: 600;
    color: #38bdf8;
    border-left: 3px solid #38bdf8;
    padding-left: 12px;
    margin: 2rem 0 1rem 0;
    letter-spacing: 0.3px;
}

/* ── cards ── */
.info-card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.card-label {
    color: #64748b;
    font-size: 0.78rem;
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.card-value {
    color: #f1f5f9;
    font-size: 1.6rem;
    font-weight: 700;
    margin-top: 4px;
}

/* ── metric override ── */
[data-testid="stMetric"] {
    background: #111827 !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.8rem !important; }
[data-testid="stMetricValue"] { color: #38bdf8 !important; font-size: 1.5rem !important; font-weight: 700 !important; }

/* ── best model highlight ── */
.best-model-box {
    background: linear-gradient(135deg, #052e16 0%, #064e3b 100%);
    border: 1px solid #059669;
    border-radius: 14px;
    padding: 1.6rem 2rem;
    text-align: center;
    margin: 1rem 0;
}
.best-model-name {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: #34d399;
}
.best-model-label {
    color: #6ee7b7;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
}

/* ── warning / alert ── */
.overfitting-warn {
    background: #1c0f05;
    border: 1px solid #b45309;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    color: #fbbf24;
    font-size: 0.9rem;
}
.overfitting-ok {
    background: #052e16;
    border: 1px solid #059669;
    border-radius: 10px;
    padding: 1rem 1.3rem;
    color: #34d399;
    font-size: 0.9rem;
}

/* ── dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; }

/* ── buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0369a1, #0284c7) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #0284c7, #0ea5e9) !important;
    transform: translateY(-1px) !important;
}

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #0a0d13 !important;
    border-right: 1px solid #1e293b !important;
}
[data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }

/* ── step pills ── */
.step-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #1e293b;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.8rem;
    color: #94a3b8;
    margin: 3px 4px 3px 0;
    font-family: 'IBM Plex Mono', monospace;
}
.step-pill.done { background: rgba(5,150,105,0.2); color: #34d399; border: 1px solid #059669; }

/* ── divider ── */
.thin-divider { border: none; border-top: 1px solid #1e293b; margin: 1.5rem 0; }

/* ── upload zone ── */
[data-testid="stFileUploader"] {
    border: 1px dashed #1e3a5f !important;
    border-radius: 12px !important;
    background: #0a0d13 !important;
    padding: 0.5rem !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
  <div class="hero-title">TECH TAX FRAUD DETECTION SYSTEM</div>
  <div class="hero-sub">By: AIMAN IFTIKHAR & MEMOONA SHEIKH</div>
  <div>
    <span class="badge">SMOTE Balancing</span>
    <span class="badge">5 ML Models</span>
    <span class="badge">IQR Outlier Handling</span>
    <span class="badge">Auto Best Model</span>
  </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Pipeline Steps")
    steps = [
        ("Upload Dataset"),
        ("Dataset Preview"),
        ("Preprocessing"),
        ("SMOTE Balancing"),
        ("Train Models"),
        ("Model Comparison"),
        ("Overfitting Check"),
        ("Best Model"),
        ("Confusion Matrices"),
    ]
    for label in steps:
        st.markdown(f"<div class='step-pill'>{label}</div>", unsafe_allow_html=True)

    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
    st.markdown("**Models Trained**")
    for m in ["Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN"]:
        st.markdown(f"<div class='step-pill'>• {m}</div>", unsafe_allow_html=True)

    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
    st.caption("Final Project")

def load_data(uploaded_file):
    """Load CSV file into a pandas DataFrame."""
    df = pd.read_csv(uploaded_file)
    # Drop taxpayer_id if it exists
    if "taxpayer_id" in df.columns:
        df.drop(columns=["taxpayer_id"], inplace=True)
    return df


def preprocess_data(df):
    """
    Full preprocessing pipeline:
    1. Handle missing values
    2. One-Hot Encode categorical columns
    3. Outlier removal with IQR
    4. Feature scaling with StandardScaler
    Returns: X_scaled (array), y (series), scaler, feature_names, summary dict
    """
    summary = {}

    # 1. Missing values
    missing_before = df.isnull().sum().sum()
    df.fillna(df.median(numeric_only=True), inplace=True)
    # For categorical columns fill with mode
    for col in df.select_dtypes(include="object").columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    missing_after = df.isnull().sum().sum()
    summary["missing_before"] = int(missing_before)
    summary["missing_after"] = int(missing_after)

    # 2. One-Hot Encoding
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    if "fraud_flag" in cat_cols:
        cat_cols.remove("fraud_flag")
    summary["encoded_cols"] = cat_cols
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
    summary["shape_after_encoding"] = df.shape

    # 3. Outlier removal (IQR method) 
    target = df["fraud_flag"]
    df_features = df.drop(columns=["fraud_flag"])
    numeric_cols = df_features.select_dtypes(include=[np.number]).columns
    rows_before = len(df_features)
    mask = pd.Series([True] * len(df_features), index=df_features.index)
    outlier_counts = {}
    for col in numeric_cols:
        Q1 = df_features[col].quantile(0.25)
        Q3 = df_features[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        col_mask = (df_features[col] >= lower) & (df_features[col] <= upper)
        outliers_in_col = (~col_mask).sum()
        if outliers_in_col > 0:
            outlier_counts[col] = int(outliers_in_col)
        mask = mask & col_mask
    df_features = df_features[mask]
    target = target[mask]
    rows_after = len(df_features)
    summary["outliers_removed"] = rows_before - rows_after
    summary["outlier_counts_per_col"] = outlier_counts

    #  4. Feature Scaling 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features)
    feature_names = df_features.columns.tolist()
    summary["n_features"] = len(feature_names)
    summary["n_samples"] = rows_after

    return X_scaled, target.values, scaler, feature_names, summary, df_features


def apply_smote(X, y):
    """Apply SMOTE to balance class distribution."""
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res


def train_models(X_train, y_train):
    """Train all 5 models and return a dict of fitted models."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree":       DecisionTreeClassifier(random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM":                 SVC(kernel="rbf", probability=True, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=5),
    }
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def evaluate_models(fitted_models, X_train, X_test, y_train, y_test):
    """
    Evaluate each model and return:
      - results_df  : comparison DataFrame
      - preds_dict  : {name: y_pred} for confusion matrices
      - overfit_dict: {name: (train_acc, test_acc)}
    """
    records = []
    preds_dict = {}
    overfit_dict = {}

    for name, model in fitted_models.items():
        y_pred = model.predict(X_test)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc  = accuracy_score(y_test, y_pred)
        records.append({
            "Model":     name,
            "Accuracy":  round(test_acc, 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1 Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        })
        preds_dict[name]  = y_pred
        overfit_dict[name] = (round(train_acc, 4), round(test_acc, 4))

    results_df = pd.DataFrame(records).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    return results_df, preds_dict, overfit_dict


def plot_confusion_matrix(y_test, y_pred, model_name):
    """Plot a styled confusion matrix for a single model."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        linewidths=0.5, linecolor="#1e293b",
        xticklabels=["Not Fraud", "Fraud"],
        yticklabels=["Not Fraud", "Fraud"],
        ax=ax, cbar=False,
        annot_kws={"size": 13, "weight": "bold", "color": "white"},
    )
    ax.set_title(model_name, color="#38bdf8", fontsize=11, pad=10, fontweight="bold")
    ax.set_xlabel("Predicted", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Actual",    color="#94a3b8", fontsize=9)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")
    plt.tight_layout()
    return fig

# SECTION 1
st.markdown("<div class='section-header'> UPLOAD DATASET</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload your CSV file  ·  Expected: tax_fraud_detection_dataset.csv",
    type=["csv"],
    label_visibility="visible"
)

if uploaded_file is None:
    st.info("⬆ Please upload the dataset CSV file to begin the analysis pipeline.")
    st.stop()

df_raw = load_data(uploaded_file)

# SECTION 2
st.markdown("<div class='section-header'> DATASET PREVIEW</div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Rows",     f"{df_raw.shape[0]:,}")
c2.metric("Total Columns",  df_raw.shape[1])
c3.metric("Fraud Cases",    f"{int(df_raw['fraud_flag'].sum()):,}" if "fraud_flag" in df_raw.columns else "—")
c4.metric("Missing Values", f"{df_raw.isnull().sum().sum():,}")

st.markdown("**First 10 rows of the dataset:**")
st.dataframe(df_raw.head(10), use_container_width=True)

with st.expander(" Column Data Types & Nulls"):
    dtype_df = pd.DataFrame({
        "Column":      df_raw.columns,
        "Data Type":   df_raw.dtypes.values.astype(str),
        "Non-Null":    df_raw.notnull().sum().values,
        "Null Count":  df_raw.isnull().sum().values,
        "Unique Vals": df_raw.nunique().values,
    })
    st.dataframe(dtype_df, use_container_width=True)

with st.expander(" Statistical Summary"):
    st.dataframe(df_raw.describe().T.round(3), use_container_width=True)

st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

# SECTION 3 
st.markdown("<div class='section-header'>PREPROCESSING SUMMARY</div>", unsafe_allow_html=True)

with st.spinner("Running preprocessing pipeline..."):
    X_scaled, y, scaler, feature_names, summary, df_clean = preprocess_data(df_raw.copy())

# Display pipeline steps
step_col1, step_col2, step_col3, step_col4 = st.columns(4)
with step_col1:
    st.markdown("<div class='info-card'><div class='card-label'>Missing Values Before</div><div class='card-value'>{}</div></div>".format(summary["missing_before"]), unsafe_allow_html=True)
with step_col2:
    st.markdown("<div class='info-card'><div class='card-label'>Missing Values After</div><div class='card-value'>{}</div></div>".format(summary["missing_after"]), unsafe_allow_html=True)
with step_col3:
    st.markdown("<div class='info-card'><div class='card-label'>Outliers Removed</div><div class='card-value'>{}</div></div>".format(summary["outliers_removed"]), unsafe_allow_html=True)
with step_col4:
    st.markdown("<div class='info-card'><div class='card-label'>Final Features</div><div class='card-value'>{}</div></div>".format(summary["n_features"]), unsafe_allow_html=True)

# Preprocessing steps log
st.markdown("**Step-by-Step Preprocessing Log:**")
steps_log = {
    " Step 1 — Missing Value Imputation":
        f"Numeric columns → filled with **median**. Categorical columns → filled with **mode**. "
        f"Nulls before: **{summary['missing_before']}** → after: **{summary['missing_after']}**",
    " Step 2 — One-Hot Encoding":
        f"Categorical columns encoded: **{summary['encoded_cols'] if summary['encoded_cols'] else 'None (all numeric)'}**. "
        f"Shape after encoding: **{summary['shape_after_encoding']}**",
    " Step 3 — IQR Outlier Removal":
        f"Used IQR method (1.5×IQR fence). Total outlier rows removed: **{summary['outliers_removed']}**. "
        f"Remaining samples: **{summary['n_samples']}**",
    " Step 4 — Feature Scaling (StandardScaler)":
        f"All numeric features standardized to mean=0, std=1. "
        f"Total features scaled: **{summary['n_features']}**",
}
for title, detail in steps_log.items():
    with st.expander(title):
        st.markdown(detail)

# Per-column outlier breakdown
if summary["outlier_counts_per_col"]:
    with st.expander("🔎 Outliers Detected Per Column"):
        oc = pd.DataFrame(summary["outlier_counts_per_col"].items(), columns=["Column", "Outlier Rows"])
        st.dataframe(oc.sort_values("Outlier Rows", ascending=False), use_container_width=True)

        # Outlier bar chart
        fig_out, ax_out = plt.subplots(figsize=(8, 3))
        fig_out.patch.set_facecolor("#111827")
        ax_out.set_facecolor("#111827")
        cols_o = list(summary["outlier_counts_per_col"].keys())
        vals_o = list(summary["outlier_counts_per_col"].values())
        bars = ax_out.barh(cols_o, vals_o, color="#0ea5e9", edgecolor="#1e293b")
        ax_out.set_title("Outlier Count by Feature", color="#38bdf8", fontsize=11)
        ax_out.tick_params(colors="#94a3b8")
        ax_out.set_xlabel("Count", color="#94a3b8")
        for spine in ax_out.spines.values():
            spine.set_edgecolor("#1e293b")
        plt.tight_layout()
        st.pyplot(fig_out)

# Feature correlation heatmap
with st.expander("🗺 Feature Correlation Heatmap (top 10 features)"):
    top_feats = df_clean.iloc[:, :min(10, df_clean.shape[1])]
    fig_corr, ax_corr = plt.subplots(figsize=(9, 6))
    fig_corr.patch.set_facecolor("#111827")
    ax_corr.set_facecolor("#111827")
    sns.heatmap(
        top_feats.corr(), annot=True, fmt=".2f",
        cmap="coolwarm", linewidths=0.3, linecolor="#1e293b",
        ax=ax_corr, annot_kws={"size": 7}, cbar_kws={"shrink": 0.7}
    )
    ax_corr.set_title("Correlation Matrix", color="#38bdf8", fontsize=11)
    ax_corr.tick_params(colors="#94a3b8", labelsize=7)
    plt.tight_layout()
    st.pyplot(fig_corr)

st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

# SECTION 4 
st.markdown("<div class='section-header'> CLASS BALANCING (SMOTE)</div>", unsafe_allow_html=True)

unique, counts_before = np.unique(y, return_counts=True)
before_dict = dict(zip(unique.astype(str), counts_before))

with st.spinner("Applying SMOTE..."):
    X_balanced, y_balanced = apply_smote(X_scaled, y)

unique2, counts_after = np.unique(y_balanced, return_counts=True)
after_dict = dict(zip(unique2.astype(str), counts_after))

smote_c1, smote_c2 = st.columns(2)

with smote_c1:
    st.markdown("**Before SMOTE**")
    for cls, cnt in before_dict.items():
        label = " Fraud" if cls == "1" else " Not Fraud"
        st.metric(label, f"{cnt:,}")

    fig_b, ax_b = plt.subplots(figsize=(4, 3))
    fig_b.patch.set_facecolor("#111827")
    ax_b.set_facecolor("#111827")
    ax_b.bar(
        ["Not Fraud (0)", "Fraud (1)"],
        [before_dict.get("0", 0), before_dict.get("1", 0)],
        color=["#0ea5e9", "#ef4444"], edgecolor="#1e293b", width=0.5
    )
    ax_b.set_title("Before SMOTE", color="#38bdf8", fontsize=10)
    ax_b.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax_b.spines.values():
        spine.set_edgecolor("#1e293b")
    plt.tight_layout()
    st.pyplot(fig_b)

with smote_c2:
    st.markdown("**After SMOTE**")
    for cls, cnt in after_dict.items():
        label = " Fraud" if cls == "1" else " Not Fraud"
        st.metric(label, f"{cnt:,}")

    fig_a, ax_a = plt.subplots(figsize=(4, 3))
    fig_a.patch.set_facecolor("#111827")
    ax_a.set_facecolor("#111827")
    ax_a.bar(
        ["Not Fraud (0)", "Fraud (1)"],
        [after_dict.get("0", 0), after_dict.get("1", 0)],
        color=["#0ea5e9", "#10b981"], edgecolor="#1e293b", width=0.5
    )
    ax_a.set_title("After SMOTE", color="#38bdf8", fontsize=10)
    ax_a.tick_params(colors="#94a3b8", labelsize=8)
    for spine in ax_a.spines.values():
        spine.set_edgecolor("#1e293b")
    plt.tight_layout()
    st.pyplot(fig_a)

st.success(f"✅ SMOTE applied successfully. Dataset expanded from **{len(y):,}** → **{len(y_balanced):,}** samples.")

st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

# SECTION 5 
st.markdown("<div class='section-header'> TRAIN MODELS</div>", unsafe_allow_html=True)

split_pct = st.slider("Train/Test Split (train %)", min_value=60, max_value=90, value=80, step=5)
st.caption(f"Training on **{split_pct}%** of data · Testing on **{100 - split_pct}%**")

train_btn = st.button("  Train All 5 Models", use_container_width=False)

if "trained" not in st.session_state:
    st.session_state.trained = False

if train_btn:
    st.session_state.trained = True
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced,
        test_size=(100 - split_pct) / 100,
        random_state=42, stratify=y_balanced
    )
    with st.spinner("Training models — please wait..."):
        fitted_models = train_models(X_train, y_train)
        results_df, preds_dict, overfit_dict = evaluate_models(fitted_models, X_train, X_test, y_train, y_test)

    # Store in session state
    st.session_state.X_train      = X_train
    st.session_state.X_test       = X_test
    st.session_state.y_train      = y_train
    st.session_state.y_test       = y_test
    st.session_state.fitted       = fitted_models
    st.session_state.results_df   = results_df
    st.session_state.preds_dict   = preds_dict
    st.session_state.overfit_dict = overfit_dict
    st.success(" All models trained successfully!")

if not st.session_state.trained:
    st.info("Click **Train All 5 Models** to start the training pipeline.")
    st.stop()

# Restore from session state
results_df   = st.session_state.results_df
preds_dict   = st.session_state.preds_dict
overfit_dict = st.session_state.overfit_dict
y_test       = st.session_state.y_test
y_train      = st.session_state.y_train

st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

# SECTION 6 
st.markdown("<div class='section-header'> MODEL COMPARISON TABLE</div>", unsafe_allow_html=True)

# Add rank column
results_display = results_df.copy()
results_display.insert(0, "Rank", [f"#{i+1}" for i in range(len(results_display))])

st.dataframe(
    results_display.style
        .background_gradient(subset=["Accuracy", "Precision", "Recall", "F1 Score"],
                             cmap="Blues")
        .format({"Accuracy": "{:.4f}", "Precision": "{:.4f}", "Recall": "{:.4f}", "F1 Score": "{:.4f}"}),
    use_container_width=True, height=230
)

# Bar chart comparison
fig_cmp, ax_cmp = plt.subplots(figsize=(10, 4))
fig_cmp.patch.set_facecolor("#111827")
ax_cmp.set_facecolor("#111827")
x = np.arange(len(results_df))
w = 0.2
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score"]
colors_plot     = ["#38bdf8", "#818cf8", "#f472b6", "#34d399"]
for i, (metric, color) in enumerate(zip(metrics_to_plot, colors_plot)):
    ax_cmp.bar(x + i * w, results_df[metric], width=w, label=metric, color=color, edgecolor="#1e293b")
ax_cmp.set_xticks(x + 1.5 * w)
ax_cmp.set_xticklabels(results_df["Model"], rotation=15, ha="right", color="#94a3b8", fontsize=8)
ax_cmp.set_ylim(0, 1.12)
ax_cmp.set_title("Model Performance Comparison", color="#38bdf8", fontsize=12)
ax_cmp.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8", fontsize=8)
ax_cmp.tick_params(colors="#94a3b8")
for spine in ax_cmp.spines.values():
    spine.set_edgecolor("#1e293b")
plt.tight_layout()
st.pyplot(fig_cmp)

st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

# SECTION 7 
st.markdown("<div class='section-header'> OVERFITTING ANALYSIS</div>", unsafe_allow_html=True)

overfit_rows = []
for name, (tr_acc, te_acc) in overfit_dict.items():
    diff = round(tr_acc - te_acc, 4)
    status = " Overfitting" if diff > 0.05 else " Good Fit"
    overfit_rows.append({"Model": name, "Train Acc": tr_acc, "Test Acc": te_acc,
                         "Δ Gap": diff, "Status": status})
overfit_df = pd.DataFrame(overfit_rows)
st.dataframe(overfit_df, use_container_width=True, hide_index=True)

for _, row in overfit_df.iterrows():
    if row["Δ Gap"] > 0.05:
        st.markdown(
            f"<div class='overfitting-warn'> <b>{row['Model']}</b> — Gap of <b>{row['Δ Gap']:.4f}</b> "
            f"suggests overfitting. Train Acc: {row['Train Acc']} | Test Acc: {row['Test Acc']}</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"<div class='overfitting-ok'> <b>{row['Model']}</b> — Generalising well. "
            f"Train Acc: {row['Train Acc']} | Test Acc: {row['Test Acc']}</div>",
            unsafe_allow_html=True
        )

# Train vs Test accuracy grouped bar
fig_ov, ax_ov = plt.subplots(figsize=(9, 3.5))
fig_ov.patch.set_facecolor("#111827")
ax_ov.set_facecolor("#111827")
names_ov = overfit_df["Model"].tolist()
xo = np.arange(len(names_ov))
ax_ov.bar(xo - 0.2, overfit_df["Train Acc"], 0.35, label="Train Accuracy", color="#38bdf8", edgecolor="#1e293b")
ax_ov.bar(xo + 0.2, overfit_df["Test Acc"],  0.35, label="Test Accuracy",  color="#10b981", edgecolor="#1e293b")
ax_ov.set_xticks(xo)
ax_ov.set_xticklabels(names_ov, rotation=15, ha="right", color="#94a3b8", fontsize=8)
ax_ov.set_ylim(0, 1.12)
ax_ov.set_title("Train vs Test Accuracy", color="#38bdf8", fontsize=11)
ax_ov.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#94a3b8", fontsize=8)
ax_ov.tick_params(colors="#94a3b8")
for spine in ax_ov.spines.values():
    spine.set_edgecolor("#1e293b")
plt.tight_layout()
st.pyplot(fig_ov)

st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

# SECTION 8
st.markdown("<div class='section-header'> BEST MODEL</div>", unsafe_allow_html=True)

best_row = results_df.iloc[0]
bm_c1, bm_c2, bm_c3 = st.columns([1, 2, 1])

with bm_c2:
    st.markdown(f"""
    <div class="best-model-box">
      <div class="best-model-label"> BEST PERFORMING MODEL</div>
      <div class="best-model-name">{best_row['Model']}</div>
      <div style="margin-top:1rem; display:flex; justify-content:center; gap:2rem;">
        <div><div class="best-model-label">ACCURACY</div><div style="color:#f1f5f9;font-size:1.4rem;font-weight:700;">{best_row['Accuracy']:.4f}</div></div>
        <div><div class="best-model-label">F1 SCORE</div><div style="color:#f1f5f9;font-size:1.4rem;font-weight:700;">{best_row['F1 Score']:.4f}</div></div>
        <div><div class="best-model-label">PRECISION</div><div style="color:#f1f5f9;font-size:1.4rem;font-weight:700;">{best_row['Precision']:.4f}</div></div>
        <div><div class="best-model-label">RECALL</div><div style="color:#f1f5f9;font-size:1.4rem;font-weight:700;">{best_row['Recall']:.4f}</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# Classification report for best model
with st.expander(f" Full Classification Report — {best_row['Model']}"):
    best_preds = preds_dict[best_row["Model"]]
    report_str = classification_report(y_test, best_preds, target_names=["Not Fraud", "Fraud"])
    st.code(report_str, language="text")

st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────
# SECTION: PREDICT TAX FRAUD (NEW)
# ─────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>PREDICT TAX FRAUD</div>", unsafe_allow_html=True)
st.markdown("Enter feature values below and get a fraud prediction from any trained model.")

# we need the trained models and the scaler from session state / preprocessing
# scaler and feature_names are already available from the preprocessing step above
# fitted models are in st.session_state.fitted

if not st.session_state.trained:
    st.info("Please train the models first (Section 5) before using the prediction module.")
else:
    fitted_models = st.session_state.fitted

    # ── Model selector ───────────────────────────────────
    pred_col1, pred_col2 = st.columns([2, 1])
    with pred_col1:
        selected_model_name = st.selectbox(
            "Select Model for Prediction",
            options=list(fitted_models.keys()),
            index=0,  # default to first (best by F1)
            help="Choose which trained model to use for inference."
        )
    with pred_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        best_model_name = st.session_state.results_df.iloc[0]["Model"]
        st.caption(f" Best model: **{best_model_name}**")

    selected_model = fitted_models[selected_model_name]

    st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)
    st.markdown("**Enter Feature Values:**")

    # ── Build input fields for every feature ─────────────
    # feature_names comes from the preprocessing step (already in scope above)
    # We generate one input per feature that was used after encoding + scaling.
    # Strategy: use number_input for all features (they are all numeric after OHE).
    # We group them in rows of 4 columns for a clean layout.

    user_inputs = {}
    cols_per_row = 4
    feature_chunks = [feature_names[i:i + cols_per_row]
                      for i in range(0, len(feature_names), cols_per_row)]

    for chunk in feature_chunks:
        row_cols = st.columns(len(chunk))
        for col, feat in zip(row_cols, chunk):
            with col:
                # For one-hot encoded binary columns, use a selectbox (0 or 1)
                # Heuristic: if the feature name contains "_" and the raw data
                # doesn't have it as an original column it is likely OHE → binary
                original_cols = df_raw.columns.tolist()
                is_ohe_col = (feat not in original_cols) and ("_" in feat)

                if is_ohe_col:
                    user_inputs[feat] = st.selectbox(
                        feat,
                        options=[0, 1],
                        key=f"pred_{feat}"
                    )
                else:
                    # Use the cleaned dataframe to get a sensible default (median)
                    if feat in df_clean.columns:
                        default_val = float(df_clean[feat].median())
                        col_min = float(df_clean[feat].min())
                        col_max = float(df_clean[feat].max())
                    else:
                        default_val, col_min, col_max = 0.0, 0.0, 1e9

                    user_inputs[feat] = st.number_input(
                        feat,
                        min_value=col_min,
                        max_value=col_max,
                        value=default_val,
                        step=(col_max - col_min) / 100 if col_max != col_min else 1.0,
                        key=f"pred_{feat}",
                        format="%.4f"
                    )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("Run Prediction", use_container_width=False)

    if predict_btn:
        # ── Build input DataFrame in the same feature order ──
        input_df = pd.DataFrame([user_inputs])          # shape (1, n_features)
        input_df = input_df[feature_names]              # enforce correct column order

        # ── Apply the SAME scaler fitted during training ──
        # scaler is already in scope from the preprocess_data() call above
        try:
            input_scaled = scaler.transform(input_df)
        except Exception as e:
            st.error(f"Scaling error: {e}")
            st.stop()

        # ── Predict ──────────────────────────────────────
        prediction = selected_model.predict(input_scaled)[0]

        # Probability (if model supports it)
        prob_available = hasattr(selected_model, "predict_proba")
        if prob_available:
            proba = selected_model.predict_proba(input_scaled)[0]
            fraud_prob     = round(float(proba[1]) * 100, 2)
            not_fraud_prob = round(float(proba[0]) * 100, 2)

        # ── Display result ────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        result_col1, result_col2, result_col3 = st.columns([1, 2, 1])
        with result_col2:
            if prediction == 1:
                st.markdown(f"""
                <div class='overfitting-warn' style='text-align:center; padding: 1.5rem 2rem;'>
                    <div style='font-size:2rem;'></div>
                    <div style='font-size:1.3rem; font-weight:700; margin-top:0.3rem;'>FRAUD DETECTED</div>
                    <div style='font-size:0.85rem; margin-top:0.4rem; color:#fcd34d;'>
                        Model: <b>{selected_model_name}</b> &nbsp;·&nbsp; Prediction: <b>Class 1 (Fraud)</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='overfitting-ok' style='text-align:center; padding: 1.5rem 2rem;'>
                    <div style='font-size:2rem;'></div>
                    <div style='font-size:1.3rem; font-weight:700; margin-top:0.3rem;'>NOT FRAUD</div>
                    <div style='font-size:0.85rem; margin-top:0.4rem; color:#6ee7b7;'>
                        Model: <b>{selected_model_name}</b> &nbsp;·&nbsp; Prediction: <b>Class 0 (Not Fraud)</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── Probability bar ───────────────────────────────
        if prob_available:
            st.markdown("<br>", unsafe_allow_html=True)
            prob_c1, prob_c2 = st.columns(2)
            prob_c1.metric(" Not Fraud Probability", f"{not_fraud_prob}%")
            prob_c2.metric(" Fraud Probability",     f"{fraud_prob}%")

            # simple horizontal bar chart
            fig_prob, ax_prob = plt.subplots(figsize=(6, 1.2))
            fig_prob.patch.set_facecolor("#111827")
            ax_prob.set_facecolor("#111827")
            ax_prob.barh(["Not Fraud"], [not_fraud_prob], color="#10b981",
                         edgecolor="#1e293b", height=0.4)
            ax_prob.barh(["Fraud"],     [fraud_prob],     color="#ef4444",
                         edgecolor="#1e293b", height=0.4)
            ax_prob.set_xlim(0, 100)
            ax_prob.set_xlabel("Probability (%)", color="#94a3b8", fontsize=8)
            ax_prob.tick_params(colors="#94a3b8", labelsize=8)
            for sp in ax_prob.spines.values():
                sp.set_edgecolor("#1e293b")
            plt.tight_layout()
            st.pyplot(fig_prob)

        # ── Show the input values used ────────────────────
        with st.expander(" Input Values Used for Prediction"):
            st.dataframe(input_df.T.rename(columns={0: "Value"}).round(4),
                         use_container_width=True)

st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

# SECTION 9 
st.markdown("<div class='section-header'>CONFUSION MATRICES</div>", unsafe_allow_html=True)
st.caption("All 5 models shown below — rows = actual, columns = predicted.")

model_names = list(preds_dict.keys())

# Row 1: first 3 models
row1 = st.columns(3)
for i in range(min(3, len(model_names))):
    with row1[i]:
        fig_cm = plot_confusion_matrix(y_test, preds_dict[model_names[i]], model_names[i])
        st.pyplot(fig_cm)

# Row 2: remaining models
remaining = model_names[3:]
if remaining:
    row2_cols = st.columns(len(remaining))
    for i, name in enumerate(remaining):
        with row2_cols[i]:
            fig_cm = plot_confusion_matrix(y_test, preds_dict[name], name)
            st.pyplot(fig_cm)

st.markdown("<hr class='thin-divider'>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; padding: 2rem 0; color: #334155; font-size: 0.82rem; font-family: 'IBM Plex Mono', monospace;">
  Tech Tax Fraud Detection System &nbsp;·&nbsp; Aiman Iftikhar & Memoona Sheikh 
</div>
""", unsafe_allow_html=True)