# app.py
import io
import sys
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=UserWarning)

# ---- Try to import XGBoost gracefully ----
HAS_XGB = True
try:
    from xgboost import XGBClassifier
except Exception:
    HAS_XGB = False

# ----------------------------
# Utility functions
# ----------------------------

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase, strip, replace spaces and hyphens with underscores."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(r"[ \-]+", "_", regex=True)
    )
    return df

def find_target_column(df: pd.DataFrame) -> str:
    """
    Try to find the target column. We accept 'fetal_health' or any close variant
    like 'fetal health', 'Fetal Health', case-insensitive.
    """
    candidates = {c: c.replace(" ", "_").lower() for c in df.columns}
    for original, normalized in candidates.items():
        if normalized == "fetal_health":
            return original
    # If not found, try fuzzy-ish search
    for original, normalized in candidates.items():
        if "fetal" in normalized and "health" in normalized:
            return original
    return ""

def make_synthetic_fetal_health_dataframe(n=200, seed=42):
    """
    Create a synthetic fetal health-like dataset with several numeric features
    and a multiclass target 'fetal_health' in {1,2,3}.
    This is only for UI testing and demo purposes.
    """
    rng = np.random.default_rng(seed)
    # 20 numeric features inspired by CTG-like structure (not exact)
    cols = [
        "baseline_value",
        "accelerations",
        "fetal_movement",
        "uterine_contractions",
        "light_decelerations",
        "severe_decelerations",
        "prolongued_decelerations",
        "abnormal_short_term_variability",
        "mean_value_of_short_term_variability",
        "percentage_of_time_with_abnormal_long_term_variability",
        "mean_value_of_long_term_variability",
        "histogram_width",
        "histogram_min",
        "histogram_max",
        "histogram_number_of_peaks",
        "histogram_number_of_zeroes",
        "histogram_mode",
        "histogram_mean",
        "histogram_median",
        "histogram_variance",
    ]

    X = pd.DataFrame({
        "baseline_value": rng.normal(120, 10, n),
        "accelerations": rng.gamma(2.0, 0.5, n),
        "fetal_movement": rng.poisson(3, n),
        "uterine_contractions": rng.gamma(2.0, 0.8, n),
        "light_decelerations": rng.poisson(2, n),
        "severe_decelerations": rng.binomial(1, 0.1, n),
        "prolongued_decelerations": rng.binomial(1, 0.05, n),
        "abnormal_short_term_variability": rng.integers(0, 100, n),
        "mean_value_of_short_term_variability": rng.uniform(0, 10, n),
        "percentage_of_time_with_abnormal_long_term_variability": rng.uniform(0, 100, n),
        "mean_value_of_long_term_variability": rng.uniform(0, 50, n),
        "histogram_width": rng.uniform(10, 60, n),
        "histogram_min": rng.integers(50, 120, n),
        "histogram_max": rng.integers(120, 200, n),
        "histogram_number_of_peaks": rng.integers(0, 10, n),
        "histogram_number_of_zeroes": rng.integers(0, 20, n),
        "histogram_mode": rng.integers(50, 200, n),
        "histogram_mean": rng.uniform(80, 160, n),
        "histogram_median": rng.uniform(80, 160, n),
        "histogram_variance": rng.uniform(0, 50, n),
    })

    # Create a 3-class target influenced by some features (purely synthetic)
    risk_score = (
        0.02 * (X["abnormal_short_term_variability"])
        + 0.03 * (X["percentage_of_time_with_abnormal_long_term_variability"])
        + 0.01 * (X["severe_decelerations"] * 100)
        + 0.01 * (X["prolongued_decelerations"] * 100)
        - 0.02 * X["accelerations"] * 10
    )
    # Quantiles define class boundaries 1 (normal), 2 (suspect), 3 (pathological)
    q1, q2 = np.quantile(risk_score, [0.65, 0.85])
    y = np.where(risk_score < q1, 1, np.where(risk_score < q2, 2, 3))

    df = X.copy()
    df["fetal_health"] = y
    return df

def load_dataframe_from_upload(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file, engine="openpyxl")
    elif name.endswith(".xls"):
        # Requires xlrd installed (supports .xls)
        return pd.read_excel(uploaded_file, engine="xlrd")
    elif name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload .xls, .xlsx, or .csv")

def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

def compute_metrics(y_true, y_pred, y_proba, average="weighted"):
    metrics = {}
    metrics["Accuracy"] = accuracy_score(y_true, y_pred)
    metrics["Precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics["Recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics["F1 Score"] = f1_score(y_true, y_pred, average=average, zero_division=0)

    # MCC supports multiclass directly
    metrics["MCC Score"] = matthews_corrcoef(y_true, y_pred)

    # AUC (multiclass): requires probability estimates
    if y_proba is not None:
        # Handle multiclass AUC
        try:
            metrics["AUC Score"] = roc_auc_score(
                y_true, y_proba, multi_class="ovr", average=average
            )
        except Exception:
            metrics["AUC Score"] = np.nan
    else:
        metrics["AUC Score"] = np.nan

    return metrics

# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Fetal Health – Model Tester", layout="wide")
st.title("🍼 Fetal Health – Model Tester")

st.markdown(
    """
Use this app to **download sample data**, **upload your dataset**, **select a model**, and **see evaluation metrics**.
**Target column must be named** `fetal_health` (case/space-insensitive).  
**Note:** The sample dataset is *synthetic* (for demo/testing the UI only).
"""
)

with st.sidebar:
    st.header("1) Get Sample Data")
    st.caption("Synthetic demo data with `fetal_health` as the target.")
    df_sample = make_synthetic_fetal_health_dataframe(n=200, seed=42)

    # Offer .xlsx for download
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df_sample.to_excel(writer, index=False, sheet_name="fetal_health_sample")
    st.download_button(
        label="⬇️ Download sample (.xlsx)",
        data=buffer.getvalue(),
        file_name="fetal_health_sample.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.header("2) Settings")
    test_size = st.slider("Test size (validation split)", 0.1, 0.5, 0.2, 0.05)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    model_names = [
        "Logistic Regression",
        "Decision Tree Classifier",
        "K-Nearest Neighbor Classifier",
        "Naive Bayes Classifier",
        "Ensemble: Random Forest",
    ]
    if HAS_XGB:
        model_names.append("Ensemble: XGBoost")
    else:
        st.info("ℹ️ XGBoost not installed. To enable: `pip install xgboost`")

    selected_model = st.selectbox("3) Select a model", model_names, index=0)

    nb_variant = None
    if selected_model == "Naive Bayes Classifier":
        nb_variant = st.radio(
            "Naive Bayes variant",
            ["Gaussian", "Multinomial"],
            index=0,
            horizontal=True,
            help="GaussianNB for continuous features; MultinomialNB for count-like, non-negative features",
        )

    show_which = st.radio(
        "Show analysis",
        ["Confusion Matrix", "Classification Report"],
        index=0,
        horizontal=True,
    )

st.header("3) Upload Your Data")
uploaded = st.file_uploader(
    "Upload .xls / .xlsx (or .csv) file containing features and a `fetal_health` target column.",
    type=["xls", "xlsx", "csv"],
)

if uploaded is not None:
    try:
        df = load_dataframe_from_upload(uploaded)
    except Exception as ex:
        st.error(f"Could not read file: {ex}")
        st.stop()
else:
    st.warning("Upload a dataset to proceed, or download the sample from the sidebar.")
    st.stop()

# Normalize columns and locate target
df = normalize_columns(df)
target_col = find_target_column(df)
if not target_col:
    st.error("Target column `fetal_health` not found. Please include it in your file.")
    st.stop()

# Basic sanity
if df[target_col].isna().any():
    st.warning("Missing values detected in the target column; dropping such rows.")
    df = df[~df[target_col].isna()]
if df.shape[0] < 20:
    st.warning("Very few rows after cleaning; metrics may be unstable.")

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Attempt to coerce numeric types in X
for c in X.columns:
    if not pd.api.types.is_numeric_dtype(X[c]):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X[c] = pd.to_numeric(X[c], errors="coerce")

# Drop rows with all-NaN features
all_nan = X.isna().all(axis=1)
if all_nan.any():
    st.warning(f"Dropping {all_nan.sum()} rows with all-NaN features.")
    X = X.loc[~all_nan]
    y = y.loc[~all_nan]

# Train/validation split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed, stratify=y if y.nunique() > 1 else None
)

# Build model pipeline
numeric_features = X.columns.tolist()

def build_pipeline(name: str):
    if name == "Logistic Regression":
        clf = LogisticRegression(max_iter=2000, n_jobs=None)
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
        return Pipeline(steps)

    if name == "Decision Tree Classifier":
        clf = DecisionTreeClassifier(random_state=seed)
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ]
        return Pipeline(steps)

    if name == "K-Nearest Neighbor Classifier":
        clf = KNeighborsClassifier(n_neighbors=5)
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
        return Pipeline(steps)

    if name == "Naive Bayes Classifier":
        if nb_variant == "Multinomial":
            # MultinomialNB requires non-negative features; scale to [0,1]
            clf = MultinomialNB()
            steps = [
                ("imputer", SimpleImputer(strategy="median")),
                ("nonneg", MinMaxScaler()),  # ensures non-negative inputs
                ("clf", clf),
            ]
        else:
            clf = GaussianNB()
            steps = [
                ("imputer", SimpleImputer(strategy="mean")),
                ("clf", clf),
            ]
        return Pipeline(steps)

    if name == "Ensemble: Random Forest":
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ]
        return Pipeline(steps)

    if name == "Ensemble: XGBoost" and HAS_XGB:
        clf = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            eval_metric="mlogloss",
            tree_method="hist",
            nthread=-1,
        )
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", clf),
        ]
        return Pipeline(steps)

    raise ValueError("Unknown model selected or dependency missing.")

# Train and evaluate
try:
    model = build_pipeline(selected_model)
except Exception as ex:
    st.error(f"Could not build model: {ex}")
    st.stop()

with st.spinner("Training and evaluating..."):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Probability estimates (if available)
    y_proba = None
    if hasattr(model.named_steps["clf"], "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

# Metrics
metrics = compute_metrics(y_test, y_pred, y_proba, average="weighted")
metrics_df = pd.DataFrame(
    {
        "Metric": list(metrics.keys()),
        "Score": [None if np.isnan(v) else float(v) for v in metrics.values()],
    }
)

left, right = st.columns([1, 1])
with left:
    st.subheader("Evaluation Metrics")
    st.dataframe(
        metrics_df.style.format({"Score": "{:.4f}"}), use_container_width=True
    )

with right:
    st.subheader("Class Distribution (Test Set)")
    st.bar_chart(y_test.value_counts().sort_index(), use_container_width=True)

# Confusion matrix or classification report
st.subheader("Model Diagnostics")
labels_sorted = sorted(y.unique())

if show_which == "Confusion Matrix":
    plot_confusion(y_test, y_pred, labels_sorted)
else:
    report = classification_report(y_test, y_pred, digits=4)
    st.text(report)

# Helpful tips
st.markdown(
    """
**Tips**
- Ensure your file has a `fetal_health` target column (values typically 1, 2, 3).
- All other columns should be numeric features. Non-numeric columns will be coerced; rows with all-NaN features are dropped.
- For **MultinomialNB**, features are scaled to [0,1] to satisfy non-negativity.
- AUC is computed using multiclass One-vs-Rest with weighted averaging when probabilities are available.
"""
)