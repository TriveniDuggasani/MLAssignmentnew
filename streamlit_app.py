# app.py
import io
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import load_iris

# Models (6 total)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


# -----------------------------
# App Title & Description
# -----------------------------
st.set_page_config(page_title="Simple ML Classifier", layout="wide")
st.title("🔎 Simple Streamlit ML Classifier")
st.caption(
    "Upload a small CSV (test data), choose a model, and view evaluation metrics, "
    "a confusion matrix, and a classification report."
)

with st.sidebar:
    st.header("🔧 Controls")

    # Upload CSV
    uploaded = st.file_uploader(
        "Upload CSV (small/test data recommended)",
        type=["csv"],
        help="For Streamlit free tier, keep it small (e.g., < ~5 MB).",
    )

    # Optional demo dataset
    use_demo = st.checkbox("Use demo dataset (Iris)", value=not bool(uploaded))

    # Sampling to limit rows
    sample_n = st.number_input(
        "Sample rows (to keep it light)", min_value=50, max_value=10000, value=500, step=50,
        help="If your dataset is larger, a random sample of this size will be used."
    )

    test_size = st.slider(
        "Test size (%)", min_value=10, max_value=40, value=20, step=5
    ) / 100.0

    scale_numeric = st.checkbox("Scale numeric features", value=True)

    random_state = st.number_input("Random state", min_value=0, value=42, step=1)

    model_name = st.selectbox(
        "Select model",
        [
            "Logistic Regression",
            "SVM (RBF)",
            "Random Forest",
            "K-Nearest Neighbors",
            "Gaussian Naive Bayes",
            "Decision Tree",
        ],
    )

    run_btn = st.button("▶️ Train & Evaluate", use_container_width=True)


# -----------------------------
# Helpers
# -----------------------------
def load_demo_df() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    # Rename target for clarity
    df.rename(columns={"target": "target"}, inplace=True)
    return df


def read_csv_safely(file) -> pd.DataFrame:
    # Light validation: size check (approx)
    if hasattr(file, "size") and file.size > 5 * 1024 * 1024:
        st.warning("⚠️ File is larger than 5 MB. Please upload a smaller test file.")
        return None
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return None


def build_model(name: str, random_state: int):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    elif name == "SVM (RBF)":
        return SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state)
    elif name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=200, random_state=random_state, class_weight="balanced"
        )
    elif name == "K-Nearest Neighbors":
        return KNeighborsClassifier(n_neighbors=5)
    elif name == "Gaussian Naive Bayes":
        return GaussianNB()
    elif name == "Decision Tree":
        return DecisionTreeClassifier(random_state=random_state, class_weight="balanced")
    else:
        raise ValueError("Unknown model selected.")


def prepare_preprocessor(df: pd.DataFrame, target_col: str, scale_numeric: bool) -> ColumnTransformer:
    X = df.drop(columns=[target_col])
    # Identify types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_steps = []
    # impute numeric
    numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    from sklearn.pipeline import Pipeline as SKPipeline
    numeric_transformer = SKPipeline(steps=numeric_steps)

    # impute + one-hot for categorical
    # We use sparse=False for broad compatibility with GaussianNB and Streamlit plotting
    categorical_transformer = SKPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def can_compute_proba(clf) -> bool:
    # Some models support predict_proba()
    return hasattr(clf, "predict_proba")


def compute_metrics(y_true, y_pred, y_proba=None):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    metrics["precision_weighted"] = p
    metrics["recall_weighted"] = r
    metrics["f1_weighted"] = f1

    # Optional ROC-AUC (binary or multiclass)
    # For multiclass, sklearn supports average='weighted' with label_binarize-like internal handling if predict_proba provided.
    if y_proba is not None:
        try:
            # For binary, y_proba is shape (n_samples, 2) -> use probas for class 1
            if y_proba.shape[1] == 2:
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
            metrics["roc_auc"] = auc
        except Exception:
            metrics["roc_auc"] = np.nan

    return metrics


# -----------------------------
# Load Data
# -----------------------------
df = None
data_source = "demo" if use_demo or not uploaded else "uploaded"

if data_source == "uploaded" and uploaded:
    df = read_csv_safely(uploaded)
elif use_demo:
    df = load_demo_df()

if df is None:
    st.info("Upload a CSV or tick **Use demo dataset (Iris)** in the sidebar to begin.")
    st.stop()

# Sample to keep it light (if needed)
if len(df) > sample_n:
    df = df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)

st.success(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
with st.expander("👀 Preview data"):
    st.dataframe(df.head(15), use_container_width=True)

# -----------------------------
# Target Column Selection
# -----------------------------
# Heuristic: if a column named 'target' exists, preselect it.
candidate_target = "target" if "target" in df.columns else df.columns[-1]
target_col = st.selectbox(
    "Select target column (classification label)",
    options=df.columns.tolist(),
    index=df.columns.tolist().index(candidate_target) if candidate_target in df.columns else 0,
)

# Remove missing target rows
df = df.dropna(subset=[target_col])
y = df[target_col]
X = df.drop(columns=[target_col])

# Guard: target should be categorical/discrete-ish
if y.nunique() > max(50, int(0.2 * len(y))):
    st.warning(
        f"⚠️ The selected target '{target_col}' has {y.nunique()} unique values. "
        "This might be regression-like or too granular for a simple classifier."
    )

# -----------------------------
# Train & Evaluate
# -----------------------------
if run_btn:
    with st.spinner("Training and evaluating..."):
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique() > 1 else None
        )

        # Build preprocessor and model
        preprocessor = prepare_preprocessor(df, target_col, scale_numeric)
        model = build_model(model_name, random_state=random_state)

        # Full pipeline
        clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
        clf.fit(X_train, y_train)

        # Predict
        y_pred = clf.predict(X_test)

        # Probabilities (if available)
        y_proba = None
        if can_compute_proba(clf.named_steps["model"]):
            try:
                y_proba = clf.predict_proba(X_test)
            except Exception:
                y_proba = None

        # Metrics
        m = compute_metrics(y_test, y_pred, y_proba)

        # -----------------------------
        # Display Metrics
        # -----------------------------
        st.subheader("📊 Evaluation Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", f"{m['accuracy']:.3f}")
        col2.metric("Precision (weighted)", f"{m['precision_weighted']:.3f}")
        col3.metric("Recall (weighted)", f"{m['recall_weighted']:.3f}")
        col4.metric("F1 (weighted)", f"{m['f1_weighted']:.3f}")
        if "roc_auc" in m and m["roc_auc"] == m["roc_auc"]:  # not NaN
            col5.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
        else:
            col5.metric("ROC-AUC", "N/A")

        # Classification report
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).T
        # Beautify
        st.subheader("🧾 Classification Report")
        st.dataframe(
            report_df.style.format(precision=3).background_gradient(cmap="Blues"), 
            use_container_width=True
        )

        # Confusion matrix
        st.subheader("🧩 Confusion Matrix")
        labels = sorted(y.unique())
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=labels, yticklabels=labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix — {model_name}")
        st.pyplot(fig, clear_figure=True)

        st.success("Done ✅")
else:
    st.info("Set options in the sidebar and click **Train & Evaluate**.")


# -----------------------------
# Notes Section
# -----------------------------
with st.expander("ℹ️ Notes & Tips"):
    st.markdown(
        """
- For small/test data, metrics can vary significantly with different train-test splits.
- **Class imbalance**: Models using `class_weight='balanced'` (Logistic Regression, SVM, Random Forest, Decision Tree) may help.
- If your dataset has many categories, one-hot encoding can expand features quickly. Keep data **small** on free tier.
- This app supports **multiclass** problems (Iris demo is 3-class).
"""
    )