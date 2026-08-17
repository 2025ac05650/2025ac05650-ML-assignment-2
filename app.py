# ============================================================
# STREAMLIT APPLICATION
# PRODUCT SALES RETURN PREDICTION
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Product Return Prediction",
    page_icon="📦",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("📦 Product Sales Return Prediction")

st.write(
    "Machine Learning Assignment - 2"
)

st.write(
    "Upload a test CSV file, select a machine learning model, "
    "and evaluate return predictions."
)

# ============================================================
# MODEL DIRECTORY
# ============================================================

MODEL_DIR = "savedmodoles"

# ============================================================
# CHECK MODEL DIRECTORY
# ============================================================

if not os.path.exists(MODEL_DIR):

    st.error(
        "The 'saved_models' folder was not found. "
        "Please run the main assignment Python file first "
        "to save the trained models."
    )

    st.stop()

# ============================================================
# LOAD PREPROCESSOR
# ============================================================

preprocessor_path = os.path.join(
    MODEL_DIR,
    "preprocessor.pkl"
)

if not os.path.exists(preprocessor_path):

    st.error(
        "preprocessor.pkl was not found inside saved_models."
    )

    st.stop()

preprocessor = joblib.load(
    preprocessor_path
)

# ============================================================
# MODEL FILES
# ============================================================

MODEL_FILES = {

    "Logistic Regression":
        "logistic_regression.pkl",

    "Decision Tree":
        "decision_tree.pkl",

    "KNN":
        "knn.pkl",

    "Naive Bayes":
        "naive_bayes.pkl",

    "Random Forest":
        "random_forest.pkl"
}

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Select Model",
    list(MODEL_FILES.keys())
)

# ============================================================
# FILE UPLOAD
# ============================================================

st.header("1. Upload Test Data")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

# ============================================================
# PROCESS DATA
# ============================================================

if uploaded_file is not None:

    try:

        data = pd.read_csv(
            uploaded_file
        )

        st.success(
            "CSV file uploaded successfully!"
        )

        # ----------------------------------------------------
        # Display data
        # ----------------------------------------------------

        st.subheader("Uploaded Dataset")

        st.write(
            "Rows:",
            data.shape[0]
        )

        st.write(
            "Columns:",
            data.shape[1]
        )

        st.dataframe(
            data.head()
        )

        # ----------------------------------------------------
        # Check target
        # ----------------------------------------------------

        TARGET = "Returned"

        if TARGET not in data.columns:

            st.error(
                "The uploaded CSV must contain the "
                "'Returned' target column."
            )

            st.stop()

        # ----------------------------------------------------
        # Separate X and y
        # ----------------------------------------------------

        y_true = data[TARGET]

        X_data = data.drop(
            columns=[TARGET]
        )

        # ----------------------------------------------------
        # Remove columns that were not model features
        # ----------------------------------------------------

        columns_to_remove = []

        possible_columns = [
            "Order ID",
            "Customer Name",
            "Delivery Date"
        ]

        for column in possible_columns:

            if column in X_data.columns:

                columns_to_remove.append(
                    column
                )

        if columns_to_remove:

            X_data = X_data.drop(
                columns=columns_to_remove
            )

        # ----------------------------------------------------
        # Date processing
        # ----------------------------------------------------

        date_columns = []

        for column in X_data.columns:

            if "Date" in column:

                date_columns.append(
                    column
                )

        for column in date_columns:

            try:

                X_data[column] = pd.to_datetime(
                    X_data[column],
                    errors="coerce"
                )

                X_data[column + "_year"] = (
                    X_data[column].dt.year
                )

                X_data[column + "_month"] = (
                    X_data[column].dt.month
                )

                X_data[column + "_day"] = (
                    X_data[column].dt.day
                )

                X_data = X_data.drop(
                    columns=[column]
                )

            except Exception:
                pass

        # ----------------------------------------------------
        # Load selected model
        # ----------------------------------------------------

        model_file = os.path.join(
            MODEL_DIR,
            MODEL_FILES[selected_model]
        )

        if not os.path.exists(model_file):

            st.error(
                f"Model file not found: {model_file}"
            )

            st.stop()

        model = joblib.load(
            model_file
        )

        # ----------------------------------------------------
        # Transform data
        # ----------------------------------------------------

        X_processed = preprocessor.transform(
            X_data
        )

        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        y_pred = model.predict(
            X_processed
        )

        # ----------------------------------------------------
        # Prediction probability
        # ----------------------------------------------------

        y_prob = None

        if hasattr(
            model,
            "predict_proba"
        ):

            y_prob = model.predict_proba(
                X_processed
            )[:, 1]

        # ====================================================
        # RESULTS
        # ====================================================

        st.header("2. Model Evaluation")

        st.write(
            "Selected Model:",
            selected_model
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            y_true,
            y_pred
        )

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_true,
            y_pred
        )

        if y_prob is not None:

            auc = roc_auc_score(
                y_true,
                y_prob
            )

        else:

            auc = np.nan

        # ----------------------------------------------------
        # Display metrics
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Accuracy",
            f"{accuracy:.4f}"
        )

        col2.metric(
            "AUC",
            f"{auc:.4f}" if not np.isnan(auc)
            else "N/A"
        )

        col3.metric(
            "Precision",
            f"{precision:.4f}"
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Recall",
            f"{recall:.4f}"
        )

        col5.metric(
            "F1 Score",
            f"{f1:.4f}"
        )

        col6.metric(
            "MCC",
            f"{mcc:.4f}"
        )

        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        st.header("3. Confusion Matrix")

        cm = confusion_matrix(
            y_true,
            y_pred
        )

        cm_df = pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        )

        st.dataframe(
            cm_df
        )

        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        st.header("4. Classification Report")

        report = classification_report(
            y_true,
            y_pred,
            zero_division=0
        )

        st.text(
            report
        )

        # ====================================================
        # PREDICTION RESULTS
        # ====================================================

        st.header("5. Prediction Results")

        results = data.copy()

        results["Predicted_Returned"] = (
            y_pred
        )

        if y_prob is not None:

            results["Return_Probability"] = (
                y_prob
            )

        st.dataframe(
            results
        )

        # ====================================================
        # DOWNLOAD
        # ====================================================

        csv_output = results.to_csv(
            index=False
        )

        st.download_button(
            label="⬇️ Download Prediction Results",
            data=csv_output,
            file_name="prediction_results.csv",
            mime="text/csv"
        )

        st.success(
            "Evaluation completed successfully!"
        )

    except Exception as e:

        st.error(
            "An error occurred while processing the file."
        )

        st.exception(e)

else:

    st.info(
        "Please upload a CSV file to begin."
    )