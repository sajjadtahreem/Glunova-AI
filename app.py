"""
============================================================
                         GLUNOVA AI
       AI-based Gestational Diabetes Mellitus Prediction Tool
============================================================

Run:
    streamlit run app.py

Research Use Only.
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# RESOURCE PATH
# ============================================================

def resource_path(relative_path):
    """Return correct path for development and PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = Path(resource_path("model/model.pkl"))
PREPROCESSOR_PATH = Path(resource_path("model/gdm_preprocessor.pkl"))
LOGO_PATH = Path(resource_path("glunova_logo.png"))


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Glunova AI | GDM Prediction",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# MODEL + PREPROCESSOR
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_preprocessor():
    return joblib.load(PREPROCESSOR_PATH)


if not MODEL_PATH.exists():
    st.error(
        "model.pkl was not found. Please place it inside the "
        "'model' folder."
    )
    st.stop()

if not PREPROCESSOR_PATH.exists():
    st.error(
        "gdm_preprocessor.pkl was not found. Please place it inside "
        "the 'model' folder."
    )
    st.stop()

try:
    model = load_model()
    prep = load_preprocessor()
except Exception as e:
    st.error(f"Unable to load the Glunova AI model files: {e}")
    st.stop()


MODEL_FEATURES = prep["model_features"]
RAW_FEATURE_COLUMNS = prep["raw_feature_columns_after_dummies"]
LOG_COLUMNS = prep["log_columns"]
REMOVED_COLUMNS = prep["removed_columns"]
IMPUTER = prep["imputer"]


# ============================================================
# USER-FACING FEATURE DEFINITIONS
# ============================================================

CLINICAL_FEATURES = [
    "Maternal age",
    "Gravidity",
    "Family History of Diabetes",
    "Mean Diastolic BP",
    "Mean Systolic BP",
    "Fasting Glucose (mg/dl)",
    "Hb (g/dl)",
    "RBC (millions/ml)",
    "WBC (10^3/uL)",
    "MCHC (g/dl)",
    "Lymphocytes (Absolute count 10^3/uL)",
    "Eosinophils (Absolute count 10^3/uL)",
    "BMI",
    "Blood Type",
]


# ============================================================
# PREPROCESSING
# ============================================================

def add_blood_type_dummies(df):
    """
    Reproduce the notebook's Blood Type encoding:
    pd.get_dummies(..., drop_first=True).

    The saved model expects Blood Type_AB and Blood Type_B.
    Blood Type_O is a reference category and was removed later.
    """

    df = df.copy()

    if "Blood Type" not in df.columns:
        raise ValueError("Blood Type column is required.")

    blood = df["Blood Type"].astype(str).str.strip().str.upper()

    df["Blood Type_AB"] = (blood == "AB").astype(int)
    df["Blood Type_B"] = (blood == "B").astype(int)
    df["Blood Type_O"] = (blood == "O").astype(int)

    df.drop(columns=["Blood Type"], inplace=True)

    return df


def preprocess_for_model(df):
    """
    Reproduce the preprocessing used before fitting the saved model:

    1. Blood Type dummy encoding
    2. Iterative imputation using the saved training imputer
    3. Remove Random Glucose
    4. log1p selected skewed variables
    5. Remove selected features
    6. Return exact model feature order
    """

    data = df.copy()
    data.columns = data.columns.str.strip()

    # Blood type
    data = add_blood_type_dummies(data)

    # Ensure every raw training column exists.
    for col in RAW_FEATURE_COLUMNS:
        if col not in data.columns:
            data[col] = np.nan

    data = data[RAW_FEATURE_COLUMNS]

    # Iterative imputation fitted on the original training split.
    data = pd.DataFrame(
        IMPUTER.transform(data),
        columns=RAW_FEATURE_COLUMNS,
        index=data.index,
    )

    # Remove Random Glucose exactly as in notebook.
    if "Random Glucose (mg/dl)" in data.columns:
        data.drop(columns=["Random Glucose (mg/dl)"], inplace=True)

    # Log transform exactly as in notebook.
    for col in LOG_COLUMNS:
        if col in data.columns:
            # Clip tiny negative numerical artifacts caused by imputation.
            data[col] = data[col].clip(lower=0)
            data[col] = np.log1p(data[col])

    # Feature selection exactly as in notebook.
    for col in REMOVED_COLUMNS:
        if col in data.columns:
            data.drop(columns=[col], inplace=True)

    # Exact model order.
    missing = [c for c in MODEL_FEATURES if c not in data.columns]

    if missing:
        raise ValueError(
            "The preprocessed data is missing model features: "
            + ", ".join(missing)
        )

    return data[MODEL_FEATURES]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("About")

    st.info(
        """
        **Glunova AI** is an AI-based classification tool
        developed for prediction of Gestational Diabetes
        Mellitus (GDM).

        **Research Use Only.**
        """
    )

    st.divider()

    st.markdown("### Model")
    st.success("Random Forest Classifier")

    st.divider()

    st.markdown("### Prediction Output")
    st.write("**YES** = GDM predicted")
    st.write("**NO** = GDM not predicted")


# ============================================================
# HEADER / BRANDING
# ============================================================

if LOGO_PATH.exists():
    logo_left, logo_center, logo_right = st.columns([1, 2, 1])
    with logo_center:
        st.image(str(LOGO_PATH), width=520)
else:
    st.title("Glunova AI")
    st.subheader("AI-based Gestational Diabetes Mellitus Prediction Tool")

st.caption("AI-based Gestational Diabetes Mellitus Prediction Tool")
st.divider()


# ============================================================
# TABS
# ============================================================

tab_single, tab_batch, tab_about = st.tabs(
    [
        "Single Prediction",
        "Batch Prediction",
        "About",
    ]
)


# ============================================================
# TAB 1 : SINGLE PREDICTION
# ============================================================

with tab_single:

    st.subheader("Maternal Clinical Information")

    with st.form("gdm_prediction_form"):

        left, right = st.columns(2)

        # ----------------------------------------------------
        # LEFT
        # ----------------------------------------------------

        with left:

            st.markdown("## 👤 Demographic & Obstetric Information")

            age = st.number_input(
                "Maternal Age (years)",
                min_value=10.0,
                max_value=70.0,
                value=None,
                placeholder="Enter maternal age",
                step=1.0,
            )

            gravidity = st.number_input(
                "Gravidity",
                min_value=0.0,
                max_value=20.0,
                value=None,
                placeholder="Enter gravidity",
                step=1.0,
            )

            family_history = st.selectbox(
                "Family History of Diabetes",
                ["No", "Yes"],
                index=None,
                placeholder="Select",
            )

            bmi = st.number_input(
                "BMI",
                min_value=10.0,
                max_value=80.0,
                value=None,
                placeholder="Enter BMI",
                step=0.1,
                format="%.1f",
            )

            blood_type = st.selectbox(
                "Blood Type",
                ["A", "B", "AB", "O"],
                index=None,
                placeholder="Select blood type",
            )

            st.divider()

            st.markdown("## 🩺 Blood Pressure")

            systolic = st.number_input(
                "Mean Systolic BP",
                min_value=50.0,
                max_value=250.0,
                value=None,
                placeholder="Enter mean systolic BP",
                step=0.1,
                format="%.1f",
            )

            diastolic = st.number_input(
                "Mean Diastolic BP",
                min_value=30.0,
                max_value=150.0,
                value=None,
                placeholder="Enter mean diastolic BP",
                step=0.1,
                format="%.1f",
            )

        # ----------------------------------------------------
        # RIGHT
        # ----------------------------------------------------

        with right:

            st.markdown("## 🧪 Biochemical Investigation")

            fasting_glucose = st.number_input(
                "Fasting Glucose (mg/dl)",
                min_value=20.0,
                max_value=500.0,
                value=None,
                placeholder="Enter fasting glucose",
                step=0.1,
                format="%.1f",
            )

            st.divider()

            st.markdown("## 🩸 Hematological Parameters")

            hb = st.number_input(
                "Hb (g/dl)",
                min_value=1.0,
                max_value=30.0,
                value=None,
                placeholder="Enter Hb",
                step=0.01,
                format="%.2f",
            )

            rbc = st.number_input(
                "RBC (millions/ml)",
                min_value=0.1,
                max_value=15.0,
                value=None,
                placeholder="Enter RBC",
                step=0.01,
                format="%.2f",
            )

            wbc = st.number_input(
                "WBC (10^3/uL)",
                min_value=0.1,
                max_value=100.0,
                value=None,
                placeholder="Enter WBC",
                step=0.01,
                format="%.2f",
            )

            mchc = st.number_input(
                "MCHC (g/dl)",
                min_value=1.0,
                max_value=60.0,
                value=None,
                placeholder="Enter MCHC",
                step=0.01,
                format="%.2f",
            )

            lymphocytes = st.number_input(
                "Lymphocytes (Absolute count 10^3/uL)",
                min_value=0.0,
                max_value=100.0,
                value=None,
                placeholder="Enter lymphocyte count",
                step=0.01,
                format="%.2f",
            )

            eosinophils = st.number_input(
                "Eosinophils (Absolute count 10^3/uL)",
                min_value=0.0,
                max_value=100.0,
                value=None,
                placeholder="Enter eosinophil count",
                step=0.01,
                format="%.2f",
            )

        st.divider()

        predict = st.form_submit_button(
            "🔬 Predict GDM",
            use_container_width=True,
            type="primary",
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    if predict:

        values = {
            "Maternal age": age,
            "Gravidity": gravidity,
            "Family History of Diabetes": 1 if family_history == "Yes" else 0,
            "Mean Diastolic BP": diastolic,
            "Mean Systolic BP": systolic,
            "Fasting Glucose (mg/dl)": fasting_glucose,
            "Hb (g/dl)": hb,
            "RBC (millions/ml)": rbc,
            "WBC (10^3/uL)": wbc,
            "MCHC (g/dl)": mchc,
            "Lymphocytes (Absolute count 10^3/uL)": lymphocytes,
            "Eosinophils (Absolute count 10^3/uL)": eosinophils,
            "BMI": bmi,
            "Blood Type": blood_type,
        }

        missing_values = [
            name for name, value in values.items()
            if value is None
        ]

        if missing_values:

            st.warning(
                "Please provide all required values before prediction:\n\n"
                + ", ".join(missing_values)
            )

        else:

            try:

                patient = pd.DataFrame([values])
                model_input = preprocess_for_model(patient)

                prediction = int(model.predict(model_input)[0])

                st.divider()
                st.subheader("GDM Prediction")

                if prediction == 1:

                    st.error("## YES")

                    st.markdown(
                        """
                        The entered maternal clinical profile is
                        **predicted to be positive for Gestational
                        Diabetes Mellitus (GDM)**.
                        """
                    )

                else:

                    st.success("## NO")

                    st.markdown(
                        """
                        The entered maternal clinical profile is
                        **predicted to be negative for Gestational
                        Diabetes Mellitus (GDM)**.
                        """
                    )

                st.info(
                    "Research Use Only. This prediction should not "
                    "replace professional clinical diagnosis or "
                    "medical decision-making."
                )

                st.markdown("### Submitted Values")

                show = patient.T.reset_index()
                show.columns = ["Variable", "Value"]

                st.dataframe(
                    show,
                    hide_index=True,
                    use_container_width=True,
                )

                # ------------------------------------------------
                # DOWNLOAD
                # ------------------------------------------------

                result = patient.copy()
                result["GDM Prediction"] = (
                    "Yes" if prediction == 1 else "No"
                )

                st.download_button(
                    "📥 Download Prediction Report",
                    data=result.to_csv(index=False).encode("utf-8"),
                    file_name="Glunova_AI_GDM_Prediction.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            except Exception as e:

                st.error(
                    "An error occurred while generating the prediction: "
                    + str(e)
                )


# ============================================================
# TAB 2 : BATCH PREDICTION
# ============================================================

with tab_batch:

    st.subheader("Batch GDM Prediction")

    st.caption(
        "Upload a CSV file containing the clinical variables. "
        "The application will automatically preprocess the data "
        "using the same preprocessing scheme used during model "
        "development."
    )

    # Exact columns accepted for the upload.
    batch_template = pd.DataFrame(
        columns=CLINICAL_FEATURES
    )

    st.download_button(
        label="📥 Download Blank Template",
        data=batch_template.to_csv(index=False).encode("utf-8"),
        file_name="Glunova_AI_Template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded is not None:

        try:

            data = pd.read_csv(uploaded)
            data.columns = data.columns.str.strip()

            missing = [
                col for col in CLINICAL_FEATURES
                if col not in data.columns
            ]

            if missing:

                st.error(
                    "The uploaded file is missing these required "
                    "columns:\n\n"
                    + ", ".join(missing)
                )

            else:

                model_input = preprocess_for_model(
                    data[CLINICAL_FEATURES]
                )

                pred = model.predict(model_input)

                results = data.copy()

                results["GDM Prediction"] = [
                    "Yes" if int(x) == 1 else "No"
                    for x in pred
                ]

                st.success(
                    f"Successfully analyzed {len(results)} patient(s)."
                )

                c1, c2 = st.columns(2)

                with c1:
                    st.metric(
                        "Predicted GDM: YES",
                        int((pred == 1).sum()),
                    )

                with c2:
                    st.metric(
                        "Predicted GDM: NO",
                        int((pred == 0).sum()),
                    )

                st.dataframe(
                    results,
                    use_container_width=True,
                )

                st.download_button(
                    "📥 Download GDM Predictions",
                    data=results.to_csv(index=False).encode("utf-8"),
                    file_name="Glunova_AI_GDM_Predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:

            st.error(
                "Unable to process the uploaded file: " + str(e)
            )


# ============================================================
# TAB 3 : ABOUT
# ============================================================

with tab_about:

    st.header("About Glunova AI")

    st.markdown(
        """
        ### Glunova AI

        **Glunova AI** is an Artificial Intelligence-based
        classification tool developed for prediction of
        **Gestational Diabetes Mellitus (GDM)**.

        The application uses a trained **Random Forest classifier**
        to analyze selected maternal demographic, obstetric,
        clinical, biochemical, hematological and anthropometric
        variables.

        ---

        ### Model

        - Random Forest Classifier
        - Binary classification
        - **YES** = GDM predicted
        - **NO** = GDM not predicted
        - SMOTEENN was used during model training to address
          class imbalance.

        ---

        ### Input Variables

        The prediction model uses:

        - Maternal age
        - Gravidity
        - Family History of Diabetes
        - Mean Diastolic BP
        - Mean Systolic BP
        - Fasting Glucose
        - Hb
        - RBC
        - WBC
        - MCHC
        - Lymphocyte count
        - Eosinophil count
        - Blood Type
        - BMI

        ---

        ### Output

        Glunova AI provides a binary classification:

        **YES → GDM predicted**

        **NO → GDM not predicted**

        No probability score is displayed.

        ---

        ### Disclaimer

        **Research Use Only**

        Glunova AI is intended for research and educational
        purposes. It should not be used as a substitute for
        professional medical diagnosis or clinical decision-making.
        """
    )

    st.divider()

    with st.expander("📋 Variable Reference"):

        reference = pd.DataFrame(
            {
                "Variable": CLINICAL_FEATURES,
                "Description": [
                    "Maternal age in years",
                    "Number of pregnancies / gravidity",
                    "Family history of diabetes (Yes/No)",
                    "Mean diastolic blood pressure",
                    "Mean systolic blood pressure",
                    "Fasting glucose (mg/dl)",
                    "Hemoglobin (g/dl)",
                    "RBC count (millions/ml)",
                    "WBC count (10^3/uL)",
                    "Mean corpuscular hemoglobin concentration",
                    "Absolute lymphocyte count (10^3/uL)",
                    "Absolute eosinophil count (10^3/uL)",
                    "Body mass index",
                    "Maternal blood type"
                ],
            }
        )

        st.dataframe(
            reference,
            hide_index=True,
            use_container_width=True,
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "© 2026 Glunova AI | AI-based Gestational Diabetes Mellitus "
    "Prediction Tool"
)

st.caption(
    "Developed for research purposes using a Random Forest "
    "classification model."
)
