# Glunova-AI
AI based Gestational Diabetes Mellitus Prediction


# Installation and Setup
Clone the Repository. First, install Git if it is not already installed.
Then clone the Glunova-AI repository.
Open windows powershell or linux shell
```bash
git clone https://github.com/sajjadtahreem/Glunova-AI
cd Glunova-AI
conda create -n glunova python=3.11 -y
conda activate glunova
python --version
Python 3.11.x
cd path\to\Glunova-AI
python -m pip install -r requirements.txt
streamlit run app.py
```

## Output

- YES = GDM predicted
- NO = GDM not predicted


## About the Project

Glunova-AI was developed as part of a research study investigating machine-learning-based prediction of GDM using routinely available clinical, hematological, and demographic variables.
The final prediction model uses 15 selected features derived from the study's feature-engineering and feature-selection workflow.

### Final Model Features

The model uses the following variables:

1. Maternal age
2. Gravidity
3. Family History of Diabetes
4. Mean Diastolic Blood Pressure
5. Mean Systolic Blood Pressure
6. Fasting Glucose
7. Hemoglobin (Hb)
8. Red Blood Cell count (RBC)
9. White Blood Cell count (WBC)
10. Mean Corpuscular Hemoglobin Concentration (MCHC)
11. Absolute Lymphocyte count
12. Absolute Eosinophil count
13. Blood Type AB
14. Blood Type B
15. Body Mass Index (BMI)
---

## Model

The project contains two serialized model-related files:

```text
model/
├── model.pkl
└── gdm_preprocessor.pkl
