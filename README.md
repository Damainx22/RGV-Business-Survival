# RGV Business Survival Predictor

Predicting SBA loan default risk for Texas small businesses using machine learning.
**Group 12 — Los Datos | UTRGV Intro to Data Science**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rgv-business-survival-mvj29hij6qyv4gykyhzlyn.streamlit.app/)

---

## Overview

Can we predict whether a Texas small business will default on an SBA 7(a) loan?

We built an end-to-end data pipeline that merges 4 government datasets, trains an XGBoost classifier, and deploys the results as a live web app. The model achieves 94% accuracy and 0.964 ROC-AUC, correctly identifying 81% of actual loan defaults.

**Disclaimer:** SBA loan default is used as a proxy for business financial distress. Loan default does not equal business closure. This project is for educational purposes only.

---

## Live App

[Open the Streamlit App](https://rgv-business-survival-mvj29hij6qyv4gykyhzlyn.streamlit.app/)

The app has 3 pages:
- **Prediction Tool** — enter loan details and get a default risk prediction (LOW / MEDIUM / HIGH)
- **EDA & Insights** — 12 visualizations exploring default patterns across industries, years, and geographies
- **About** — project summary, data sources, and methodology

---

## Repository Structure

RGV-Business-Survival/
├── app/
│   ├── app.py               # Streamlit app
│   └── requirements.txt     # Python dependencies
├── charts/                  # 12 EDA chart PNGs
├── models/                  # Trained model files
│   ├── xgb_model.pkl
│   ├── feature_columns.pkl
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── shap_summary.png
├── notebooks/
│   ├── 01_sba_cleaning.ipynb
│   ├── 02_acs_cleaning.ipynb
│   ├── 03_cbp_cleaning.ipynb
│   ├── 04_bds_cleaning.ipynb
│   ├── 05_merge_datasets.ipynb
│   ├── 06_eda.ipynb
│   ├── 07_feature_engineering.ipynb
│   └── 08_modeling.ipynb
└── README.md

Raw data is stored in Google Drive and not included in this repo due to file size.

---

## Data Sources

| Dataset | Source |
|---------|--------|
| SBA 7(a) FOIA Loan Data | [data.sba.gov](https://data.sba.gov/dataset/7-a-504-foia) |
| US Census ACS 5-Year Estimates | [api.census.gov](https://api.census.gov/data.html) |
| ZIP Code Business Patterns | [census.gov](https://www.census.gov/data/datasets/2022/econ/cbp/zbp-datasets.html) |
| Business Dynamics Statistics | [census.gov](https://www.census.gov/data/datasets/time-series/econ/bds/bds-datasets.html) |

---

## Model Results

| Metric | Random Forest | XGBoost |
|--------|--------------|---------|
| Accuracy | 91% | 94% |
| ROC-AUC | 0.944 | 0.964 |
| Default Recall | 31% | 81% |

We chose XGBoost because it correctly identified 81% of actual defaults vs only 31% for Random Forest — the critical metric for a risk assessment problem.

Top predictors by SHAP importance:
1. Loan term — shorter loans default 38x more than long-term ones
2. Industry — trucking has ~43% default rate
3. Business age — startups default at 17.1% vs 9.7% for established businesses

---

## Key Findings

- 12% overall Texas SBA loan default rate (FY 2018–2022)
- RGV default rate 13.7% vs Texas average 11.9%
- Short-term loans (0–5 years) default at 26.5% vs 0.7% for 20–25 year loans
- COVID 2020 default rate dropped to 8.2% due to federal relief programs
- Loan structure is a stronger predictor than local demographics

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| 01_sba_cleaning | Load and clean SBA 7(a) loan data, create defaulted target variable |
| 02_acs_cleaning | Pull Census ACS data via API for all Texas zip codes 2018–2022 |
| 03_cbp_cleaning | Load ZIP Business Patterns data for business density features |
| 04_bds_cleaning | Load Business Dynamics Statistics (county firm birth/death rates) |
| 05_merge_datasets | Merge all datasets — 8,064 rows, 22 columns, 0 nulls |
| 06_eda | 12 visualizations exploring default patterns |
| 07_feature_engineering | One-hot encode categoricals, expand to 608 features |
| 08_modeling | Train XGBoost, evaluate performance, SHAP analysis, save model |

