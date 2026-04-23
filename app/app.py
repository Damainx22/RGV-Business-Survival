"""
RGV Business Survival Predictor
Streamlit App — 3 Pages: Prediction, EDA/Insights, About

Run locally:
    streamlit run app.py

Deploy: push to GitHub and connect via share.streamlit.io
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RGV Business Survival Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (clean, readable, professional)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* Root variables */
    :root {
        --primary: #1a1a2e;
        --accent: #e94560;
        --gold: #f5a623;
        --green: #27ae60;
        --red: #e74c3c;
        --surface: #16213e;
        --card: #0f3460;
        --text: #eaeaea;
        --muted: #a0a8b8;
    }

    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 100%);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f0f1f !important;
        border-right: 1px solid #2a2a4a;
    }

    /* Headers */
    h1, h2, h3 {
        font-family: 'Space Mono', monospace !important;
        color: var(--text) !important;
    }

    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #0f3460, #16213e);
        border: 1px solid #2a2a5a;
        border-radius: 12px;
        padding: 20px 24px;
        margin: 8px 0;
    }

    .risk-high {
        border-left: 4px solid var(--red);
        background: linear-gradient(135deg, rgba(231,76,60,0.15), #16213e);
    }

    .risk-medium {
        border-left: 4px solid var(--gold);
        background: linear-gradient(135deg, rgba(245,166,35,0.15), #16213e);
    }

    .risk-low {
        border-left: 4px solid var(--green);
        background: linear-gradient(135deg, rgba(39,174,96,0.15), #16213e);
    }

    /* Big number */
    .big-number {
        font-family: 'Space Mono', monospace;
        font-size: 3rem;
        font-weight: 700;
        line-height: 1;
    }

    /* Disclaimer */
    .disclaimer {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.82rem;
        color: var(--muted);
        margin-top: 16px;
    }

    /* Section label */
    .section-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--accent);
        margin-bottom: 6px;
    }

    /* Insight pill */
    .insight-pill {
        display: inline-block;
        background: rgba(233,69,96,0.15);
        border: 1px solid rgba(233,69,96,0.3);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.82rem;
        color: #f08090;
        margin: 4px 4px 4px 0;
    }

    /* Stmetric overrides */
    [data-testid="stMetricValue"] {
        font-family: 'Space Mono', monospace !important;
        color: var(--text) !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA: NAICS DESCRIPTIONS (top industries)
# These match what the model was trained on.
# ─────────────────────────────────────────────
NAICS_OPTIONS = [
    "Advertising Agencies",
    "All Other Business Support Services",
    "All Other Miscellaneous Ambulatory Health Care Services",
    "Beer, Wine, and Liquor Stores",
    "Coin-Operated Laundries and Drycleaners",
    "Computer Systems Design Services",
    "Dental Laboratories",
    "Drinking Places (Alcoholic Beverages)",
    "Electronic Shopping and Mail-Order Houses",
    "Full-Service Restaurants",
    "General Freight Trucking, Long Distance, Truckload",
    "General Freight Trucking, Long-Distance, Less Than Truckload",
    "Hair, Nail, and Skin Care Services",
    "Home Health Care Services",
    "Hotels (except Casino Hotels) and Motels",
    "Human Resources and Executive Search Consulting Services",
    "Institutional Furniture Manufacturing",
    "Investigation and Security Services",
    "Limited-Service Restaurants",
    "Management Consulting Services",
    "Medical, Dental, and Hospital Equipment and Supplies Merchant Wholesalers",
    "Miscellaneous Personal Services",
    "New Car Dealers",
    "Offices of Dentists",
    "Offices of Physicians (except Mental Health Specialists)",
    "Other Business Service Centers (including Copy Shops)",
    "Other Commercial Equipment Merchant Wholesalers",
    "Other Foundation, Structure, and Building Exterior Contractors",
    "Other Snack Food Manufacturing",
    "Real Estate Agents and Managers",
    "Sports and Recreation Instruction",
    "Tax Preparation Services",
    "Truck, Utility Trailer, and RV (Recreational Vehicle) Rental and Leasing",
    "Used Car Dealers",
    "Veterinary Services",
    "Other (not in list)",
]

BUSINESS_TYPES = [
    "Corporation",
    "Individual",
    "Limited Liability Company(LLC)",
    "Partnership",
    "Sole Proprietorship",
    "Non-Profit Organization",
]

BUSINESS_AGES = [
    "Existing or more than 2 years old",
    "New Business or 2 years or less",
    "Startup, Loan Funds will Open Business",
    "Unanswered",
    "Unknown",
]


# ─────────────────────────────────────────────
# MODEL LOADING (cached)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """
    Try multiple paths so the app works both:
    - Locally in Colab (Google Drive path)
    - On Streamlit Cloud (models/ folder in the repo)
    """
    candidate_dirs = [
        os.path.join(os.path.dirname(__file__), "..", "models"),     # repo: models/
        os.path.join(os.path.dirname(__file__), "models"),            # app/models/
        "/content/drive/MyDrive/rgv_business_survival/models",        # Colab Drive
    ]
    for d in candidate_dirs:
        model_path = os.path.join(d, "xgb_model.pkl")
        cols_path  = os.path.join(d, "feature_columns.pkl")
        if os.path.exists(model_path) and os.path.exists(cols_path):
            model   = joblib.load(model_path)
            columns = joblib.load(cols_path)
            return model, columns, d
    return None, None, None


@st.cache_data(show_spinner=False)
def find_charts():
    """Find chart PNGs from any of the known locations."""
    candidate_dirs = [
        os.path.join(os.path.dirname(__file__), "..", "charts"),
        os.path.join(os.path.dirname(__file__), "charts"),
        "/content/drive/MyDrive/rgv_business_survival/charts",
    ]
    for d in candidate_dirs:
        pngs = sorted(glob.glob(os.path.join(d, "*.png")))
        if pngs:
            return pngs
    return []


# ─────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────
def build_input_row(
    naics_desc,
    business_type,
    business_age,
    term_months,
    gross_approval,
    columns,
):
    """
    Build a single-row DataFrame that matches the 608-column feature matrix.
    Numeric columns are filled directly; one-hot columns are set to 0 by default,
    then the matching column for each categorical selection is set to 1.
    """
    row = pd.DataFrame([{col: 0 for col in columns}])

    # ── Numeric features ──────────────────────────────────────────────
    if "terminmonths" in columns:
        row["terminmonths"] = term_months
    if "grossapproval" in columns:
        row["grossapproval"] = gross_approval

    # ── One-hot: naicsdescription ──────────────────────────────────────
    # The training data was one-hot encoded with drop_first=True via
    # pd.get_dummies, so there's a "base" category that maps to all zeros.
    # We search for the matching column name pattern.
    naics_col = f"naicsdescription_{naics_desc}"
    if naics_col in row.columns:
        row[naics_col] = 1
    # If "Other (not in list)" or column not found → all zeros (base category)

    # ── One-hot: businesstype ──────────────────────────────────────────
    btype_col = f"businesstype_{business_type}"
    if btype_col in row.columns:
        row[btype_col] = 1

    # ── One-hot: businessage ──────────────────────────────────────────
    bage_col = f"businessage_{business_age}"
    if bage_col in row.columns:
        row[bage_col] = 1

    return row


def risk_level(prob):
    if prob >= 0.15:
        return "HIGH", "risk-high", "🔴"
    elif prob >= 0.08:
        return "MEDIUM", "risk-medium", "🟡"
    else:
        return "LOW", "risk-low", "🟢"


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 RGV Business\nSurvival Predictor")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔮 Prediction Tool", "📈 EDA & Insights", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Group 12 · Los Datos\nUTRGV Intro to Data Science")


# ─────────────────────────────────────────────
# PAGE 1 — PREDICTION TOOL
# ─────────────────────────────────────────────
if page == "🔮 Prediction Tool":

    st.markdown("## 🔮 SBA Loan Default Risk Predictor")
    st.markdown(
        "Enter details about a Texas small business SBA loan to estimate its probability of default. "
        "This model was trained on ~8,000 Texas SBA 7(a) loans from fiscal years 2018–2022."
    )

    model, columns, model_dir = load_model()

    if model is None:
        st.error(
            "⚠️ Model files not found. Make sure `xgb_model.pkl` and `feature_columns.pkl` "
            "are in a `models/` folder at the root of the repository."
        )
        st.stop()

    # ── Input form ────────────────────────────────────────────────────
    st.markdown("### Loan Details")

    col1, col2 = st.columns(2)

    with col1:
        naics_desc = st.selectbox(
            "Industry (NAICS Description)",
            NAICS_OPTIONS,
            index=10,  # default: General Freight Trucking
            help="Select the primary industry of the business",
        )
        business_type = st.selectbox(
            "Business Type",
            BUSINESS_TYPES,
            index=0,  # Corporation
            help="Legal entity type of the business",
        )
        business_age = st.selectbox(
            "Business Age at Loan Origination",
            BUSINESS_AGES,
            index=0,  # Existing
            help="Was the business new (≤2 yrs) or established when the loan was issued?",
        )

    with col2:
        term_months = st.slider(
            "Loan Term (months)",
            min_value=12,
            max_value=360,
            value=120,
            step=12,
            help="Duration of the SBA loan in months. Shorter terms have higher default rates.",
        )
        gross_approval = st.number_input(
            "Gross Approval Amount ($)",
            min_value=5_000,
            max_value=5_000_000,
            value=150_000,
            step=5_000,
            format="%d",
            help="Total approved loan amount in USD",
        )

    # ── Predict ───────────────────────────────────────────────────────
    if st.button("🔮 Predict Default Risk", use_container_width=True, type="primary"):

        input_df = build_input_row(
            naics_desc=naics_desc,
            business_type=business_type,
            business_age=business_age,
            term_months=term_months,
            gross_approval=gross_approval,
            columns=columns,
        )

        prob_default = float(model.predict_proba(input_df)[0][1])
        prob_survive = 1 - prob_default
        level, css_class, emoji = risk_level(prob_default)

        st.markdown("---")
        st.markdown("### Prediction Result")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.markdown(f"""
            <div class="metric-card {css_class}">
                <div class="section-label">Default Risk Level</div>
                <div class="big-number">{emoji} {level}</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="section-label">Probability of Default</div>
                <div class="big-number" style="color:#e74c3c">{prob_default:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="section-label">Probability of Survival</div>
                <div class="big-number" style="color:#27ae60">{prob_survive:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        # ── What drives this prediction ───────────────────────────────
        st.markdown("#### Key Risk Factors for This Prediction")

        factors = []

        if term_months <= 60:
            factors.append(f"⚠️ **Short loan term ({term_months} months)** — loans under 5 years default at ~26.5%")
        elif term_months >= 240:
            factors.append(f"✅ **Long loan term ({term_months} months)** — loans over 20 years default at only ~0.7%")

        if business_age == "New Business or 2 years or less":
            factors.append("⚠️ **New business** — startups default at 17.1% vs 9.7% for established businesses")
        elif business_age == "Existing or more than 2 years old":
            factors.append("✅ **Established business** — lower default risk than new businesses")

        if business_type == "Individual":
            factors.append("⚠️ **Individual owner** — individual-owned businesses default at 16.9%")
        elif business_type == "Corporation":
            factors.append("✅ **Corporation** — corporations default at 11.7%, lower than individuals")

        if naics_desc == "General Freight Trucking, Long Distance, Truckload":
            factors.append("⚠️ **Trucking industry** — historically has ~43% default rate in this dataset")

        if not factors:
            factors.append("ℹ️ No extreme risk factors detected for this combination.")

        for f in factors:
            st.markdown(f)

        # ── Disclaimer ────────────────────────────────────────────────
        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Disclaimer:</strong> This tool predicts <em>SBA loan default risk</em> as a proxy for
            small business financial survival. Loan default ≠ business closure. This model is trained on
            Texas SBA 7(a) loans (FY 2018–2022) and is for <strong>educational purposes only</strong>.
            It should not be used for actual lending or business decisions. Model accuracy: 94% overall,
            ROC-AUC: 0.964.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 2 — EDA & INSIGHTS
# ─────────────────────────────────────────────
elif page == "📈 EDA & Insights":

    st.markdown("## 📈 EDA & Insights")
    st.markdown(
        "Exploratory analysis of ~8,000 Texas SBA 7(a) loans (FY 2018–2022). "
        "Overall default rate: **12.0%**"
    )

    # Key findings
    st.markdown("### 🔑 Key Findings")

    kf1, kf2, kf3, kf4 = st.columns(4)
    with kf1:
        st.metric("RGV Default Rate", "13.7%", delta="+1.8% vs Texas avg", delta_color="inverse")
    with kf2:
        st.metric("Startup Default Rate", "17.1%", delta="+7.4% vs established", delta_color="inverse")
    with kf3:
        st.metric("Short-Term Default Rate", "26.5%", help="Loans 0–5 years")
    with kf4:
        st.metric("Long-Term Default Rate", "0.7%", help="Loans 20–25 years")

    st.markdown("---")

    # Chart display
    charts = find_charts()

    if not charts:
        st.warning(
            "📂 Chart images not found. To display charts here, copy your saved PNGs "
            "from Google Drive into a `charts/` folder at the root of the repository and "
            "push to GitHub."
        )

        # Show hardcoded insight text as fallback
        st.markdown("### Insights Summary (text fallback)")

        insights = {
            "Default by Industry": [
                "Appliance repair: **63.6%** default rate (highest)",
                "Trucking (long-distance): **~43%** default rate",
                "Restaurants and food services: elevated risk",
            ],
            "Default by Loan Term": [
                "0–5 year loans: **26.5%** default rate",
                "5–10 year loans: moderate risk",
                "20–25 year loans: **0.7%** default rate (lowest)",
            ],
            "Default by Business Type": [
                "Individual owners: **16.9%** default",
                "Corporations: **11.7%** default",
                "Partnerships: **9.1%** default (lowest)",
            ],
            "COVID Impact (2020–2021)": [
                "Default rate **dropped** during COVID — likely due to government relief programs",
                "2020 default rate: **8.2%** (vs ~12% baseline)",
            ],
            "RGV vs Rest of Texas": [
                "RGV default rate: **13.7%**",
                "Rest of Texas: **11.9%**",
                "Demographics (income, poverty) weakly correlated with defaults",
            ],
        }

        for title, bullets in insights.items():
            with st.expander(f"📊 {title}", expanded=False):
                for b in bullets:
                    st.markdown(f"- {b}")

    else:
        # Display charts in a 2-column grid
        chart_labels = [
            "Default Rate by Industry (Top 15)",
            "Default Rate by Year",
            "Loan Amount vs Default",
            "RGV vs Rest of Texas",
            "Income vs Default (by Zip)",
            "Default by Business Type",
            "Default by Business Age",
            "Feature Correlation Heatmap",
            "Top RGV Industries by Volume",
            "Default by Loan Term",
            "RGV Survived vs Defaulted by Industry",
            "Income Quartile: RGV vs Texas",
        ]

        for i in range(0, len(charts), 2):
            col_a, col_b = st.columns(2)
            with col_a:
                label = chart_labels[i] if i < len(chart_labels) else os.path.basename(charts[i])
                st.markdown(f"**{label}**")
                st.image(charts[i], use_container_width=True)
            if i + 1 < len(charts):
                with col_b:
                    label = chart_labels[i+1] if i+1 < len(chart_labels) else os.path.basename(charts[i+1])
                    st.markdown(f"**{label}**")
                    st.image(charts[i+1], use_container_width=True)

    # Model performance section
    st.markdown("---")
    st.markdown("### 🤖 Model Performance")

    pm1, pm2, pm3, pm4 = st.columns(4)
    with pm1:
        st.metric("Model", "XGBoost")
    with pm2:
        st.metric("Overall Accuracy", "94%")
    with pm3:
        st.metric("ROC-AUC Score", "0.964")
    with pm4:
        st.metric("Default Recall", "81%", help="% of actual defaults correctly identified")

    st.markdown(
        "XGBoost significantly outperformed Random Forest on default recall "
        "(81% vs 31%), which is the critical metric for identifying at-risk loans."
    )

    # Model charts
    _, model_dir = None, None
    _, _, model_dir = load_model()

    if model_dir:
        conf_path  = os.path.join(model_dir, "confusion_matrix.png")
        feat_path  = os.path.join(model_dir, "feature_importance.png")
        shap_path  = os.path.join(model_dir, "shap_summary.png")

        mc1, mc2 = st.columns(2)
        with mc1:
            if os.path.exists(conf_path):
                st.markdown("**Confusion Matrix**")
                st.image(conf_path, use_container_width=True)
        with mc2:
            if os.path.exists(feat_path):
                st.markdown("**Top 20 Feature Importances**")
                st.image(feat_path, use_container_width=True)

        if os.path.exists(shap_path):
            st.markdown("**SHAP Summary Plot** — why the model makes each prediction")
            st.image(shap_path, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 3 — ABOUT
# ─────────────────────────────────────────────
elif page == "ℹ️ About":

    st.markdown("## ℹ️ About This Project")

    st.markdown("""
    ### RGV Business Survival Predictor

    This project was built for the **UTRGV Intro to Data Science** class as a final project.
    Our team — **Group 12 "Los Datos"** — used machine learning to predict whether an
    SBA 7(a) small business loan would default, as a proxy for small business financial survival.

    ---

    ### The Team
    - Damian Ramirez
    - Bukola
    - Roen

    ---

    ### What We Built

    | Step | Description |
    |------|-------------|
    | Data Collection | SBA 7(a) FOIA loan data, US Census ACS, County Business Patterns, Business Dynamics Statistics |
    | Data Cleaning | Filtered to Texas, FY 2018–2022, ~8,000 loans |
    | EDA | 12 visualizations exploring default patterns by industry, time, geography, and business attributes |
    | Modeling | XGBoost classifier — 94% accuracy, ROC-AUC 0.964, 81% recall on defaults |
    | Explainability | SHAP values to explain individual predictions |
    | Deployment | This Streamlit app |

    ---

    ### Data Sources
    - **SBA 7(a) Loan Data** — [SBA FOIA Data](https://data.sba.gov/dataset/7-a-504-foia)
    - **US Census ACS** — Zip-code level income, poverty, and unemployment
    - **County Business Patterns / ZBP** — Business density by zip code
    - **Business Dynamics Statistics** — County-level firm birth/death rates

    ---

    ### ⚠️ Important Disclaimer

    > **Loan default ≠ business closure.**
    >
    > This model predicts whether an SBA loan will be charged off (defaulted),
    > which we use as a *proxy* for business financial distress. A business that defaults
    > on an SBA loan may not close, and a business may close without defaulting on a loan.
    >
    > This tool is for **educational purposes only** and should not be used for
    > actual lending, investment, or business decisions.
    >
    > The model was trained on Texas data from FY 2018–2022 and may not generalize
    > to other states, time periods, or loan types.

    ---

    ### Technical Details
    - **Framework**: Python, scikit-learn, XGBoost, Pandas
    - **App**: Streamlit
    - **Training data**: 8,064 rows × 608 features (after one-hot encoding)
    - **Class imbalance handling**: `scale_pos_weight` (negative/positive ratio = 7.36)
    - **Top predictor**: Loan term (`terminmonths`) — SHAP rank #1
    """)
