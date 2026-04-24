"""
RGV Business Survival Predictor
Streamlit App — 3 Pages: Prediction, EDA/Insights, About

Run locally:
    streamlit run app.py

Deploy: push to GitHub and connect via share.streamlit.io
"""

import streamlit as st
import pandas as pd
import joblib
import os
import glob

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RGV Business Survival Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (minimal, light, professional)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --background: #f8fafc;
        --surface: #ffffff;
        --surface-soft: #f1f5f9;
        --border: #e2e8f0;
        --text: #0f172a;
        --muted: #64748b;
        --accent: #2563eb;
        --accent-soft: #eff6ff;
        --success: #15803d;
        --warning: #b45309;
        --danger: #b91c1c;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
    }

    section[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    h1, h2, h3, h4 {
        color: var(--text) !important;
        letter-spacing: -0.03em;
    }

    p, li, label, span, div {
        color: var(--text);
    }

    .page-header {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 32px;
        margin-bottom: 28px;
    }

    .eyebrow {
        color: var(--accent);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .subtitle {
        color: var(--muted);
        font-size: 1.02rem;
        line-height: 1.65;
        max-width: 760px;
        margin-top: 8px;
    }

    .metric-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 22px 24px;
        margin: 8px 0;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 8px;
    }

    .big-number {
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1;
        letter-spacing: -0.04em;
    }

    .risk-high {
        border-left: 4px solid var(--danger);
    }

    .risk-medium {
        border-left: 4px solid var(--warning);
    }

    .risk-low {
        border-left: 4px solid var(--success);
    }

    .info-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px 22px;
        margin: 14px 0;
    }

    .soft-card {
        background: var(--accent-soft);
        border: 1px solid #bfdbfe;
        border-radius: 16px;
        padding: 20px 22px;
        margin: 14px 0;
    }

    .disclaimer {
        background: var(--surface-soft);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 16px 18px;
        font-size: 0.9rem;
        color: var(--muted);
        margin-top: 18px;
        line-height: 1.55;
    }

    .chart-title {
        font-weight: 650;
        font-size: 0.98rem;
        margin: 10px 0 8px 0;
        color: var(--text);
    }

    .small-note {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.55;
    }

    div[data-testid="stMetric"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    [data-testid="stMetricLabel"] p {
        color: var(--muted) !important;
        font-size: 0.82rem !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-size: 1.8rem !important;
        letter-spacing: -0.04em;
    }

    .stButton > button {
        background: var(--text);
        color: white;
        border: 1px solid var(--text);
        border-radius: 12px;
        padding: 0.7rem 1rem;
        font-weight: 650;
    }

    .stButton > button:hover {
        background: #1e293b;
        border-color: #1e293b;
        color: white;
    }

    hr {
        border: none;
        border-top: 1px solid var(--border);
        margin: 28px 0;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DATA: NAICS DESCRIPTIONS
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
# MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    candidate_dirs = [
        os.path.join(os.path.dirname(__file__), "..", "models"),
        os.path.join(os.path.dirname(__file__), "models"),
        "/content/drive/MyDrive/rgv_business_survival/models",
    ]
    for d in candidate_dirs:
        model_path = os.path.join(d, "xgb_model.pkl")
        cols_path = os.path.join(d, "feature_columns.pkl")
        if os.path.exists(model_path) and os.path.exists(cols_path):
            model = joblib.load(model_path)
            columns = joblib.load(cols_path)
            return model, columns, d
    return None, None, None


@st.cache_data(show_spinner=False)
def find_charts():
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
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def build_input_row(
    naics_desc,
    business_type,
    business_age,
    term_months,
    gross_approval,
    columns,
):
    row = pd.DataFrame([{col: 0 for col in columns}])

    if "terminmonths" in columns:
        row["terminmonths"] = term_months
    if "grossapproval" in columns:
        row["grossapproval"] = gross_approval

    naics_col = f"naicsdescription_{naics_desc}"
    if naics_col in row.columns:
        row[naics_col] = 1

    btype_col = f"businesstype_{business_type}"
    if btype_col in row.columns:
        row[btype_col] = 1

    bage_col = f"businessage_{business_age}"
    if bage_col in row.columns:
        row[bage_col] = 1

    return row


def risk_level(prob):
    if prob >= 0.15:
        return "High", "risk-high"
    elif prob >= 0.08:
        return "Medium", "risk-medium"
    else:
        return "Low", "risk-low"


def page_header(label, title, subtitle):
    st.markdown(f"""
    <div class="page-header">
        <div class="eyebrow">{label}</div>
        <h1 style="margin:0;">{title}</h1>
        <div class="subtitle">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### RGV Business Survival Predictor")
    st.caption("Group 12 · Los Datos")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Prediction Tool", "EDA & Insights", "About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("UTRGV Intro to Data Science")


# ─────────────────────────────────────────────
# PAGE 1 — PREDICTION TOOL
# ─────────────────────────────────────────────
if page == "Prediction Tool":

    page_header(
        "Prediction Tool",
        "SBA Loan Default Risk Predictor",
        "Estimate the probability that a Texas SBA 7(a) loan may default based on loan details and business characteristics."
    )

    model, columns, model_dir = load_model()

    if model is None:
        st.error(
            "Model files not found. Make sure `xgb_model.pkl` and `feature_columns.pkl` are in a `models/` folder at the root of the repository."
        )
        st.stop()

    st.markdown("### Loan Details")

    col1, col2 = st.columns(2)

    with col1:
        naics_desc = st.selectbox(
            "Industry",
            NAICS_OPTIONS,
            index=10,
            help="Select the primary industry of the business.",
        )
        business_type = st.selectbox(
            "Business Type",
            BUSINESS_TYPES,
            index=0,
            help="Legal entity type of the business.",
        )
        business_age = st.selectbox(
            "Business Age at Loan Origination",
            BUSINESS_AGES,
            index=0,
            help="Whether the business was new or established when the loan was issued.",
        )

    with col2:
        term_months = st.slider(
            "Loan Term (months)",
            min_value=12,
            max_value=360,
            value=120,
            step=12,
            help="Duration of the SBA loan in months.",
        )
        gross_approval = st.number_input(
            "Gross Approval Amount ($)",
            min_value=5_000,
            max_value=5_000_000,
            value=150_000,
            step=5_000,
            format="%d",
            help="Total approved loan amount in USD.",
        )

    if st.button("Predict Default Risk", use_container_width=True, type="primary"):

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
        level, css_class = risk_level(prob_default)

        st.markdown("---")
        st.markdown("### Prediction Result")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.markdown(f"""
            <div class="metric-card {css_class}">
                <div class="metric-label">Default Risk Level</div>
                <div class="big-number">{level}</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Probability of Default</div>
                <div class="big-number" style="color:var(--danger);">{prob_default:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        with res_col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Probability of Survival</div>
                <div class="big-number" style="color:var(--success);">{prob_survive:.1%}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Main Factors Behind This Prediction")

        factors = []

        if term_months <= 60:
            factors.append(f"Short loan term ({term_months} months): loans under 5 years default at about 26.5% in this dataset.")
        elif term_months >= 240:
            factors.append(f"Long loan term ({term_months} months): loans over 20 years default at about 0.7% in this dataset.")

        if business_age == "New Business or 2 years or less":
            factors.append("New business: businesses 2 years or younger default at about 17.1%, compared with 9.7% for established businesses.")
        elif business_age == "Existing or more than 2 years old":
            factors.append("Established business: established businesses show lower default risk than new businesses in this dataset.")

        if business_type == "Individual":
            factors.append("Individual owner: individual-owned businesses default at about 16.9% in this dataset.")
        elif business_type == "Corporation":
            factors.append("Corporation: corporations default at about 11.7%, lower than individual-owned businesses.")

        if naics_desc == "General Freight Trucking, Long Distance, Truckload":
            factors.append("Trucking industry: long-distance truckload businesses show elevated default risk in this dataset.")

        if not factors:
            factors.append("No extreme risk factors were detected for this combination.")

        for f in factors:
            st.markdown(f"<div class='info-card'>{f}</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
            <strong>Disclaimer:</strong> This tool predicts SBA loan default risk as a proxy for small business financial survival. Loan default does not always mean business closure. This model is trained on Texas SBA 7(a) loans from FY 2018-2022 and is for educational purposes only. It should not be used for actual lending or business decisions. Model accuracy: 94% overall. ROC-AUC: 0.964.
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 2 — EDA & INSIGHTS
# ─────────────────────────────────────────────
elif page == "EDA & Insights":

    page_header(
        "EDA & Insights",
        "What Drives SBA Loan Default Risk?",
        "A focused view of the most important patterns from about 8,000 Texas SBA 7(a) loans from FY 2018-2022. Overall default rate: 12.0%."
    )

    st.markdown("### Key Findings")

    kf1, kf2, kf3, kf4 = st.columns(4)
    with kf1:
        st.metric("RGV Default Rate", "13.7%", delta="+1.8% vs Texas avg", delta_color="inverse")
    with kf2:
        st.metric("Startup Default Rate", "17.1%", delta="+7.4% vs established", delta_color="inverse")
    with kf3:
        st.metric("Short-Term Default Rate", "26.5%")
    with kf4:
        st.metric("Long-Term Default Rate", "0.7%")

    st.markdown("---")

    charts = find_charts()

    if not charts:
        st.warning(
            "Chart images were not found. Add your saved PNG files to a `charts/` folder at the root of the repository."
        )

        st.markdown("### Insights Summary")

        insights = {
            "Industry Risk": [
                "Appliance repair and trucking showed some of the highest default rates.",
                "Restaurants and food-service businesses also showed elevated risk.",
            ],
            "Loan Term": [
                "Loans from 0-5 years had the highest default rate at about 26.5%.",
                "Loans from 20-25 years had the lowest default rate at about 0.7%.",
            ],
            "Business Type and Age": [
                "New businesses were riskier than established businesses.",
                "Individual-owned businesses showed higher default risk than corporations and partnerships.",
            ],
            "RGV vs Texas": [
                "The RGV default rate was 13.7%, compared with 11.9% for the rest of Texas.",
                "Income, poverty, and unemployment were weakly correlated with default risk in this dataset.",
            ],
        }

        for title, bullets in insights.items():
            with st.expander(title, expanded=False):
                for b in bullets:
                    st.markdown(f"- {b}")

    else:
        chart_labels = [
            "Default Rate by Industry",
            "Default Rate by Year",
            "Loan Amount vs Default",
            "RGV vs Rest of Texas",
            "Income vs Default by ZIP Code",
            "Default by Business Type",
            "Default by Business Age",
            "Default by Loan Term",
        ]

        selected_charts = charts[:8]

        st.markdown("### Most Important Visuals")
        st.caption("Showing only the strongest visuals for the main takeaway.")

        for i in range(0, len(selected_charts), 2):
            col_a, col_b = st.columns(2)
            with col_a:
                label = chart_labels[i] if i < len(chart_labels) else os.path.basename(selected_charts[i])
                st.markdown(f"<div class='chart-title'>{label}</div>", unsafe_allow_html=True)
                st.image(selected_charts[i], use_container_width=True)
            if i + 1 < len(selected_charts):
                with col_b:
                    label = chart_labels[i + 1] if i + 1 < len(chart_labels) else os.path.basename(selected_charts[i + 1])
                    st.markdown(f"<div class='chart-title'>{label}</div>", unsafe_allow_html=True)
                    st.image(selected_charts[i + 1], use_container_width=True)

    st.markdown("---")
    st.markdown("### Model Performance")

    pm1, pm2, pm3, pm4 = st.columns(4)
    with pm1:
        st.metric("Model", "XGBoost")
    with pm2:
        st.metric("Overall Accuracy", "94%")
    with pm3:
        st.metric("ROC-AUC Score", "0.964")
    with pm4:
        st.metric("Default Recall", "81%")

    st.markdown("""
    <div class="soft-card">
        XGBoost performed best for this project because it identified a much larger share of actual defaults than the baseline models. Default recall matters most here because the goal is to flag higher-risk loans, not just predict the majority class.
    </div>
    """, unsafe_allow_html=True)

    _, _, model_dir = load_model()

    if model_dir:
        conf_path = os.path.join(model_dir, "confusion_matrix.png")
        shap_path = os.path.join(model_dir, "shap_summary.png")

        mc1, mc2 = st.columns(2)
        with mc1:
            if os.path.exists(conf_path):
                st.markdown("<div class='chart-title'>Confusion Matrix</div>", unsafe_allow_html=True)
                st.image(conf_path, use_container_width=True)
        with mc2:
            if os.path.exists(shap_path):
                st.markdown("<div class='chart-title'>SHAP Summary Plot</div>", unsafe_allow_html=True)
                st.image(shap_path, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE 3 — ABOUT
# ─────────────────────────────────────────────
elif page == "About":

    page_header(
        "About",
        "About This Project",
        "A machine learning project built to study small business loan default risk in Texas, with a focus on the Rio Grande Valley."
    )

    st.markdown("""
    ### Project Overview

    This project was built for the UTRGV Intro to Data Science class as a final project. Group 12, Los Datos, used machine learning to predict whether an SBA 7(a) small business loan would default. Loan default is used as a proxy for business financial distress.

    ### Team

    - Damian Ramirez
    - Bukola
    - Roen

    ### What We Built

    | Step | Description |
    |------|-------------|
    | Data Collection | SBA 7(a) FOIA loan data, US Census ACS, County Business Patterns, and Business Dynamics Statistics |
    | Data Cleaning | Filtered to Texas, FY 2018-2022, with about 8,000 loans |
    | EDA | Visualized default patterns by industry, time, geography, business type, business age, and loan term |
    | Statistical Analysis | Estimated default rates, tested differences across groups, and explored correlations and feature relationships |
    | Modeling | Built an XGBoost classifier with 94% accuracy, 0.964 ROC-AUC, and 81% recall on defaults |
    | Explainability | Used SHAP values to interpret model predictions |
    | Deployment | Built this Streamlit app |

    ### Data Sources

    - SBA 7(a) Loan Data
    - US Census ACS
    - County Business Patterns / ZBP
    - Business Dynamics Statistics

    ### Important Disclaimer

    Loan default does not always mean business closure. This model predicts whether an SBA loan was charged off, which is used as a proxy for financial distress. A business may default without closing, and a business may close without defaulting on an SBA loan.

    This tool is for educational purposes only and should not be used for real lending, investment, or business decisions. The model was trained on Texas data from FY 2018-2022 and may not generalize to other states, time periods, or loan types.

    ### Technical Details

    - Framework: Python, scikit-learn, XGBoost, Pandas
    - App: Streamlit
    - Training data: 8,064 rows by 608 features after one-hot encoding
    - Class imbalance handling: `scale_pos_weight`
    - Main predictive features: loan term, business age, interest rate, industry, and approval amount
    """)
