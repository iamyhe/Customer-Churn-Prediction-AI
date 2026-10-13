"""
churn_dashboard.py -- Cyberpunk / Neon AI SaaS Customer Churn Dashboard
=========================================================================
A premium Streamlit dashboard with Dark (Cyberpunk/Neon) and Light
(Professional) themes. Loads the trained Random Forest model and
visualises churn predictions with dynamic styling.

Usage:
    streamlit run churn_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, auc
import joblib

# =========================================================================
# PAGE CONFIG
# =========================================================================
st.set_page_config(
    page_title="ChurnGuard AI | Predictive Analytics",
    page_icon="https://em-content.zobj.net/source/twitter/408/shield_1f6e1-fe0f.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================================
# THEME SYSTEM
# =========================================================================
# The toggle MUST be created before any CSS / chart rendering so the
# entire page can react to the selection in a single Streamlit pass.
# We use session_state to persist the choice across reruns.

if "theme" not in st.session_state:
    st.session_state.theme = "Dark"


def get_theme_tokens(mode: str) -> dict:
    """Return a complete set of design tokens for the chosen theme."""
    if mode == "Dark":
        return dict(
            # -- backgrounds
            bg_primary="#0B0F19",
            bg_card="#151C2C",
            bg_card_hover="#1A2235",
            bg_header="rgba(11, 15, 25, 0.85)",
            bg_sidebar_top="#0D1220",
            bg_sidebar_bot="#111827",
            bg_input="#1A2235",
            bg_table_header="#1E293B",
            # -- accents
            accent_purple="#A855F7",
            accent_cyan="#06B6D4",
            accent_pink="#EC4899",
            accent_green="#22D3EE",
            # -- text
            text_primary="#F1F5F9",
            text_muted="#94A3B8",
            text_chart="white",
            text_footer="#475569",
            # -- borders / glows
            border_subtle="rgba(168, 85, 247, 0.15)",
            glow_purple="0 0 15px rgba(168, 85, 247, 0.25)",
            glow_cyan="0 0 15px rgba(6, 182, 212, 0.25)",
            glow_hover="0 0 25px rgba(168, 85, 247, 0.35)",
            grid_color="rgba(148,163,184,0.08)",
            border_input="rgba(6, 182, 212, 0.3)",
            border_table_row="rgba(148, 163, 184, 0.1)",
            # -- plotly
            plotly_template="plotly_dark",
            fill_roc="rgba(168, 85, 247, 0.12)",
            fill_trend="rgba(6, 182, 212, 0.10)",
            # -- title animation
            title_glow_a="0 0 10px rgba(168,85,247,0.5), 0 0 20px rgba(6,182,212,0.3)",
            title_glow_b="0 0 20px rgba(168,85,247,0.8), 0 0 40px rgba(6,182,212,0.5)",
            section_border="rgba(6, 182, 212, 0.2)",
            # -- button
            btn_gradient="linear-gradient(135deg, #A855F7 0%, #06B6D4 100%)",
            btn_hover_glow=("0 0 20px rgba(168, 85, 247, 0.4), "
                            "0 0 20px rgba(6, 182, 212, 0.4)"),
            # -- scrollbar
            scroll_track="#0B0F19",
            scroll_thumb="linear-gradient(180deg, #A855F7, #06B6D4)",
        )
    else:  # Light
        return dict(
            bg_primary="#F0F2F6",
            bg_card="#FFFFFF",
            bg_card_hover="#F8FAFC",
            bg_header="rgba(240, 242, 246, 0.92)",
            bg_sidebar_top="#FFFFFF",
            bg_sidebar_bot="#F8FAFC",
            bg_input="#F1F5F9",
            bg_table_header="#E2E8F0",
            accent_purple="#7C3AED",
            accent_cyan="#0891B2",
            accent_pink="#DB2777",
            accent_green="#0E7490",
            text_primary="#1E293B",
            text_muted="#64748B",
            text_chart="#334155",
            text_footer="#94A3B8",
            border_subtle="rgba(100, 116, 139, 0.15)",
            glow_purple="0 2px 8px rgba(124, 58, 237, 0.08)",
            glow_cyan="0 2px 8px rgba(8, 145, 178, 0.08)",
            glow_hover="0 4px 16px rgba(124, 58, 237, 0.14)",
            grid_color="rgba(100,116,139,0.10)",
            border_input="rgba(8, 145, 178, 0.25)",
            border_table_row="rgba(100, 116, 139, 0.1)",
            plotly_template="plotly_white",
            fill_roc="rgba(124, 58, 237, 0.08)",
            fill_trend="rgba(8, 145, 178, 0.08)",
            title_glow_a="none",
            title_glow_b="none",
            section_border="rgba(8, 145, 178, 0.18)",
            btn_gradient="linear-gradient(135deg, #7C3AED 0%, #0891B2 100%)",
            btn_hover_glow=("0 4px 16px rgba(124, 58, 237, 0.25), "
                            "0 4px 16px rgba(8, 145, 178, 0.25)"),
            scroll_track="#F0F2F6",
            scroll_thumb="linear-gradient(180deg, #7C3AED, #0891B2)",
        )


def inject_css(t: dict):
    """Inject a full CSS stylesheet derived from the active theme tokens."""
    st.markdown(f"""
<style>
/* ---------- Google Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ========== TRANSITION (smooth theme switch) ========== */
*, *::before, *::after {{
    transition: background-color 0.35s ease,
                color 0.25s ease,
                border-color 0.3s ease,
                box-shadow 0.3s ease;
}}

/* ---------- Global Overrides ---------- */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
.main, .block-container {{
    background-color: {t["bg_primary"]} !important;
    color: {t["text_primary"]} !important;
    font-family: 'Inter', sans-serif !important;
}}
[data-testid="stHeader"] {{
    background-color: {t["bg_header"]} !important;
    backdrop-filter: blur(12px);
}}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {t["bg_sidebar_top"]} 0%, {t["bg_sidebar_bot"]} 100%) !important;
    border-right: 1px solid {t["border_subtle"]} !important;
}}
[data-testid="stSidebar"] * {{
    color: {t["text_primary"]} !important;
}}

/* ---------- KPI Metric Cards ---------- */
div[data-testid="stMetric"] {{
    background: {t["bg_card"]};
    border: 1px solid {t["border_subtle"]};
    border-radius: 16px;
    padding: 20px 24px;
    box-shadow: {t["glow_purple"]};
    transition: all 0.3s ease;
}}
div[data-testid="stMetric"]:hover {{
    background: {t["bg_card_hover"]};
    border-color: {t["accent_purple"]};
    box-shadow: {t["glow_hover"]};
    transform: translateY(-2px);
}}
div[data-testid="stMetric"] label {{
    color: {t["text_muted"]} !important;
    font-weight: 500 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: {t["text_primary"]} !important;
    font-weight: 800 !important;
    font-size: 1.9rem !important;
}}

/* ---------- Section Headers ---------- */
h1, h2, h3 {{
    color: {t["text_primary"]} !important;
    font-family: 'Inter', sans-serif !important;
}}

/* ---------- Chart Containers ---------- */
[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"],
.stDataFrame {{
    background: {t["bg_card"]} !important;
    border: 1px solid {t["border_subtle"]} !important;
    border-radius: 16px !important;
    padding: 8px !important;
    box-shadow: {t["glow_cyan"]};
}}

/* ---------- Dataframe Styling ---------- */
.stDataFrame table {{
    border-collapse: collapse !important;
}}
.stDataFrame th {{
    background: {t["bg_table_header"]} !important;
    color: {t["accent_cyan"]} !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    font-size: 0.75rem !important;
}}
.stDataFrame td {{
    color: {t["text_primary"]} !important;
    border-bottom: 1px solid {t["border_table_row"]} !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}}

/* ---------- Sidebar Text Input ---------- */
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {{
    background: {t["bg_input"]} !important;
    border: 1px solid {t["border_input"]} !important;
    border-radius: 10px !important;
    color: {t["text_primary"]} !important;
    font-family: 'JetBrains Mono', monospace !important;
}}
[data-testid="stSidebar"] input:focus,
[data-testid="stSidebar"] textarea:focus {{
    border-color: {t["accent_cyan"]} !important;
    box-shadow: {t["glow_cyan"]} !important;
}}

/* ---------- Buttons ---------- */
.stButton > button {{
    background: {t["btn_gradient"]} !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.3s ease !important;
}}
.stButton > button:hover {{
    box-shadow: {t["btn_hover_glow"]} !important;
    transform: translateY(-1px) !important;
}}

/* ---------- Selectbox / Multiselect ---------- */
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stMultiSelect > div > div {{
    background: {t["bg_input"]} !important;
    border: 1px solid {t["border_input"]} !important;
    border-radius: 10px !important;
}}

/* ---------- Divider ---------- */
hr {{
    border-color: {t["border_subtle"]} !important;
}}

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {t["scroll_track"]}; }}
::-webkit-scrollbar-thumb {{
    background: {t["scroll_thumb"]};
    border-radius: 3px;
}}

/* ---------- Hide Streamlit Branding ---------- */
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}

/* ---------- Neon Title Animation ---------- */
@keyframes neonPulse {{
    0%, 100% {{ text-shadow: {t["title_glow_a"]}; }}
    50%      {{ text-shadow: {t["title_glow_b"]}; }}
}}
.neon-title {{
    animation: neonPulse 3s ease-in-out infinite;
    font-size: 2.1rem !important;
    font-weight: 900 !important;
    background: linear-gradient(135deg, {t["accent_purple"]} 0%, {t["accent_cyan"]} 50%, {t["accent_pink"]} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}}
.subtitle {{
    color: {t["text_muted"]} !important;
    font-size: 0.92rem;
    font-weight: 400;
    margin-top: -10px;
    letter-spacing: 0.3px;
}}
.section-header {{
    color: {t["accent_cyan"]} !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 2px solid {t["section_border"]};
}}

/* ---------- Toggle Pill Styling ---------- */
.theme-toggle-label {{
    color: {t["text_muted"]};
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 2px;
}}
</style>
""", unsafe_allow_html=True)


# =========================================================================
# DATA & MODEL LOADING (cached for performance)
# =========================================================================

@st.cache_data
def load_raw_data():
    """Load the original CSV and keep customerID for the dashboard table."""
    df = pd.read_csv("churn_data.csv")
    return df


@st.cache_resource
def load_model_artifacts():
    """Load the persisted Random Forest model and its expected feature list."""
    model = joblib.load("churn_model.pkl")
    feature_cols = joblib.load("model_features.pkl")
    return model, feature_cols


def preprocess_for_prediction(df, feature_cols):
    """
    Reproduce *exactly* the same preprocessing used during training so
    that the feature matrix aligns with what the model expects.
    """
    data = df.copy()

    # Clean TotalCharges (same logic as train_model.py)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data["TotalCharges"].fillna(0.0, inplace=True)

    # Identify column groups (excluding customerID and Churn)
    cols_to_encode = data.select_dtypes(include=["object"]).columns.tolist()
    for skip in ["customerID", "Churn"]:
        if skip in cols_to_encode:
            cols_to_encode.remove(skip)

    binary_cols = [c for c in cols_to_encode if data[c].nunique() == 2]
    multi_cols = [c for c in cols_to_encode if data[c].nunique() > 2]

    # Label-encode binary columns
    for col in binary_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])

    # One-hot-encode multi-class columns
    data = pd.get_dummies(data, columns=multi_cols, drop_first=True)

    # Drop non-feature columns
    for col in ["customerID", "Churn"]:
        if col in data.columns:
            data.drop(col, axis=1, inplace=True)

    # Align columns exactly with training schema
    for col in feature_cols:
        if col not in data.columns:
            data[col] = 0
    data = data[feature_cols]

    return data


# -------------------------------------------------------------------------
# Load everything
# -------------------------------------------------------------------------
raw_df = load_raw_data()
model, feature_cols = load_model_artifacts()

features_df = preprocess_for_prediction(raw_df, feature_cols)
predictions = model.predict(features_df)
probabilities = model.predict_proba(features_df)[:, 1]  # P(Churn)

# Attach predictions back to raw data for display
display_df = raw_df.copy()
display_df["Churn_Prediction"] = np.where(predictions == 1, "At-Risk", "Safe")
display_df["Churn_Probability"] = np.round(probabilities * 100, 1)

# Convert actual Churn to numeric for ROC
actual_churn_numeric = (raw_df["Churn"] == "Yes").astype(int)

# Compute model accuracy on *entire* dataset (for the KPI card)
model_accuracy = (predictions == actual_churn_numeric).mean()

# =========================================================================
# SIDEBAR  (theme toggle lives here, at the very top)
# =========================================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px 0;'>
        <span style='font-size:2.2rem;'>&#128737;</span>
        <h2 style='margin:0; background: linear-gradient(135deg, #A855F7, #06B6D4);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-weight:900; letter-spacing:-0.3px;'>ChurnGuard AI</h2>
        <p style='color:#94A3B8; font-size:0.78rem; margin-top:2px;
            letter-spacing:1px; text-transform:uppercase;'>Predictive Analytics Engine</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ---- Theme Toggle ----
    st.markdown("<p class='theme-toggle-label'>Appearance</p>",
                unsafe_allow_html=True)
    is_dark = st.toggle(
        "Dark Mode",
        value=(st.session_state.theme == "Dark"),
        key="theme_toggle",
    )
    st.session_state.theme = "Dark" if is_dark else "Light"

    st.markdown("---")

    # Navigation
    st.markdown("<p style='color:#06B6D4; font-weight:700; font-size:0.75rem; "
                "text-transform:uppercase; letter-spacing:1.5px; margin-bottom:4px;'>"
                "Navigation</p>", unsafe_allow_html=True)
    page = st.selectbox(
        "View",
        ["Dashboard", "Customer Explorer", "Model Insights"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Risk threshold slider
    st.markdown("<p style='color:#06B6D4; font-weight:700; font-size:0.75rem; "
                "text-transform:uppercase; letter-spacing:1.5px; margin-bottom:4px;'>"
                "Risk Threshold</p>", unsafe_allow_html=True)
    risk_threshold = st.slider(
        "Churn probability threshold (%)",
        min_value=30, max_value=90, value=50, step=5,
        label_visibility="collapsed",
    )

    st.markdown("---")

    # Contract filter
    st.markdown("<p style='color:#06B6D4; font-weight:700; font-size:0.75rem; "
                "text-transform:uppercase; letter-spacing:1.5px; margin-bottom:4px;'>"
                "Filter by Contract</p>", unsafe_allow_html=True)
    contract_filter = st.multiselect(
        "Contract",
        options=raw_df["Contract"].unique().tolist(),
        default=raw_df["Contract"].unique().tolist(),
        label_visibility="collapsed",
    )

    st.markdown("---")

    # AI Assistant
    st.markdown("<p style='color:#A855F7; font-weight:700; font-size:0.75rem; "
                "text-transform:uppercase; letter-spacing:1.5px; margin-bottom:4px;'>"
                "&#9889; AI Assistant</p>", unsafe_allow_html=True)
    user_query = st.text_area(
        "Ask about churn patterns...",
        placeholder="e.g. Which contract type has the highest churn?",
        height=80,
        label_visibility="collapsed",
    )
    if st.button("Analyze", use_container_width=True):
        if user_query.strip():
            _q = user_query.lower()

            # --- Contract-related questions ---
            contract_keywords = ["contract", "month-to-month", "monthly",
                                 "one year", "two year", "plan", "upgrade"]
            # --- Feature / risk-factor questions ---
            feature_keywords = ["feature", "factor", "risk", "driver",
                                "important", "importance", "top", "why",
                                "cause", "reason", "predict"]

            if any(kw in _q for kw in contract_keywords):
                st.success(
                    "**Contract Analysis**\n\n"
                    "Based on the data, **Month-to-month** contracts have the "
                    "highest churn rate. We recommend targeting these customers "
                    "with **1-year upgrade offers** and loyalty incentives to "
                    "reduce attrition."
                )
            elif any(kw in _q for kw in feature_keywords):
                # Pull the real top-3 features from the trained model
                _imp = pd.Series(model.feature_importances_, index=feature_cols)
                _top3 = _imp.nlargest(3).index.tolist()
                _top3_str = ", ".join(f"**{f}**" for f in _top3)
                st.success(
                    "**Top Risk Factors**\n\n"
                    f"The top factors driving churn are {_top3_str}. "
                    "Customers with short tenure, high monthly charges, and "
                    "low total lifetime value are significantly more likely "
                    "to churn."
                )
            else:
                st.info(
                    "I am currently optimized to answer questions about "
                    "**contract types** and **churn risk factors** based on "
                    "this dataset. Try asking:\n"
                    '- *"Which contract type has the highest churn?"*\n'
                    '- *"What are the top risk factors?"*'
                )
        else:
            st.warning("Please type a question above before clicking Analyze.")

    # Footer
    st.markdown("""
    <div style='position:fixed; bottom:16px; left:16px; right:16px; text-align:center;'>
        <p style='color:#475569; font-size:0.68rem; letter-spacing:0.5px;'>
            ChurnGuard AI v2.0 &middot; Built with Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================================
# APPLY THEME  (must happen after toggle is created)
# =========================================================================
T = get_theme_tokens(st.session_state.theme)
inject_css(T)

# Set Plotly template for all charts rendered below
pio.templates.default = T["plotly_template"]

# Convenience aliases used by every chart
PURPLE  = T["accent_purple"]
CYAN    = T["accent_cyan"]
PINK    = T["accent_pink"]
GREEN   = T["accent_green"]
CARD_BG = T["bg_card"]
MUTED   = T["text_muted"]
TXT     = T["text_chart"]
GRID    = T["grid_color"]

# =========================================================================
# APPLY DATA FILTERS
# =========================================================================
filtered_df = display_df[display_df["Contract"].isin(contract_filter)].copy()
at_risk_df = filtered_df[filtered_df["Churn_Probability"] >= risk_threshold]

# =========================================================================
# HEADER
# =========================================================================
st.markdown('<p class="neon-title">ChurnGuard AI Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Real-time customer churn prediction powered by '
            'machine learning &mdash; identify at-risk accounts before they leave.</p>',
            unsafe_allow_html=True)
st.markdown("")

# =========================================================================
# TOP KPI ROW
# =========================================================================
total_customers = len(filtered_df)
at_risk_count = len(at_risk_df)
revenue_at_risk = at_risk_df["MonthlyCharges"].sum()
churn_rate = (at_risk_count / total_customers * 100) if total_customers else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers", f"{total_customers:,}")
k2.metric("At-Risk Customers", f"{at_risk_count:,}", delta=f"{churn_rate:.1f}% of base",
          delta_color="inverse")
k3.metric("Revenue at Risk", f"${revenue_at_risk:,.0f}/mo")
k4.metric("Model Accuracy", f"{model_accuracy * 100:.1f}%")

st.markdown("")

# =========================================================================
# ROW 1 -- Donut Chart + Churn by Contract Bar Chart
# =========================================================================
st.markdown('<p class="section-header">Risk Distribution & Contract Analysis</p>',
            unsafe_allow_html=True)

col_donut, col_bar = st.columns([1, 1.4])

# ---- Donut Chart ----
with col_donut:
    safe_count = total_customers - at_risk_count
    fig_donut = go.Figure(go.Pie(
        labels=["Safe", "At-Risk"],
        values=[safe_count, at_risk_count],
        hole=0.65,
        marker=dict(colors=[CYAN, PURPLE],
                    line=dict(color=CARD_BG, width=3)),
        textinfo="percent+label",
        textfont=dict(size=13, family="Inter", color=TXT),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    fig_donut.update_layout(
        title=dict(text="Customer Risk Split", font=dict(size=15, color=MUTED)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=370,
        margin=dict(t=50, b=20, l=20, r=20),
        annotations=[dict(
            text=f"<b>{at_risk_count}</b><br><span style='font-size:11px;color:{MUTED}'>at risk</span>",
            x=0.5, y=0.5, font=dict(size=28, color=PURPLE, family="Inter"),
            showarrow=False,
        )],
    )
    st.plotly_chart(fig_donut, use_container_width=True)

# ---- Churn by Contract Type (Grouped Bar) ----
with col_bar:
    contract_churn = (
        filtered_df.groupby(["Contract", "Churn_Prediction"])
        .size()
        .reset_index(name="Count")
    )
    colour_map = {"Safe": CYAN, "At-Risk": PURPLE}
    fig_bar = px.bar(
        contract_churn,
        x="Contract",
        y="Count",
        color="Churn_Prediction",
        barmode="group",
        color_discrete_map=colour_map,
        text="Count",
    )
    fig_bar.update_traces(
        textposition="outside",
        textfont=dict(size=11, color=TXT),
        marker_line_width=0,
    )
    fig_bar.update_layout(
        title=dict(text="Churn Risk by Contract Type", font=dict(size=15, color=MUTED)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="", tickfont=dict(color=MUTED)),
        yaxis=dict(title="Customers", gridcolor=GRID,
                   tickfont=dict(color=MUTED)),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(color=MUTED)),
        height=370,
        margin=dict(t=60, b=30, l=40, r=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("")

# =========================================================================
# ROW 2 -- ROC-AUC Curve + Revenue Risk by Tenure
# =========================================================================
st.markdown('<p class="section-header">Model Performance & Revenue Risk Trend</p>',
            unsafe_allow_html=True)

col_roc, col_trend = st.columns(2)

# ---- ROC-AUC Curve ----
with col_roc:
    fpr, tpr, _ = roc_curve(actual_churn_numeric, probabilities)
    roc_auc = auc(fpr, tpr)

    fig_roc = go.Figure()
    # Diagonal reference
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(dash="dot", color=MUTED, width=1),
        showlegend=False,
    ))
    # ROC curve with gradient fill
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines",
        name=f"AUC = {roc_auc:.3f}",
        line=dict(color=PURPLE, width=2.5),
        fill="tozeroy",
        fillcolor=T["fill_roc"],
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
    ))
    fig_roc.update_layout(
        title=dict(text=f"ROC-AUC Curve  (AUC = {roc_auc:.3f})",
                   font=dict(size=15, color=MUTED)),
        xaxis=dict(title=dict(text="False Positive Rate", font=dict(color=MUTED)),
                   gridcolor=GRID, tickfont=dict(color=MUTED)),
        yaxis=dict(title=dict(text="True Positive Rate", font=dict(color=MUTED)),
                   gridcolor=GRID, tickfont=dict(color=MUTED)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color=CYAN, size=13), x=0.55, y=0.1,
                    bgcolor="rgba(0,0,0,0)"),
        height=380,
        margin=dict(t=50, b=50, l=50, r=30),
    )
    st.plotly_chart(fig_roc, use_container_width=True)

# ---- Revenue Risk by Tenure Bins ----
with col_trend:
    # Bucket tenure into bands
    tenure_bins = [0, 6, 12, 24, 36, 48, 60, 72]
    tenure_labels = ["0-6", "7-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
    filtered_df["Tenure_Band"] = pd.cut(
        filtered_df["tenure"], bins=tenure_bins, labels=tenure_labels, right=True
    )
    rev_trend = (
        filtered_df[filtered_df["Churn_Prediction"] == "At-Risk"]
        .groupby("Tenure_Band", observed=False)["MonthlyCharges"]
        .sum()
        .reset_index()
    )
    rev_trend.columns = ["Tenure (months)", "Revenue at Risk ($)"]

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(
        x=rev_trend["Tenure (months)"],
        y=rev_trend["Revenue at Risk ($)"],
        mode="lines+markers",
        line=dict(color=CYAN, width=2.5, shape="spline"),
        marker=dict(size=8, color=CYAN,
                    line=dict(width=2, color=CARD_BG)),
        fill="tozeroy",
        fillcolor=T["fill_trend"],
        hovertemplate="Tenure: %{x}<br>Revenue at Risk: $%{y:,.0f}<extra></extra>",
    ))
    fig_trend.update_layout(
        title=dict(text="Revenue at Risk by Customer Tenure",
                   font=dict(size=15, color=MUTED)),
        xaxis=dict(title=dict(text="Tenure Band (months)", font=dict(color=MUTED)),
                   gridcolor=GRID, tickfont=dict(color=MUTED)),
        yaxis=dict(title=dict(text="Monthly Revenue ($)", font=dict(color=MUTED)),
                   gridcolor=GRID, tickfont=dict(color=MUTED)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(t=50, b=50, l=50, r=30),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("")

# =========================================================================
# ROW 3 -- Feature Importance + Churn by Internet Service
# =========================================================================
st.markdown('<p class="section-header">Feature Importance & Service Insights</p>',
            unsafe_allow_html=True)

col_fi, col_inet = st.columns([1.4, 1])

# ---- Feature Importance (Horizontal Bar) ----
with col_fi:
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top_n = importances.nlargest(12).sort_values()

    # Gradient colours from cyan to purple for each bar
    n_bars = len(top_n)
    bar_colors = [
        f"rgba({int(6 + (168-6)*i/(n_bars-1))}, "
        f"{int(182 + (85-182)*i/(n_bars-1))}, "
        f"{int(212 + (247-212)*i/(n_bars-1))}, 0.85)"
        for i in range(n_bars)
    ]

    fig_fi = go.Figure(go.Bar(
        x=top_n.values,
        y=top_n.index,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"{v:.3f}" for v in top_n.values],
        textposition="outside",
        textfont=dict(size=10, color=MUTED),
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
    ))
    fig_fi.update_layout(
        title=dict(text="Top 12 Feature Importances (Random Forest)",
                   font=dict(size=15, color=MUTED)),
        xaxis=dict(title=dict(text="Importance", font=dict(color=MUTED)),
                   gridcolor=GRID, tickfont=dict(color=MUTED)),
        yaxis=dict(tickfont=dict(size=11, color=MUTED)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=420,
        margin=dict(t=50, b=40, l=180, r=60),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

# ---- Churn by Internet Service ----
with col_inet:
    inet_churn = (
        filtered_df.groupby(["InternetService", "Churn_Prediction"])
        .size()
        .reset_index(name="Count")
    )
    fig_inet = px.bar(
        inet_churn, x="InternetService", y="Count",
        color="Churn_Prediction", barmode="stack",
        color_discrete_map=colour_map,
        text="Count",
    )
    fig_inet.update_traces(
        textposition="inside",
        textfont=dict(size=11, color="white"),
        marker_line_width=0,
    )
    fig_inet.update_layout(
        title=dict(text="Risk by Internet Service", font=dict(size=15, color=MUTED)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="", tickfont=dict(color=MUTED)),
        yaxis=dict(title="Customers", gridcolor=GRID,
                   tickfont=dict(color=MUTED)),
        legend=dict(title="", orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(color=MUTED)),
        height=420,
        margin=dict(t=60, b=30, l=40, r=20),
    )
    st.plotly_chart(fig_inet, use_container_width=True)

st.markdown("")

# =========================================================================
# ROW 4 -- Top At-Risk Customers Table
# =========================================================================
st.markdown('<p class="section-header">Top At-Risk Customers</p>',
            unsafe_allow_html=True)

# Build the display table
top_risk = (
    filtered_df[filtered_df["Churn_Prediction"] == "At-Risk"]
    .nlargest(25, "Churn_Probability")[
        ["customerID", "Contract", "InternetService", "tenure",
         "MonthlyCharges", "TotalCharges", "Churn_Probability"]
    ]
    .reset_index(drop=True)
)
top_risk.columns = [
    "Customer ID", "Contract", "Internet Service", "Tenure (mo)",
    "Monthly ($)", "Total ($)", "Churn Prob (%)",
]

# Ensure numeric types so Pandas .style.format() can apply float formatters
for num_col in ["Monthly ($)", "Total ($)", "Churn Prob (%)", "Tenure (mo)"]:
    top_risk[num_col] = pd.to_numeric(top_risk[num_col], errors="coerce").fillna(0)

top_risk.index = top_risk.index + 1  # 1-based rank

st.dataframe(
    top_risk.style
        .format({
            "Monthly ($)": "${:,.2f}",
            "Total ($)": "${:,.2f}",
            "Churn Prob (%)": "{:.1f}%",
        })
        .background_gradient(subset=["Churn Prob (%)"], cmap="RdPu", vmin=50, vmax=100),
    use_container_width=True,
    height=460,
)

st.markdown("")

# =========================================================================
# FOOTER
# =========================================================================
st.markdown(f"""
<div style='text-align:center; padding:24px 0 8px 0;'>
    <span style='color:{T["text_footer"]}; font-size:0.75rem; letter-spacing:0.5px;'>
        ChurnGuard AI &middot; Powered by Random Forest &middot;
        Built with Streamlit &amp; Plotly
    </span>
</div>
""", unsafe_allow_html=True)
