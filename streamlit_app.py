"""
Churn Risk Console — Streamlit edition.

A polished, standalone Streamlit interface for the Customer Churn
Prediction System. Loads the same trained artifacts used by the FastAPI
service (final_churn_model.h5 + preprocessor.pkl) and gives a retention
agent a live read on a customer's churn risk.

Run locally:
    streamlit run streamlit_app.py

Deploy for free on Streamlit Community Cloud:
    1. Push this repo to GitHub (already done).
    2. Go to https://share.streamlit.io -> New app -> pick this repo.
    3. Main file path: streamlit_app.py
    4. Deploy. Make sure final_churn_model.h5 and preprocessor.pkl are
       committed to the repo, since the app loads them at startup.
"""

import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import plotly.graph_objects as go

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Risk Console",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = "final_churn_model.h5"
PREPROCESSOR_PATH = "preprocessor.pkl"

SAFE = "#2DD4BF"
WATCH = "#F5A524"
ALERT = "#F04438"
INK = "#0F1720"
PANEL = "#16212C"
PANEL_LINE = "#24323F"
PAPER = "#E7ECF1"
FOG = "#8493A3"

# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {{
            font-family: 'Inter', sans-serif;
        }}
        .stApp {{
            background-color: {INK};
            background-image:
                radial-gradient(circle at 85% -10%, rgba(45, 212, 191, 0.08), transparent 45%),
                radial-gradient(circle at -5% 110%, rgba(240, 68, 56, 0.06), transparent 45%);
            color: {PAPER};
        }}

        /* Header */
        .crc-title {{
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            font-size: 30px;
            margin-bottom: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .crc-dot {{
            width: 10px; height: 10px; border-radius: 50%;
            background: {SAFE};
            box-shadow: 0 0 10px rgba(45,212,191,0.8);
            display: inline-block;
        }}
        .crc-tagline {{
            color: {FOG}; font-size: 14px; margin-top: 2px;
        }}
        .crc-tag {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 12px; color: {FOG}; text-align: right; line-height: 1.6;
        }}
        .crc-tag b {{ color: {SAFE}; font-weight: 500; }}

        /* Section eyebrow */
        .crc-eyebrow {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase;
            color: {FOG}; margin: 18px 0 6px;
            border-top: 1px dashed {PANEL_LINE};
            padding-top: 14px;
        }}
        .crc-eyebrow.first {{ border-top: none; padding-top: 0; margin-top: 0; }}

        /* Panels */
        .crc-panel {{
            background: {PANEL};
            border: 1px solid {PANEL_LINE};
            border-radius: 10px;
            padding: 20px 22px;
        }}

        .crc-recommend {{
            background: {INK};
            border: 1px solid {PANEL_LINE};
            border-radius: 8px;
            padding: 14px 16px;
            font-size: 14px;
            line-height: 1.55;
            color: {FOG};
            margin-top: 14px;
        }}
        .crc-recommend b {{ color: {PAPER}; }}

        .crc-legend {{
            display: flex; gap: 16px; margin-top: 14px;
            font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: {FOG};
        }}
        .crc-legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
        .crc-dotlegend {{ width: 8px; height: 8px; border-radius: 50%; display: inline-block; }}

        /* Streamlit widget overrides */
        div[data-baseweb="select"] > div, .stNumberInput input {{
            background-color: {INK} !important;
            border-color: {PANEL_LINE} !important;
            color: {PAPER} !important;
        }}
        label, .stMarkdown p {{ color: {FOG} !important; }}
        .stButton button {{
            background-color: {SAFE};
            color: #06201C;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 700;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
        }}
        .stButton button:hover {{
            box-shadow: 0 0 0 3px rgba(45,212,191,0.18);
            color: #06201C;
        }}
        footer {{ visibility: hidden; }}
        #MainMenu {{ visibility: hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Load artifacts (cached — only runs once per server process)
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor


try:
    model, preprocessor = load_artifacts()
    artifacts_ready = True
except Exception as e:
    artifacts_ready = False
    load_error = str(e)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
col_title, col_tag = st.columns([3, 1])
with col_title:
    st.markdown(
        '<div class="crc-title"><span class="crc-dot"></span>Churn Risk Console</div>'
        '<div class="crc-tagline">Read a customer\'s churn risk before you hang up.</div>',
        unsafe_allow_html=True,
    )
with col_tag:
    st.markdown(
        f'<div class="crc-tag">model &middot; <b>final_churn_model.h5</b><br>served via Streamlit</div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

if not artifacts_ready:
    st.error(
        f"Model artifacts not found or failed to load ({load_error}). "
        f"Make sure '{MODEL_PATH}' and '{PREPROCESSOR_PATH}' are in the app directory."
    )
    st.stop()

# ----------------------------------------------------------------------
# Layout: form (left) + signal meter (right)
# ----------------------------------------------------------------------
col_form, col_meter = st.columns([1.15, 0.85], gap="medium")

with col_form:
    st.markdown('<div class="crc-panel">', unsafe_allow_html=True)

    with st.form("churn_form"):
        st.markdown('<div class="crc-eyebrow first">Profile</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        gender = c1.selectbox("Gender", ["Female", "Male"])
        senior = c2.selectbox("Senior citizen", ["No", "Yes"])
        partner = c3.selectbox("Partner", ["No", "Yes"])
        dependents = c4.selectbox("Dependents", ["No", "Yes"])

        st.markdown('<div class="crc-eyebrow">Account</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        tenure = c1.number_input("Tenure (months)", min_value=0, value=2, step=1)
        contract = c2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = c3.selectbox("Paperless billing", ["Yes", "No"])

        st.markdown('<div class="crc-eyebrow">Services</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        phone = c1.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = c2.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = c3.selectbox("Internet service", ["Fiber optic", "DSL", "No"])

        c1, c2, c3 = st.columns(3)
        online_security = c1.selectbox("Online security", ["No", "Yes", "No internet service"])
        online_backup = c2.selectbox("Online backup", ["No", "Yes", "No internet service"])
        device_protection = c3.selectbox("Device protection", ["No", "Yes", "No internet service"])

        c1, c2, c3 = st.columns(3)
        tech_support = c1.selectbox("Tech support", ["No", "Yes", "No internet service"])
        streaming_tv = c2.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = c3.selectbox("Streaming movies", ["Yes", "No", "No internet service"])

        st.markdown('<div class="crc-eyebrow">Billing</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        payment_method = c1.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        monthly_charges = c2.number_input("Monthly charges ($)", min_value=0.0, value=85.70, step=0.10, format="%.2f")
        total_charges = c3.number_input("Total charges ($)", min_value=0.0, value=171.40, step=0.10, format="%.2f")

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Assess risk")

    st.markdown("</div>", unsafe_allow_html=True)

with col_meter:
    st.markdown('<div class="crc-panel">', unsafe_allow_html=True)
    st.markdown('<div class="crc-eyebrow first">Signal meter</div>', unsafe_allow_html=True)

    gauge_slot = st.empty()
    label_slot = st.empty()
    recommend_slot = st.empty()
    legend_html = f"""
        <div class="crc-legend">
            <span><i class="crc-dotlegend" style="background:{SAFE}"></i>Low</span>
            <span><i class="crc-dotlegend" style="background:{WATCH}"></i>Medium</span>
            <span><i class="crc-dotlegend" style="background:{ALERT}"></i>High</span>
        </div>
    """

    def render_gauge(prob=None):
        if prob is None:
            color = FOG
            display_val = 0
            label = "Awaiting input"
        else:
            if prob < 0.3:
                color, label = SAFE, "Low signal"
            elif prob < 0.6:
                color, label = WATCH, "Medium signal"
            else:
                color, label = ALERT, "High signal"
            display_val = round(prob * 100, 1)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=display_val,
            number={"suffix": "%", "font": {"size": 40, "color": color, "family": "Space Grotesk"}},
            gauge={
                "axis": {"range": [0, 100], "visible": False},
                "bar": {"color": color, "thickness": 0.28},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [{"range": [0, 100], "color": PANEL_LINE}],
            },
            domain={"x": [0, 1], "y": [0, 1]},
        ))
        fig.update_layout(
            height=220,
            margin=dict(t=10, b=0, l=20, r=20),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": FOG, "family": "IBM Plex Mono"},
        )
        gauge_slot.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        label_slot.markdown(
            f'<p style="text-align:center; font-family:\'IBM Plex Mono\'; font-size:11px; '
            f'letter-spacing:0.1em; text-transform:uppercase; color:{color}; margin-top:-14px;">{label}</p>',
            unsafe_allow_html=True,
        )
        return label

    render_gauge(None)
    st.markdown(legend_html, unsafe_allow_html=True)
    recommend_slot.markdown(
        '<div class="crc-recommend">Fill in the customer\'s profile and press '
        '<b>Assess risk</b> to read their churn signal.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Handle submission
# ----------------------------------------------------------------------
if submitted:
    customer_data = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    df_customer = pd.DataFrame([customer_data])
    processed = preprocessor.transform(df_customer)
    prob = float(model.predict(processed, verbose=0)[0][0])
    pred = int(prob > 0.5)
    risk_label = "High Risk (Likely to Churn)" if pred == 1 else "Low Risk (Likely to Stay)"

    with col_meter:
        render_gauge(prob)

        if prob < 0.3:
            msg = f"<b>Stable.</b> No action needed — customer shows a low churn signal ({risk_label})."
        elif prob < 0.6:
            msg = "<b>Monitor.</b> Some churn indicators present. A light retention touch (loyalty offer, plan review) is worthwhile."
        else:
            msg = f"<b>Retention call recommended.</b> Strong churn signal ({risk_label}) — consider a targeted offer or contract incentive now."

        recommend_slot.markdown(f'<div class="crc-recommend">{msg}</div>', unsafe_allow_html=True)

st.markdown(
    f'<p style="text-align:center; font-family:\'IBM Plex Mono\'; font-size:11px; color:{FOG}; margin-top:32px;">'
    f'Customer Churn Prediction System &middot; Streamlit edition</p>',
    unsafe_allow_html=True,
)
