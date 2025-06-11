import streamlit as st

# Page settings
st.set_page_config(
    page_title="Loan Approval ML App",
    page_icon="💳",
    layout="centered"
)

# Title

st.markdown("<h1 style='text-align: center; color: teal;'>💼 LoanScope – Focused on predicting both risk and approval.</h1>", unsafe_allow_html=True)

# Visible & Styled Intro Card
with st.container():
    st.markdown("""
    <div style="
        background-color: #e6f7ff;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 0 0 8px rgba(0,0,0,0.05);
        color: #333333;
    ">
        <h3 style='color: #005f73;'>🏦 Purpose</h3>
        <p style='font-size: 16px; line-height: 1.6;'>
            This project simulates a simplified <strong>loan approval system</strong> using machine learning and real-world decision criteria like credit score, income, and debt ratio.
        </p>
        <p style='font-size: 16px; line-height: 1.6;'>
            It’s a full-stack ML app showcasing how data science can be deployed as a real-time product with frontend and backend integration.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Overview Section
st.subheader("📌 Project Breakdown")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 🔧 Data & Training")
    st.markdown("- Clean and encode data\n- Scale numerical features\n- Train classification/regression models")

with col2:
    st.markdown("### ⚙️ Backend API (FastAPI)")
    st.markdown("- Deploy ML models via REST\n- Use JSON payloads\n- Return predictions in real-time")

with col3:
    st.markdown("### 🖥️ Frontend (Streamlit)")
    st.markdown("- User form input\n- Display loan approval decisions\n- Explain risk levels")

# Technical Details (Collapsible)
with st.expander("🔬 Technical Details"):
    st.markdown("""
    - Pipelines built with `scikit-learn` for consistent preprocessing  
    - Evaluation with metrics: Accuracy, AUC, MAE  
    - Backend is fully decoupled with `FastAPI`  
    - Frontend calls the API live for a realistic simulation
    """)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Built using Streamlit, FastAPI & scikit-learn</p>", unsafe_allow_html=True)
