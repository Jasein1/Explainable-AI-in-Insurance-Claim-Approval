import streamlit as st
import pandas as pd
import pickle
import lime.lime_tabular
import matplotlib.pyplot as plt

# 1. Page Config MUST be the very first command
st.set_page_config(page_title="AI Insurance Underwriter", page_icon="🛡️", layout="wide")

# 2. Hide default Streamlit menus for a cleaner look (Optional but recommended)
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# 3. Load Machine Learning Assets
@st.cache_resource
def load_assets():
    with open("insurance_model.pkl", "rb") as f:
        model = pickle.load(f)
    df = pd.read_csv("insurance_data.csv")
    X = df.drop("Claim_Approved", axis=1)
    return model, X

try:
    model, X_train = load_assets()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# 4. Initialize LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Denied', 'Approved'],
    mode='classification'
)

# 5. Clean Dashboard Layout
st.title("🛡️ AI Insurance Underwriting Dashboard")
st.markdown("Predict claim approvals and generate transparent LIME explanations.")
st.divider()

# Create two columns for a professional dashboard feel
col1, col2 = st.columns([1, 2])

with col1:
    st.header("📝 Customer Details")
    age = st.slider("Customer Age", 18, 80, 35)
    tenure = st.slider("Policy Tenure (Months)", 1, 120, 24)
    amount = st.number_input("Claim Amount ($)", min_value=500, max_value=50000, value=15000)
    prev_claims = st.selectbox("Previous Claims", [0, 1, 2, 3, 4, 5])
    credit = st.slider("Credit Score", 300, 850, 650)
    location = st.selectbox("Location Type", [0, 1, 2], format_func=lambda x: ["Urban", "Suburban", "Rural"][x])
    
    predict_button = st.button("Run AI Prediction", type="primary", use_container_width
