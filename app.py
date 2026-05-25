import streamlit as st
import pandas as pd
import pickle
import lime.lime_tabular
import matplotlib.pyplot as plt

# 1. Page Config (MUST be first)
st.set_page_config(
    page_title="AI Insurance Claim Underwriter",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. CSS INJECTION (Your custom UI/UX design)
st.markdown("""
    <style>
        /* Hide Streamlit default menus */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        [data-testid="collapsedControl"] {display: none;}

        /* App Background */
        .stApp {
            background-color: #f8fafc;
        }

        /* Dashboard Panels */
        div[data-testid="column"] {
            background-color: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            border: 1px solid #e2e8f0;
        }

        /* Custom Button Styling */
        .stButton > button {
            background-color: #2563eb !important;
            color: white !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 0.75rem 1rem !important;
            font-weight: 600 !important;
            transition: all 0.2s ease !important;
            width: 100% !important;
        }
        .stButton > button:hover {
            background-color: #1d4ed8 !important;
            box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
            transform: translateY(-1px) !important;
        }

        /* Input Boxes */
        div[data-baseweb="select"] > div, 
        input[type="number"], 
        div[data-baseweb="slider"] {
            border-radius: 8px !important;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #0f172a !important;
        }
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
    st.error("Error: Could not find 'insurance_model.pkl' or 'insurance_data.csv'. Please ensure they are uploaded to GitHub.")
    st.stop()

# 4. Initialize LIME
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Denied', 'Approved'],
    mode='classification'
)

# 5. Dashboard Layout
st.title("🛡️ AI Insurance Underwriting Dashboard")
st.markdown("Predict claim approvals and generate transparent LIME explanations.")
st.write("") # Spacer

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.subheader("📝 Customer Profile")
    age = st.slider("Customer Age", 18, 80, 35)
    tenure = st.slider("Policy Tenure (Months)", 1, 120, 24)
    amount = st.number_input("Claim Amount ($)", min_value=500, max_value=50000, value=5000, step=500)
    prev_claims = st.selectbox("Previous Claims", [0, 1, 2, 3, 4, 5])
    credit = st.slider("Credit Score", 300, 850, 650)
    location = st.selectbox("Location Type", [0, 1, 2], format_func=lambda x: ["Urban", "Suburban", "Rural"][x])
    
    st.write("") # Spacer
    predict_button = st.button("Run AI Prediction", use_container_width=True)

with col2:
    st.subheader("📊 AI Decision Engine")
    
    if predict_button:
        # Build Dataframe
        input_data = pd.DataFrame({
            "Customer_Age": [age],
            "Policy_Tenure_Months": [tenure],
            "Claim_Amount": [amount],
            "Previous_Claims": [prev_claims],
            "Credit_Score": [credit],
            "Location_Type": [location]
        })
        
        # Predictions
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        
        # UI Feedback
        if prediction == 1:
            st.success(f"✅ **CLAIM APPROVED** (Confidence: {probabilities[1]*100:.1f}%)")
        else:
            st.error(f"❌ **CLAIM DENIED** (Confidence: {probabilities[0]*100:.1f}%)")
            
        st.write("---")
        st.markdown("**Transparent LIME Explanation**")
        st.caption("Green bars indicate factors pushing toward Approval. Red bars indicate factors pushing toward Denial.")
        
        # Generate LIME
        with st.spinner("Processing explanation..."):
            exp = explainer.explain_instance(
                data_row=input_data.iloc[0].values,
                predict_fn=model.predict_proba
            )
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)
            
    else:
        st.info("👈 Configure the customer profile on the left and click 'Run AI Prediction' to generate an underwriting decision.")
