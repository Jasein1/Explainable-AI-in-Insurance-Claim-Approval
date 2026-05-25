import streamlit as st
import pandas as pd
import pickle
import lime.lime_tabular
import matplotlib.pyplot as plt

# 1. Page Config
st.set_page_config(
    page_title="AI Underwriter",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 2. EXACT UI MATCH CSS INJECTION
st.markdown("""
    <style>
        /* Base Dark Background */
        .stApp {
            background-color: #121826 !important;
        }

        /* Global Text Styling */
        h1, h2, h3, h4, h5, h6, p, span, label, div {
            color: #E2E8F0 !important;
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Hide default Streamlit fluff */
        header, footer, #MainMenu {visibility: hidden;}
        [data-testid="collapsedControl"] {display: none;}

        /* --- LAYOUT PANELS --- */
        
        /* Left Panel (Inputs) */
        [data-testid="column"]:nth-of-type(1) {
            background-color: #1A2235;
            padding: 2rem !important;
            border-radius: 12px;
            border: 1px solid #2A3449;
            height: 100%;
        }

        /* Right Result Cards (Sub-columns) */
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(1) > div > div > div > div[data-testid="stVerticalBlock"],
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(2) > div > div > div > div[data-testid="stVerticalBlock"] {
            background-color: #1A2235;
            padding: 1.5rem !important;
            border-radius: 12px;
            border: 1px solid #2A3449;
        }

        /* --- CONTROLS & ACCENTS --- */

        /* Indigo Accent Button */
        .stButton > button {
            background-color: #6366F1 !important;
            color: white !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 0.6rem 1.5rem !important;
            font-weight: 600 !important;
            transition: 0.3s !important;
            width: 100%;
        }
        .stButton > button:hover {
            background-color: #4F46E5 !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
        }

        /* Sliders */
        .stSlider > div[data-baseweb="slider"] > div > div {
            background-color: #6366F1 !important; 
        }

        /* Dark Input Boxes */
        div[data-baseweb="select"] > div, 
        input[type="number"] {
            background-color: #0F172A !important;
            color: white !important;
            border: 1px solid #334155 !important;
            border-radius: 6px !important;
        }
        
        /* Matplotlib Chart Background Fix */
        .stImage > img {
            border-radius: 8px;
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
    st.error("Error loading model. Make sure 'insurance_model.pkl' is uploaded.")
    st.stop()

# 4. Initialize LIME
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Denied', 'Approved'],
    mode='classification'
)

# 5. UI ARCHITECTURE (Matches Screenshot Layout)

# Main Grid: Left Panel (1) : Right Area (3)
left_panel, right_area = st.columns([1, 3], gap="large")

with left_panel:
    st.markdown("### 🛡️ AI Underwriter")
    st.caption("Smart. Transparent. Trusted.")
    st.write("---")
    
    st.markdown("#### Customer Details")
    st.caption("Provide customer and claim information")
    
    age = st.slider("Customer Age", 18, 80, 35)
    tenure = st.slider("Policy Tenure (Months)", 1, 120, 24)
    amount = st.number_input("Claim Amount ($)", min_value=500, max_value=50000, value=5000, step=500)
    prev_claims = st.selectbox("Previous Claims", [0, 1, 2, 3, 4, 5])
    credit = st.slider("Credit Score", 300, 850, 650)
    location = st.selectbox("Location Type", [0, 1, 2], format_func=lambda x: ["Urban", "Suburban", "Rural"][x])

with right_area:
    st.markdown("## 🛡️ AI Insurance Claim Underwriter")
    st.caption("This system predicts claim approvals and provides transparent, AI-driven explanations for its decisions.")
    st.write("")
    
    # Analyze Button row
    analyze_btn_col, spacer = st.columns([1, 4])
    with analyze_btn_col:
        analyze_button = st.button("✨ Analyze Claim")
    
    st.write("")
    
    # Result Cards (Side by Side)
    res_col1, res_col2 = st.columns(2, gap="medium")
    
    if analyze_button:
        # Data Prep
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
        
        with res_col1:
            st.markdown("#### ✅ AI Decision")
            st.write("---")
            if prediction == 1:
                st.success(f"**CLAIM APPROVED**")
                st.info(f"Confidence: **{probabilities[1]*100:.1f}%**")
            else:
                st.error(f"**CLAIM DENIED**")
                st.warning(f"Confidence: **{probabilities[0]*100:.1f}%**")
                
            st.markdown("**Key Insights:**")
            if credit >= 700: st.write("• Strong Credit Score")
            if prev_claims == 0: st.write("• Clean Claims History")
            if amount > 15000: st.write("• High Claim Amount Review")
                
        with res_col2:
            st.markdown("#### 👁️ Decision Transparency (LIME)")
            st.write("---")
            with st.spinner("Processing explanation..."):
                exp = explainer.explain_instance(
                    data_row=input_data.iloc[0].values,
                    predict_fn=model.predict_proba
                )
                
                # Dark theme for the matplotlib figure
                fig = exp.as_pyplot_figure()
                fig.patch.set_facecolor('#1A2235') # Match card background
                ax = fig.gca()
                ax.set_facecolor('#1A2235')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                
                st.pyplot(fig)
                
    else:
        # Default empty state
        with res_col1:
            st.markdown("#### ✅ AI Decision")
            st.write("---")
            st.info("No analysis run yet. Adjust details and click Analyze.")
        with res_col2:
            st.markdown("#### 👁️ Decision Transparency (LIME)")
            st.write("---")
            st.caption("Waiting for parameters...")
