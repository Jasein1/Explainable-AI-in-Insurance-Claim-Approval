import streamlit as st
import pandas as pd
import pickle
import lime.lime_tabular
import matplotlib.pyplot as plt

# 1. Page Config MUST be the very first Streamlit command
st.set_page_config(page_title="AI Insurance Underwriter", page_icon="🛡️", layout="wide")

# 2. Load Assets
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

# Initialize LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Denied', 'Approved'],
    mode='classification'
)

# 3. Clean UI Layout
st.title("🛡️ AI Insurance Underwriting Dashboard")
st.markdown("Predict claim approvals and generate transparent LIME explanations.")
st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    st.header("📝 Customer Details")
    age = st.slider("Customer Age", 18, 80, 35)
    tenure = st.slider("Policy Tenure (Months)", 1, 120, 24)
    amount = st.number_input("Claim Amount ($)", min_value=500, max_value=50000, value=5000)
    prev_claims = st.selectbox("Previous Claims", [0, 1, 2, 3, 4, 5])
    credit = st.slider("Credit Score", 300, 850, 650)
    location = st.selectbox("Location Type", [0, 1, 2], format_func=lambda x: ["Urban", "Suburban", "Rural"][x])
    
    predict_button = st.button("Run AI Prediction", type="primary", use_container_width=True)

with col2:
    st.header("📊 AI Decision & Explanation")
    
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
        
        if prediction == 1:
            st.success(f"✅ Claim Approved (Confidence: {probabilities[1]*100:.1f}%)")
        else:
            st.error(f"❌ Claim Denied (Confidence: {probabilities[0]*100:.1f}%)")
            
        st.subheader("Transparent LIME Explanation")
        st.info("The chart below reverse-engineers the AI's decision. Green bars push toward Approval, red bars push toward Denial.")
        
        # Generate and Display LIME
        with st.spinner("Generating explanation..."):
            exp = explainer.explain_instance(
                data_row=input_data.iloc[0].values,
                predict_fn=model.predict_proba
            )
            
            # Display the plot directly in Streamlit
            fig = exp.as_pyplot_figure()
            st.pyplot(fig)
            
    else:
        st.info("👈 Enter customer details and click 'Run AI Prediction' to see results.")
