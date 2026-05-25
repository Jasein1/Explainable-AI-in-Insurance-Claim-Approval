import streamlit as st
import pandas as pd
import pickle
import lime.lime_tabular
import streamlit.components.v1 as components
import socket
import threading
import time
import os
import json
from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse
from starlette.routing import Route, Mount
from starlette.staticfiles import StaticFiles
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import uvicorn

# Determine paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(BASE_DIR, "templates", "index.html")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Ensure folders exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "templates"), exist_ok=True)

# Load machine learning assets
@st.cache_resource
def load_assets():
    with open(os.path.join(BASE_DIR, "insurance_model.pkl"), "rb") as f:
        model = pickle.load(f)
    df = pd.read_csv(os.path.join(BASE_DIR, "insurance_data.csv"))
    X = df.drop("Claim_Approved", axis=1)
    return model, X

model, X_train = load_assets()

# Initialize LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Denied', 'Approved'],
    mode='classification'
)

# Define Starlette API
api_app = Starlette()

# Add CORS Middleware to ensure frontend can talk to backend
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Index HTML
async def get_index(request):
    try:
        if os.path.exists(TEMPLATE_PATH):
            with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
                html_content = f.read()
            return HTMLResponse(content=html_content, status_code=200)
        else:
            return HTMLResponse(content="<h1>Templates not loaded yet. Redesigning UI...</h1>", status_code=200)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Error loading page: {str(e)}</h1>", status_code=500)

# Prediction Endpoint
async def predict_claim(request):
    try:
        data = await request.json()
        
        # Extract inputs
        age = int(data.get("Customer_Age", 35))
        tenure = int(data.get("Policy_Tenure_Months", 24))
        amount = float(data.get("Claim_Amount", 5000))
        prev_claims = int(data.get("Previous_Claims", 0))
        credit = int(data.get("Credit_Score", 650))
        location = int(data.get("Location_Type", 0)) # 0: Urban, 1: Suburban, 2: Rural
        
        # Build dataframe
        input_data = pd.DataFrame({
            "Customer_Age": [age],
            "Policy_Tenure_Months": [tenure],
            "Claim_Amount": [amount],
            "Previous_Claims": [prev_claims],
            "Credit_Score": [credit],
            "Location_Type": [location]
        })
        
        # Run model
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        
        prob_denied = float(probabilities[0])
        prob_approved = float(probabilities[1])
        
        # Generate LIME explanation
        exp = explainer.explain_instance(
            data_row=input_data.iloc[0].values,
            predict_fn=model.predict_proba
        )
        
        lime_list = exp.as_list()
        
        # Format LIME factors
        lime_factors = []
        for rule, weight in lime_list:
            # Map column names in rule to readable formats for the UI
            rule_readable = rule
            rule_readable = rule_readable.replace("Previous_Claims", "Previous Claims")
            rule_readable = rule_readable.replace("Claim_Amount", "Claim Amount")
            rule_readable = rule_readable.replace("Credit_Score", "Credit Score")
            rule_readable = rule_readable.replace("Policy_Tenure_Months", "Policy Tenure")
            rule_readable = rule_readable.replace("Customer_Age", "Customer Age")
            
            # Map location type numbers to names if present in rule
            if "Location_Type" in rule:
                # E.g. 'Location_Type <= 0.00' -> 'Location Type is Urban'
                rule_readable = rule_readable.replace("Location_Type", "Location Type")
            
            lime_factors.append({
                "rule": rule_readable,
                "weight": float(weight)
            })
            
        # Determine key insights dynamically
        insights = []
        
        # 1. Risk profile
        if prev_claims >= 2 or amount > 25000:
            insights.append({
                "type": "risk",
                "status": "negative",
                "label": "High Risk Profile",
                "icon": "shield-alert"
            })
        else:
            insights.append({
                "type": "risk",
                "status": "positive",
                "label": "Low Risk Profile",
                "icon": "shield-check"
            })
            
        # 2. Credit profile
        if credit >= 700:
            insights.append({
                "type": "credit",
                "status": "positive",
                "label": "Excellent Credit",
                "icon": "sparkles"
            })
        elif credit >= 600:
            insights.append({
                "type": "credit",
                "status": "neutral",
                "label": "Strong Credit Score",
                "icon": "brain"
            })
        else:
            insights.append({
                "type": "credit",
                "status": "negative",
                "label": "Poor Credit Score",
                "icon": "alert-circle"
            })
            
        # 3. Claims history profile
        if prev_claims == 0:
            insights.append({
                "type": "claims",
                "status": "positive",
                "label": "No Prior Claims",
                "icon": "file-check"
            })
        elif prev_claims == 1:
            insights.append({
                "type": "claims",
                "status": "neutral",
                "label": "Single Prior Claim",
                "icon": "file-text"
            })
        else:
            insights.append({
                "type": "claims",
                "status": "negative",
                "label": "Multiple Prior Claims",
                "icon": "files"
            })

        response_data = {
            "prediction": prediction,
            "confidence": prob_approved if prediction == 1 else prob_denied,
            "probabilities": {
                "Denied": prob_denied,
                "Approved": prob_approved
            },
            "lime_factors": lime_factors,
            "insights": insights
        }
        
        return JSONResponse(response_data)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Deploy Endpoint
async def deploy_model(request):
    # Simulates a multi-step deployment flow
    return JSONResponse({
        "status": "success",
        "message": "Deployment simulation initialized"
    })

# Define Routes
api_app.routes.extend([
    Route("/", get_index, methods=["GET"]),
    Route("/api/predict", predict_claim, methods=["POST"]),
    Route("/api/deploy", deploy_model, methods=["POST"]),
    Mount("/static", app=StaticFiles(directory=STATIC_DIR), name="static")
])

# Helper function to check if port is in use
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

# Start server in background thread if not already running
PORT = 8505
if not is_port_in_use(PORT):
    def run_api():
        uvicorn.run(api_app, host="127.0.0.1", port=PORT, log_level="warning")
    
    server_thread = threading.Thread(target=run_api, daemon=True)
    server_thread.start()
    time.sleep(0.5) # Allow server to start

# --- Streamlit Presentation Layer ---
# This serves as a container that displays the custom HTML in a full-screen iframe
if __name__ == "__main__" or "streamlit" in os.path.basename(st.__file__):
    st.set_page_config(
        page_title="AI Insurance Claim Underwriter",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Clean UI styling to hide all Streamlit elements
    st.markdown("""
        <style>
            /* Hide Streamlit header, footer, and sidebar toggle */
            header, footer, [data-testid="collapsedControl"] {
                display: none !important;
            }
            
            /* Remove standard page margins and padding */
            .main .block-container {
                padding: 0 !important;
                margin: 0 !important;
                max-width: 100% !important;
                width: 100% !important;
                height: 100vh !important;
            }
            
            .stApp {
                margin: 0 !important;
                padding: 0 !important;
                overflow: hidden !important;
                height: 100vh !important;
                background: #f8fafc;
            }
            
            iframe {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                border: none;
                margin: 0;
                padding: 0;
                overflow: hidden;
                z-index: 999999;
                background: #f8fafc;
            }
        </style>
    """, unsafe_allow_html=True)

    # Load custom HTML inside iframe pointing to our background server
    components.html(
        f'<iframe src="http://127.0.0.1:{PORT}/" allow="clipboard-write"></iframe>',
        height=1000,
    )
