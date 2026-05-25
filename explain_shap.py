import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

df = pd.read_csv("insurance_data.csv")
X = df.drop("Claim_Approved", axis=1)

with open("insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Calculating SHAP values for all 1,000 customers (this might take a few seconds)...")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
if isinstance(shap_values, list):
    shap_values_approved = shap_values[1]        # Older SHAP versions
else:
    shap_values_approved = shap_values[:, :, 1]  # Newer SHAP versions

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_approved, X, show=False)
plt.savefig("shap_summary.png", bbox_inches='tight', dpi=300)
print("Done! The global insights chart is saved as 'shap_summary.png'.")