import pandas as pd
import pickle
import lime
import lime.lime_tabular

# 1. Load the data and the trained model
df = pd.read_csv("insurance_data.csv")
X = df.drop("Claim_Approved", axis=1)

with open("insurance_model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Find a specific customer who was DENIED by the model
# We will just pick the first person in the dataset that the model rejects
predictions = model.predict(X)
denied_indices = [i for i, pred in enumerate(predictions) if pred == 0]
customer_to_explain = denied_indices[0] 
customer_data = X.iloc[customer_to_explain]

print(f"Analyzing Customer #{customer_to_explain} who was DENIED...")

# 3. Set up the LIME Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X.values,
    feature_names=X.columns.tolist(),
    class_names=['Denied', 'Approved'],
    mode='classification'
)

# 4. Generate the explanation for this specific customer
exp = explainer.explain_instance(
    data_row=customer_data.values, 
    predict_fn=model.predict_proba
)

# 5. Save the visual explanation as an HTML file
exp.save_to_file("lime_explanation.html")
print("Done! The visual explanation has been saved to 'lime_explanation.html'.")