import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1000

data = {
	"Customer_Age": np.random.randint(18, 80, n_samples),
	"Policy_Tenure_Months": np.random.randint(1, 120, n_samples),
	"Claim_Amount": np.random.uniform(500, 50000, n_samples).round(2),
	"Previous_Claims": np.random.randint(0, 5, n_samples),
	"Credit_Score": np.random.randint(300, 850, n_samples),
	"Location_Type": np.random.choice([0, 1, 2], n_samples),
}

df = pd.DataFrame(data)

approval_prob = (
	(df["Credit_Score"] > 600).astype(int) * 0.3 +
	(df["Previous_Claims"] < 2).astype(int) * 0.3 +
	(df["Claim_Amount"] < 15000).astype(int) * 0.2 +
	(df["Policy_Tenure_Months"] > 12).astype(int) * 0.2
)
approval_prob -= (df["Location_Type"] == 2).astype(int) * 0.15
approval_prob += np.random.normal(0, 0.1, n_samples)

df["Claim_Approved"] = (approval_prob > 0.5).astype(int)

df.to_csv("insurance_data.csv", index=False)
print("Dataset 'insurance_data.csv' successfully generated!")
