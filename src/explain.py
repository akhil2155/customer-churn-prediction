from preprocess import load_data
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = load_data()
df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

X = df.drop(["customerID", "Churn"], axis=1)

# Encode text columns
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Load trained model
model = joblib.load("models/churn_model.pkl")

# Feature importance
importance = model.feature_importances_

feat_imp = pd.DataFrame({
    "feature": X.columns,
    "importance": importance
})

feat_imp = feat_imp.sort_values("importance", ascending=True)

# Plot
plt.figure(figsize=(10,7))
plt.barh(feat_imp["feature"], feat_imp["importance"])
plt.title("Top Features Driving Customer Churn")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("models/shap_summary.png", dpi=300)

print("Feature importance chart saved successfully")