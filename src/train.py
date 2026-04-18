from preprocess import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

# Load data
df = load_data()

# Convert target column
df["Churn"] = df["Churn"].map({"Yes":1, "No":0})

# Features / target
X = df.drop(["customerID", "Churn"], axis=1)
y = df["Churn"]

# Encode text columns
for col in X.select_dtypes(include="object").columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Predict
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:,1]

# Scores
print("Accuracy:", round(accuracy_score(y_test, pred), 3))
print("ROC-AUC :", round(roc_auc_score(y_test, proba), 3))

# Save model
joblib.dump(model, "models/churn_model.pkl")

print("Model saved successfully.")