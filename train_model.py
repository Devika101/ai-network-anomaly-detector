import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("network_traffic.csv")
print(f" Loaded dataset with {len(df)} rows")


le = LabelEncoder()
df["protocol"] = le.fit_transform(df["protocol"])


X = df[["time", "protocol", "length"]]
y = df["label"]


print("\n Running Isolation Forest (Unsupervised)...")
iso_model = IsolationForest(contamination=0.5, random_state=42)
iso_preds = iso_model.fit_predict(X)

iso_preds = [0 if p == 1 else 1 for p in iso_preds]


print(" Isolation Forest Metrics:")
print("Accuracy:", accuracy_score(y, iso_preds))
print("Precision:", precision_score(y, iso_preds))
print("Recall:", recall_score(y, iso_preds))
print("F1 Score:", f1_score(y, iso_preds))

print("\n Running Random Forest (Supervised)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)


print(" Random Forest Metrics:")
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("Precision:", precision_score(y_test, rf_preds))
print("Recall:", recall_score(y_test, rf_preds))
print("F1 Score:", f1_score(y_test, rf_preds))


plt.figure(figsize=(8,5))
plt.hist(df[df["label"]==0]["length"], bins=50, alpha=0.6, label="Normal", color="green")
plt.hist(df[df["label"]==1]["length"], bins=50, alpha=0.6, label="Attack", color="red")
plt.title("Packet Length Distribution")
plt.xlabel("Packet Length")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("packet_length_distribution.png")
plt.show()

import joblib
joblib.dump(rf_model, "rf_model.pkl")
print(" Model saved as rf_model.pkl")

