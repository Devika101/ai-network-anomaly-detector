import joblib
import pandas as pd

# Load the saved model
model = joblib.load("rf_model.pkl")
print("✅ Model loaded")

# Example input: [time, protocol, length]
# You can adjust these values based on your CSV
test_data = pd.DataFrame([
    [1750397778.533793, 9, 72]   # Replace with actual values from your CSV
], columns=["time", "protocol", "length"])

# Make prediction
prediction = model.predict(test_data)
print("🧠 Prediction:", "Attack" if prediction[0] == 1 else "Normal")

