import joblib
import pandas as pd


model = joblib.load("rf_model.pkl")
print("✅ Model loaded")


test_data = pd.DataFrame([
    [1750397778.533793, 9, 72]   
], columns=["time", "protocol", "length"])


prediction = model.predict(test_data)
print(" Prediction:", "Attack" if prediction[0] == 1 else "Normal")

