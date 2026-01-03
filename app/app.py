import streamlit as st
import joblib
import pandas as pd
# Import the specific model type you used for training
from sklearn.ensemble import RandomForestRegressor 
from pathlib import Path

# Get the base directory (where app.py is located) and construct the model path
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "gold_model_2026.pkl"

# Load the model robustly
try:
    # This loads the Random Forest model you just trained
    model = joblib.load(model_path)
    # st.success("Model loaded successfully!") # Optional success message
except FileNotFoundError:
    st.error(f"Error: The model file was not found at {model_path}")
    st.stop()

st.title("Gold Price Predictor 2026")

# Use the latest inputs that gave you a good prediction
# Ensure these match the columns your model expects ('SPX', 'USO', 'SLV', 'EUR/USD')
spx = st.number_input("SPX (S&P 500 Index)", value=6932.05)
uso = st.number_input("USO (United States Oil Fund LP)", value=70.20)
slv = st.number_input("SLV (iShares Silver Trust)", value=65.22)
eur = st.number_input("EUR/USD", value=1.1796)

if st.button("Predict"):
    # Create a dictionary with matching keys and convert to DataFrame
    data_dict = {
        'SPX': spx,
        'USO': uso,
        'SLV': slv,
        'EUR/USD': eur
    }
    new_data = pd.DataFrame([data_dict])
    
    # Reorder columns to match the model's expected feature order
    new_data = new_data[model.feature_names_in_]
    
    prediction = model.predict(new_data)
    st.success(f"Predicted GLD Price: ${prediction[0]:.2f}")

