from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib

# Load the model once when API starts
model = joblib.load("credit_risk_predictor_model.pkl")

# Create FastAPI application
app = FastAPI(title="Credit Risk Predictor", description="Machine Learning API for predicting loan default risk",
              version="1.0")

# User Input schema
class LoanApplication(BaseModel): # The user is expected to provide the below fields with correct data type
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int

# Function for Prediction of results by the model
@app.post("/predict")
def predict(data: LoanApplication):
    
    # Convert input JSON to DataFrame
    input_df = pd.DataFrame([data.model_dump()])
    input_df["employment_stability_ratio"] = input_df["person_emp_length"]/input_df["person_age"]

    # Predict probability of default
    probability = float(model.predict_proba(input_df)[0, 1])

    # Get model prediction (uses default threshold=0.5)
    prediction = int(model.predict(input_df)[0])

    # Return the results in dictionary format
    return {"default_probability": round(probability, 4), 
            "default_prediction": prediction, 
            "risk_level": "High Risk" if prediction == 1 else "Low Risk"}
    
app.mount("/", StaticFiles(directory="static", html=True), name="static")    