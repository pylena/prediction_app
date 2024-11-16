from typing import Optional
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI()
model = joblib.load('kmeans_model.joblib')
scaler = joblib.load('Models/scaler.joblib')

@app.get("/")
async def root():
    return {"message": "Hello World"}






# Define a Pydantic model for input data validation

class InputFeatures(BaseModel):
    current_value : float
    goals: float
    appearance: float	
    position_numeric: int
    

def preprocessing(input_features: InputFeatures):
    dict_f = {
        'appearance' : input_features.appearance,
        'goals': input_features.goals,
        'position_numeric': input_features.position_numeric,
        'current_value': input_features.current_value
        }
    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    # Scale the input features
    scaled_features = scaler.transform([list(dict_f.values())])

    return scaled_features

@app.get("/")
def read_root():
    return {"message": "Welcome to the Football Player Clustering "}
  



@app.post("/predict")
async def predict(input_features: InputFeatures):
    data = preprocessing(input_features)
    y_pred = model.predict(data)
    return {"pred": y_pred.tolist()[0]}

