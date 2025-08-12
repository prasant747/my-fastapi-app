import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the model
model = joblib.load('model.joblib')
level_encoder = joblib.load('label_encoder.joblib')

app = FastAPI()

class InputData(BaseModel):
    input1: float
    input2: str
    input3: float

def predict_value(input_data: InputData):
    input_df = pd.DataFrame({
        'input1': [input_data.input1],
        'input2': [input_data.input2],
        'input3': [input_data.input3]
    })
    input_df['input2'] = level_encoder.fit_transform(input_df['input2'])
    prediction = model.predict(input_df)
    return prediction[0]

@app.post("/predict")

def predict(input_data: InputData):
    prediction = predict_value(input_data)
    return {"prediction": prediction,
            'message': "Prediction successful"
            }

# test this by running the FastAPI server
# uvicorn app_new.main:app --reload


