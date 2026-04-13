from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Wine Quality API Running"}

@app.post("/predict")
def predict(data: dict):
    features = list(data.values())
    features = np.array(features).reshape(1, -1)

    prediction = model.predict(features)

    return {
        "prediction": float(prediction[0]),
        "name": "Varun Sandesh",
        "reg_no": "2022BCS0094"
    }
