import os
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# --- 1. Pydantic model for input validation ---
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# --- 2. Initialize FastAPI app ---
app = FastAPI(title="Iris Classifier API", version="1.0.0")

# --- 3. Load the MLflow model ---

# Read the tracking URI from an environment variable, with a default for local runs
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load the model from the "Staging" stage
MODEL_NAME = "IrisClassifier"
MODEL_STAGE = "Staging"
model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
print("Model loaded successfully!")

# --- 4. Define the prediction endpoint ---
@app.post("/predict")
def predict(iris_input: IrisInput):
    """
    Takes Iris flower features and returns the predicted species.
    """
    # Convert Pydantic model to a pandas DataFrame
    input_data = pd.DataFrame([iris_input.dict()])

    # Make prediction
    prediction = model.predict(input_data)

    # The model returns a numpy array, get the first element
    predicted_class = int(prediction[0])

    # Return the result
    return {"predicted_species": predicted_class}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!"}