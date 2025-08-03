import os
import mlflow
import pandas as pd
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- Pydantic model for input validation ---
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# --- Model Dependency ---
def get_model(request: Request):
    """A dependency function to get the model from the app state."""
    return request.app.state.model

# --- Lifespan manager for the FastAPI app ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs on application startup and shutdown.
    It loads the model on startup and stores it in the app's state.
    """
    print("Application startup: Loading model...")
    # Read the tracking URI from an environment variable, with a default for local runs
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Load the model from the "Staging" stage
    MODEL_NAME = "IrisClassifier"
    MODEL_STAGE = "Staging"
    
    # Store the model in the app's state
    app.state.model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    print("Model loaded successfully and stored in app state!")
    
    yield
    
    # Code below this 'yield' runs on shutdown
    print("Application shutdown...")


# --- Initialize FastAPI app with the lifespan manager ---
app = FastAPI(
    title="Iris Classifier API",
    version="1.0.0",
    lifespan=lifespan
)


# --- Define the prediction endpoint ---
@app.post("/predict")
def predict(iris_input: IrisInput, model = Depends(get_model)):
    """
    Takes Iris flower features and returns the predicted species.
    The model is now injected as a dependency.
    """
    # Convert Pydantic model to a pandas DataFrame
    input_data = pd.DataFrame([iris_input.model_dump()]) # Use model_dump() for Pydantic v2
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # The model returns a numpy array, get the first element
    predicted_class = int(prediction[0])
    
    # Return the result
    return {"predicted_species": predicted_class}


@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!"}