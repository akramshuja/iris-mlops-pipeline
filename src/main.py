import os
import mlflow
import pandas as pd
import logging
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from contextlib import asynccontextmanager

# --- 1. Configure Logging ---
# Set up a logger that writes to a file named 'api.log'
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler() # Also log to console
    ]
)
logger = logging.getLogger(__name__)

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
    It loads the model on startup and initializes metrics.
    """
    logger.info("Application startup: Loading model...")
    # Read the tracking URI from an environment variable, with a default for local runs
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Load the model from the "Staging" stage
    MODEL_NAME = "IrisClassifier"
    MODEL_STAGE = "Staging"
    
    # Store the model and initialize metrics in the app's state
    app.state.model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    app.state.prediction_count = 0
    logger.info("Model loaded successfully and stored in app state!")
    
    yield
    
    # Code below this 'yield' runs on shutdown
    logger.info("Application shutdown...")


# --- Initialize FastAPI app with the lifespan manager ---
app = FastAPI(
    title="Iris Classifier API",
    version="1.0.0",
    lifespan=lifespan
)


# --- Define API Endpoints ---
@app.post("/predict")
def predict(iris_input: IrisInput, model = Depends(get_model)):
    """
    Takes Iris flower features and returns the predicted species.
    Logs the input and output.
    """
    # --- 2. Log the incoming request ---
    logger.info(f"Received prediction request: {iris_input.model_dump()}")

    # Convert Pydantic model to a pandas DataFrame
    input_data = pd.DataFrame([iris_input.model_dump()])
    
    # Make prediction
    prediction = model.predict(input_data)
    predicted_class = int(prediction[0])
    
    # --- 3. Log the prediction result and update metrics ---
    logger.info(f"Prediction result: {predicted_class}")
    app.state.prediction_count += 1
    
    return {"predicted_species": predicted_class}


@app.get("/metrics")
def get_metrics():
    """
    Returns simple application metrics.
    """
    logger.info("Metrics endpoint was accessed.")
    return {"prediction_count": app.state.prediction_count}


@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!"}

