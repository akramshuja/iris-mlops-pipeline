import os
import mlflow
import pandas as pd
import logging
from fastapi import FastAPI, Request, Depends
from pydantic import BaseModel
from contextlib import asynccontextmanager
from prometheus_fastapi_instrumentator import Instrumentator

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
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
    It loads the model on startup.
    """
    logger.info("Application startup: Loading model...")
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    MODEL_NAME = "IrisClassifier"
    MODEL_STAGE = "Staging"
    
    app.state.model = mlflow.pyfunc.load_model(model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}")
    logger.info("Model loaded successfully!")
    
    yield
    
    logger.info("Application shutdown...")


# --- Initialize FastAPI app ---
app = FastAPI(
    title="Iris Classifier API",
    version="1.0.0",
    lifespan=lifespan
)

# --- Set up Prometheus metrics (BEFORE app starts) ---
Instrumentator().instrument(app).expose(app)


# --- Define API Endpoints ---
@app.post("/predict")
def predict(iris_input: IrisInput, model = Depends(get_model)):
    """
    Takes Iris flower features and returns the predicted species.
    """
    logger.info(f"Received prediction request: {iris_input.model_dump()}")

    input_data = pd.DataFrame([iris_input.model_dump()])
    prediction = model.predict(input_data)
    predicted_class = int(prediction[0])
    
    logger.info(f"Prediction result: {predicted_class}")
    
    return {"predicted_species": predicted_class}


@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!"}
