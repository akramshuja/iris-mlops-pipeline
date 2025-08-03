import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_models():
    """Loads data, trains two models, and logs them with MLflow."""
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    print("Loading data...")
    # Load the processed data from the DVC-tracked file
    df = pd.read_csv('data/raw/iris.csv')

    # Prepare data for modeling
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define models and their parameters
    models = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    for model_name, model in models.items():
        print(f"--- Training {model_name} ---")
        # Start a new MLflow run
        with mlflow.start_run(run_name=model_name):
            # Train the model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)

            # Log parameters, metrics, and the model
            print(f"  Accuracy: {accuracy:.4f}")
            mlflow.log_param("model_type", model_name)
            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", accuracy)
            
            # Log the model itself
            mlflow.sklearn.log_model(model, "model")
            print(f"--- {model_name} run logged in MLflow ---")

if __name__ == "__main__":
    train_models()