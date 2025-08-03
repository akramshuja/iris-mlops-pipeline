import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def mock_model_load(mocker):
    """
    This is a pytest fixture that mocks the MLflow model loading function.
    It replaces the real function with a fake one that returns a dummy model.
    """
    class FakeModel:
        def predict(self, data):
            # Return a predictable, dummy prediction
            return [0]

    # Use mocker to patch the real function in src/main.py
    return mocker.patch("src.main.mlflow.pyfunc.load_model", return_value=FakeModel())


def test_api_with_mocked_model(mock_model_load):
    """
    Tests the entire API using a mocked model.
    The TestClient is used as a context manager to handle the app's lifespan.
    """
    # Import the app *after* the mock is potentially active
    from src.main import app
    
    # Use the TestClient as a context manager
    with TestClient(app) as client:
        # Test the root endpoint
        response_root = client.get("/")
        assert response_root.status_code == 200
        assert response_root.json() == {"message": "Welcome to the Iris Classifier API!"}

        # Test the predict endpoint
        test_data = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        response_predict = client.post("/predict", json=test_data)
        assert response_predict.status_code == 200
        # Assert that we get the dummy prediction (0) from our FakeModel
        assert response_predict.json() == {"predicted_species": 0}
