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


@pytest.fixture
def client(mock_model_load):
    """

    This fixture creates a TestClient for our API.
    Crucially, it depends on the mock_model_load fixture, ensuring the
    model loading is mocked *before* the application is imported.
    """
    from src.main import app  # Import app here, after the mock is active
    return TestClient(app)


def test_read_root(client):
    """Tests the root endpoint, which doesn't require the model."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Iris Classifier API!"}


def test_predict_endpoint(client):
    """Tests the /predict endpoint using the mocked model."""
    test_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    # Assert that we get the dummy prediction (0) from our FakeModel
    assert response.json() == {"predicted_species": 0}