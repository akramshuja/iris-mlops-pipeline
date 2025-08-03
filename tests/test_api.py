from fastapi.testclient import TestClient
from src.main import app
import pytest

# Mark the test function as an asyncio test
@pytest.mark.asyncio
async def test_api_endpoints():
    """
    Tests the API endpoints. The MLflow model loading is automatically
    mocked by the fixture in conftest.py.
    """
    # The TestClient will trigger the app's lifespan events,
    # but the model loading within it is now safely mocked.
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
