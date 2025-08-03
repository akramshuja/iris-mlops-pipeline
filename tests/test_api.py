from fastapi.testclient import TestClient
from src.main import app, get_model

# --- Define a fake model and a dependency override function ---
class FakeModel:
    def predict(self, data):
        return [0] # Always return a dummy prediction

def get_fake_model():
    """This function will be used to override the real get_model dependency."""
    return FakeModel()

# --- Use the dependency override on the app ---
app.dependency_overrides[get_model] = get_fake_model

# --- Test the API ---
def test_api_endpoints():
    """
    Tests the API endpoints. The real model dependency is now
    overridden for the entire test session.
    """
    # We no longer need to manage the lifespan manually, as the override
    # prevents the real model loading from ever being called in the tests.
    client = TestClient(app)

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
    assert response_predict.json() == {"predicted_species": 0}
