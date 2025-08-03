from fastapi.testclient import TestClient
from src.main import app

# Create a test client for the FastAPI app
client = TestClient(app)

def test_read_root():
    """Test that the root endpoint returns a 200 OK status."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Iris Classifier API!"}