import pytest

@pytest.fixture(scope="session", autouse=True)
def mock_model_load_on_startup(session_mocker):
    """
    This fixture is automatically applied to the entire test session.
    It mocks the MLflow model loading function before any tests are run,
    preventing any real network calls during test collection and execution.
    """
    class FakeModel:
        def predict(self, data):
            # Return a predictable, dummy prediction
            return [0]

    # Use session_mocker to patch the function where it is used.
    # This patch will persist for the whole test session.
    session_mocker.patch("src.main.mlflow.pyfunc.load_model", return_value=FakeModel())
