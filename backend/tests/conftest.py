import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    # Using TestClient as a context manager runs the FastAPI lifespan, so the
    # real face-mesh model is loaded once for the whole test module - the
    # same "load once, reuse" behaviour as production.
    with TestClient(app) as c:
        yield c
