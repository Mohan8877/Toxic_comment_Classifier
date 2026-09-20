from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_predict_endpoint():
    r = client.post('/predict', json={'comment': 'You are a great person'})
    assert r.status_code == 200
    data = r.json()
    assert 'is_toxic' in data
    assert 'categories' in data
