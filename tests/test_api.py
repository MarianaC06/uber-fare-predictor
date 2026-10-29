from fastapi.testclient import TestClient
from app.main import app
client=TestClient(app)
BASE={'pickup_datetime':'2015-05-07T19:52:00Z','pickup_latitude':40.738354,'pickup_longitude':-73.999817,'dropoff_latitude':40.723217,'dropoff_longitude':-73.999512,'passenger_count':1}
def test_health():assert client.get('/health').json()['status']=='ok'
def test_home():assert client.get('/').status_code==200
def test_predict():
    r=client.post('/predict',json=BASE);assert r.status_code==200;assert r.json()['estimated_fare_usd']>=0
def test_invalid_passengers():assert client.post('/predict',json={**BASE,'passenger_count':0}).status_code==422
def test_invalid_latitude():assert client.post('/predict',json={**BASE,'pickup_latitude':0}).status_code==422
def test_same_location():assert client.post('/predict',json={**BASE,'dropoff_latitude':BASE['pickup_latitude'],'dropoff_longitude':BASE['pickup_longitude']}).status_code==422
def test_missing_field():
    d=BASE.copy();d.pop('pickup_latitude');assert client.post('/predict',json=d).status_code==422
