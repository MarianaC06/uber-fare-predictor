from pathlib import Path
from datetime import datetime
from typing import Annotated
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from train import features, FEATURES
ROOT=Path(__file__).resolve().parents[1]
app=FastAPI(title='Uber Fare Predictor API',version='1.0.0')
model=joblib.load(ROOT/'model'/'uber_model.joblib')
class Ride(BaseModel):
    pickup_datetime:datetime
    pickup_latitude:Annotated[float,Field(ge=40,le=42)]
    pickup_longitude:Annotated[float,Field(ge=-75,le=-72)]
    dropoff_latitude:Annotated[float,Field(ge=40,le=42)]
    dropoff_longitude:Annotated[float,Field(ge=-75,le=-72)]
    passenger_count:Annotated[int,Field(ge=1,le=6)]
@app.get('/')
def home():return FileResponse(ROOT/'app'/'static'/'index.html')
@app.get('/health')
def health():return {'status':'ok','model_loaded':True}
@app.post('/predict')
def predict(ride:Ride):
    d=ride.model_dump();d['pickup_datetime']=ride.pickup_datetime.isoformat()
    X=features(pd.DataFrame([d]));distance=float(X.distance_km.iloc[0])
    if not .1<=distance<=100:raise HTTPException(422,'La distancia debe estar entre 0.1 y 100 km')
    fare=max(0,float(model.predict(X[FEATURES])[0]))
    return {'estimated_fare_usd':round(fare,2),'distance_km':round(distance,2),'note':'Estimación académica basada en viajes históricos; no es una tarifa actual de Uber.'}
