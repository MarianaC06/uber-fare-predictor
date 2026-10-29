from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parent
FEATURES=['distance_km','pickup_hour','pickup_dayofweek','pickup_year','passenger_count','pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude']
def features(df):
    df=df.copy()
    dt=pd.to_datetime(df['pickup_datetime'],utc=True,errors='coerce')
    a=np.radians(df['pickup_latitude']);b=np.radians(df['dropoff_latitude']);dl=np.radians(df['dropoff_latitude']-df['pickup_latitude']);dn=np.radians(df['dropoff_longitude']-df['pickup_longitude'])
    h=np.sin(dl/2)**2+np.cos(a)*np.cos(b)*np.sin(dn/2)**2
    df['distance_km']=6371*2*np.arcsin(np.sqrt(np.clip(h,0,1)))
    df['pickup_hour']=dt.dt.hour;df['pickup_dayofweek']=dt.dt.dayofweek;df['pickup_year']=dt.dt.year
    return df

def train(csv_path):
    df=pd.read_csv(csv_path)
    df=features(df)
    valid=(df.fare_amount.between(2.5,200)&df.passenger_count.between(1,6)&df.pickup_latitude.between(40,42)&df.dropoff_latitude.between(40,42)&df.pickup_longitude.between(-75,-72)&df.dropoff_longitude.between(-75,-72)&df.distance_km.between(0.1,100)&df[FEATURES].notna().all(axis=1))
    df=df.loc[valid]
    X_train,X_test,y_train,y_test=train_test_split(df[FEATURES],df.fare_amount,test_size=.2,random_state=42)
    model=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=25,random_state=42)
    model.fit(X_train,y_train);pred=model.predict(X_test)
    metrics={'total_rows':int(len(valid)),'clean_rows':int(len(df)),'train_rows':len(X_train),'test_rows':len(X_test),'mae_usd':round(float(mean_absolute_error(y_test,pred)),3),'rmse_usd':round(float(np.sqrt(mean_squared_error(y_test,pred))),3),'r2':round(float(r2_score(y_test,pred)),4)}
    ROOT.joinpath('model').mkdir(exist_ok=True)
    joblib.dump(model,ROOT/'model'/'uber_model.joblib')
    (ROOT/'model'/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/uber.csv');train(p.parse_args().data)
