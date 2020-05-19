import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import Counter
import os,time,random,pickle

trainPath = './tcdata/hy_round1_train_20200102'
train2Path = '/tcdata/hy_round2_train_20200225'
test2Path = '/tcdata/hy_round2_testB_20200312'

lat0,lon0 = 21.67612*np.pi/180,60*np.pi/180

a,b = 6378137.0000,6356752.3142
e1,e2 = np.sqrt(1-(b/a)**2),np.sqrt((a/b)**2-1)

K = np.cos(lat0) * (a*a/b) / np.sqrt(1+e2**2*np.cos(lat0)**2)

def to_state_seq(data):
    data['time'] = pd.to_datetime(data['time'], format='%m%d %H:%M:%S').map(lambda x: (x-pd.to_datetime('19000101')).total_seconds()).values
    if 'x' not in data.columns:
        lon,lat = data['lon']*np.pi/180.0,data['lat']*np.pi/180.0
        data['x'] = K*(lon-lon0) + 129333.08761032454
        data['y'] = K*np.log( np.tan(np.pi/4 + lat/2) * ((1-e1*np.sin(lat)) / (1+e1*np.sin(lat)))**(e1/2) )+ 2587634.124389379
    state = data[['x','y','速度','方向','time']].values
    return state

train2Out = []

for i,file in enumerate(tqdm(os.listdir(trainPath))):
    data = pd.read_csv(os.path.join(trainPath, file))

    train2Out.append( to_state_seq(data) )

for i,file in enumerate(tqdm(os.listdir(train2Path))):
    data = pd.read_csv(os.path.join(train2Path, file))
    # fill zero
    lat,lon = data['lat'],data['lon']
    lat[lat==0],lon[lon==0] = np.nan,np.nan
    lat,lon = lat.fillna(method='bfill').fillna(method='ffill'),lon.fillna(method='bfill').fillna(method='ffill')

    train2Out.append( to_state_seq(data) )

with open('./data/nnTrain2.pkl', 'wb') as f:
    pickle.dump(train2Out, f)

test2Out = []
for i,file in enumerate(tqdm(os.listdir(test2Path))):
    data = pd.read_csv(os.path.join(test2Path, file))
    # fill zero
    lat,lon = data['lat'],data['lon']
    lat[lat==0],lon[lon==0] = np.nan,np.nan
    lat,lon = lat.fillna(method='bfill').fillna(method='ffill'),lon.fillna(method='bfill').fillna(method='ffill')

    test2Out.append( to_state_seq(data) )

with open('./data/nnTest2.pkl', 'wb') as f:
    pickle.dump(test2Out, f)

label = []
for i,file in enumerate(tqdm(os.listdir(trainPath))):
    data = pd.read_csv(os.path.join(trainPath, file))
    label.append( data['type'][0] )
for i,file in enumerate(tqdm(os.listdir(train2Path))):
    data = pd.read_csv(os.path.join(train2Path, file))
    label.append( data['type'][0] )
with open('./data/nnLabel2.pkl', 'wb') as f:
    pickle.dump(label, f)