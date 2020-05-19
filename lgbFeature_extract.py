import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import Counter
import os,time,random
from scipy import stats

import warnings
warnings.filterwarnings("ignore")

train2Path = '/tcdata/hy_round2_train_20200225'
test2Path = '/tcdata/hy_round2_testB_20200312'

def convex_hull(points, plot=False):
    if plot:
        plt.figure(figsize=(12,8))
        x,y = points[:,0],points[:,1]
        plt.plot( x,y,'.' )
        plt.plot( x[-1],y[-1], 'r*' )
        plt.plot( x[0],y[0],'g*' )
    # find point with min y
    p0 = points[points.argmin(axis=0)[1]]
    # sort points by radian and distance
    vectors = points-p0
    radians = np.arctan( vectors[:,1] / (vectors[:,0]+1e-10) )
    radians[radians<0] += np.pi
    distances = np.sqrt((vectors**2).sum(axis=1))
    tmp = np.array([(r,d) for r,d in zip(radians,distances)], dtype=[('radian',float),('distance',float)])
    indices = np.argsort(tmp, order=['radian','distance'])
    points = points[indices]
    # graham algorithm
    stack,i = [],0
    while i<len(points):
        if len(stack)<2:
            stack.append(i)
            i += 1
            continue
        p0,p1,p2 = points[stack[-2]],points[stack[-1]],points[i]
        if (p1[0]-p0[0])*(p2[1]-p1[1]) - (p1[1]-p0[1])*(p2[0]-p1[0])>0:
            stack.append(i)
            i += 1
        else:
            stack.pop()
    stack.append(stack[0])
    points = points[stack]
    if plot:
        plt.plot( points[:,0],points[:,1],  )
    # calculate area
    tmp = points[1:-1] - points[0]
    a,b = tmp[:-1],tmp[1:]
    area = (a[:,0]*b[:,1] - a[:,1]*b[:,0]).sum()/2
    return area, len(points)
def skew_kurt(x):
    mean = np.mean(x)
    var = np.var(x)
    return np.mean((x-mean)**3),np.mean((x-mean)**4/(var**2+1e-10))
def bins_fea(data, x, num, name, includeLeft=False, xmin=None, xmax=None):
    xmin,xmax = x.min() if xmin is None else xmin,x.max() if xmax is None else xmax
    bins = [xmin+i*(xmax-xmin)/num for i in range(num+1)]
    if includeLeft:
        bins[0] -= 1.0
    for i in range(1,num+1):
        data[f'{name}{i}'] = ((bins[i-1]<x)&(x<=bins[i])).sum() / len(x)
    return data
def time_bins_fea(data, x, time, num, name):
    timemin,timemax = time.min(),time.max()
    bins = [timemin+i*(timemax-timemin)/num for i in range(num+1)]
    bins[0] -= 1.0
    for i in range(1,num+1):
        data[f'{name}{i}'] = x[(bins[i-1]<time)&(time<=bins[i])].mean()
    return data
def work_duration(data):
    t = 0
    v,diff_time = data['速度'],data['time'].diff(-1).values
    for i in range(1,len(data)):
        tmp = 0
        if v[i-1]>0: tmp += 1
        if v[i]>0: tmp += 1
        t += diff_time[i-1]*tmp/2
    return t
def mode(x):
    return stats.mode(x)[0][0]

cols = ['label', 'lat_mean', 'lat_min', 'lat_25', 'lat_50', 'lat_75',
       'lat_max', 'lon_mode', 'lat_mode', 'x_max_y_min', 'ru_mean',
       'thetau_mean', 'ru_min', 'ru_25', 'ru_50', 'ru_75', 'ru_max',
       'thetau_min', 'thetau_25', 'thetau_50', 'thetau_75', 'thetau_max',
       'ru_mode', 'thetau_mode', 'rd_mean', 'thetad_mean', 'rd_min',
       'rd_25', 'rd_50', 'rd_75', 'rd_max', 'thetad_min', 'thetad_25',
       'thetad_50', 'thetad_75', 'thetad_max', 'rd_mode', 'thetad_mode',
       'r0_mean', 'r0_min', 'r0_25', 'r0_50', 'r0_75', 'r0_max',
       'r0_max_min', 'r0_75_25', 'theta0_std', 'dist_mean', 'dist_min',
       'dist_25', 'dist_50', 'dist_75', 'dist_max', 'dist_std',
       'dist_mode', 'dist_port0', 'dist_port1', 'dist_port2',
       'dist_port3', 'dist_port4', 'dist_port5', 'dist_port6',
       'dist_port7', 'dist_port8', 'dist_port9', 'dist_port10',
       'dist_port11', 'dist_port12', 'dist_port13', 'dist_port14',
       'dist_port15', 'dist_port16', 'dist_port17', 'beginHour',
       'endHour', 'hour_mode', 'workDuration', 'workDuration_ratio',
       'convexArea', 'valid_num', 'v_mean', 'v_min', 'v_25', 'v_50',
       'v_75', 'v_max', 'v_std', 'v_max_min', 'v_75_25', 'v_skew',
       'v_kurt', 'v_mode', 'count_v1', 'count_v2', 'count_v3', 'count_v4',
       'count_v5', 'count_v6', 'count_v7', 'count_v8', 'count_v9',
       'count_v10', 'count_vn', 'total_v_mean', 'count_a_', 'count_a1',
       'count_a2', 'count_a3', 'count_a4', 'count_a5', 'count_a6',
       'count_an', 'tdv_mean', 'tdv_min', 'tdv_25', 'tdv_50', 'tdv_75',
       'tdv_max', 'tdv_max_min', 'tdv_75_25', 'tdv_std', 'tdv_skew',
       'tdv_kurt', 'tdv_mode', 'v_tdv_pearson', 'rho_mean', 'rho_min',
       'rho_25', 'rho_50', 'rho_75', 'rho_max', 'rho_std', 'rho_mode',
       'count_rho1', 'count_rho2', 'count_rho3', 'count_rho4',
       'count_rho5', 'count_rhon']

xmin, ymin = 3282308.716983761, 764432.0618863769
xmax, ymax = 6030355.254537644, 5553070.539537183
v_min,v_max = 0,10
a_min,a_max = -0.1, 0.5
rho_min,rho_max = 0,50
def coastline(x):
    return 5.41667713e-13*x**3 - 6.98254368e-06*x**2 + 3.03530663e+01*x - 4.20991085e+07
coastline = np.array( [[x-1,coastline(x)+1] for x in np.arange(xmin, xmax, 200)] ).reshape(1,-1,2)
def dist_to_coastline(points):
    points = points.reshape(-1,1,2)
    return np.sqrt(((points-coastline)**2).sum(axis=2)).min(axis=1)

def dfm_to_num(jwd):
    s = 0
    d,f,m = 0,0,0
    if "°" in jwd:
        d = jwd[s:jwd.find("°")]
        s = jwd.find("°")+1
    if "′" in jwd:
        f = jwd[s:jwd.find("′")]
        s = jwd.find("′")+1
    if "″" in jwd:
        s = jwd[s:jwd.find("″")]
        s = jwd.find("″")+1
    return float(d) + float(f)/60 + float(m)/3600

lat0,lon0 = 4.0*np.pi/180,73.5*np.pi/180

a,b = 6378137.0000,6356752.3142
e1,e2 = np.sqrt(1-(b/a)**2),np.sqrt((a/b)**2-1)

K = np.cos(lat0) * (a*a/b) / np.sqrt(1+e2**2*np.cos(lat0)**2)

port = np.array(
       [[dfm_to_num("119°00′"), dfm_to_num("39°12′")], 
        [dfm_to_num("119°27′"), dfm_to_num("34°44′")],
        [dfm_to_num("120°59′"), dfm_to_num("40°43′")],
        [dfm_to_num("117°23′"), dfm_to_num("38°26′")],
        [dfm_to_num("113°36′"), dfm_to_num("23°6′")],
        [dfm_to_num("120°16′"), dfm_to_num("22°37′")],
        [dfm_to_num("121°55′"), dfm_to_num("29°59′")],
        [dfm_to_num("113°33′"), dfm_to_num("22°12′")],
        [dfm_to_num("119°00′"), dfm_to_num("39°12′")],
        [dfm_to_num("120°19′5″"), dfm_to_num("36°04′00″")],
        [dfm_to_num("121°51′05″"), dfm_to_num("29°56′28″")],
        [dfm_to_num("117°45′"), dfm_to_num("38°59′")],
        [dfm_to_num("121°29′4.5″"), dfm_to_num("31°14′18.8″")],
        [dfm_to_num("118°04′15″"), dfm_to_num("24°29′20″")],
        [dfm_to_num("114°03′10.4″"), dfm_to_num("22°32′43.86″")],
        [dfm_to_num("121°39′"), dfm_to_num("38°55′")],
        [dfm_to_num("122°13′48.89″"), dfm_to_num("40°39′55.62″")],
        [dfm_to_num("119°38′9.39″"), dfm_to_num("39°55′39.23″")]]
)
lon,lat = port[:,0]*np.pi/180,port[:,1]*np.pi/180
port[:,0] = K*(lon-lon0)
port[:,1] = K*np.log( np.tan(np.pi/4 + lat/2) * ((1-e1*np.sin(lat)) / (1+e1*np.sin(lat)))**(e1/2) )
port = port.reshape(1,-1,2)

def fea_extract(data):
    data['time'] = pd.to_datetime(data['time'], format='%m%d %H:%M:%S').map(lambda x: (x-pd.to_datetime('19000101')).total_seconds()).values
    out = pd.DataFrame(-np.ones((1,len(cols))), columns=cols)
    # label
    try:
        out['label'] = data['type'][0]
    except:
        pass
    
    lat,lon = data['lat']*np.pi/180,data['lon']*np.pi/180
    data['x'] = K*(lon-lon0)
    data['y'] = K*np.log( np.tan(np.pi/4 + lat/2) * ((1-e1*np.sin(lat)) / (1+e1*np.sin(lat)))**(e1/2) )
    
    difftime = data['time'].diff(-1).values[:-1]
    
    # location
    x,y = data['lon'],data['lat']
    out['lat_mean'] = data['lat'].mean()
    out['lat_min'],out['lat_25'],out['lat_50'],out['lat_75'],out['lat_max'] = y.min(),np.quantile(y,0.25),np.median(y),np.quantile(y,0.75),y.max()
    out['lon_mode'],out['lat_mode'] = mode(data['lon']),mode(data['lat'])
    
    out['x_max_y_min']= data['x'].max()-data['y'].min()
    
    x,y = data['x']-xmin+1,ymax-data['y']+1
    r,theta = np.sqrt(x**2 + y**2),np.arctan(y/x)
    out['ru_mean'],out['thetau_mean'] = r.mean(),theta.mean()
    out['ru_min'],out['ru_25'],out['ru_50'],out['ru_75'],out['ru_max'] = r.min(),np.quantile(r,0.25),np.median(r),np.quantile(r,0.75),r.max()
    out['thetau_min'],out['thetau_25'],out['thetau_50'],out['thetau_75'],out['thetau_max'] = theta.min(),np.quantile(theta,0.25),np.median(theta),np.quantile(theta,0.75),theta.max()
    out['ru_mode'],out['thetau_mode'] = mode(r),mode(theta)
    
    x,y = data['x']-xmin+1,data['y']-ymin+1
    r,theta = np.sqrt(x**2 + y**2),np.arctan(y/x)
    out['rd_mean'],out['thetad_mean'] = r.mean(),theta.mean()
    out['rd_min'],out['rd_25'],out['rd_50'],out['rd_75'],out['rd_max'] = r.min(),np.quantile(r,0.25),np.median(r),np.quantile(r,0.75),r.max()
    out['thetad_min'],out['thetad_25'],out['thetad_50'],out['thetad_75'],out['thetad_max'] = theta.min(),np.quantile(theta,0.25),np.median(theta),np.quantile(theta,0.75),theta.max()
    out['rd_mode'],out['thetad_mode'] = mode(r),mode(theta)
    
    x,y = data['x']-(data['x'].min()+data['x'].max())/2+1,data['y']-(data['y'].min()+data['y'].max())/2+1
    r,theta = np.sqrt(x**2 + y**2),np.arctan(y/x)
    out['r0_mean'] = r.mean()
    out['r0_min'],out['r0_25'],out['r0_50'],out['r0_75'],out['r0_max'] = r.min(),np.quantile(r,0.25),np.median(r),np.quantile(r,0.75),r.max()
    out['r0_max_min'] = out['r0_max']-out['r0_min']
    out['r0_75_25'] = out['r0_75']-out['r0_25']
    out['theta0_std'] = theta.std()
    
    # coastline distance
    dist = dist_to_coastline(data[['x','y']].values)
    out['dist_mean'] = dist.mean()
    out['dist_min'],out['dist_25'],out['dist_50'],out['dist_75'],out['dist_max'] = dist.min(),np.quantile(dist,0.25),np.median(dist),np.quantile(dist,0.75),dist.max()
    out['dist_std'] = dist.std()
    out['dist_mode'] = mode(dist)
    
    # port distance
    dist_port = np.sqrt( ((data[['x','y']].values.reshape(-1,1,2) - port)**2).sum(axis=2) ).mean(axis=0)
    for i in range(len(port[0])):
        out[f"dist_port{i}"] = dist_port[i]
    
    # time
    totalTime =  data['time'].values[0] - data['time'].values[-1]
    hour = ((data['time']%(3600*24))//3600).values.astype('int')
    out['beginHour'],out['endHour'] = hour[-1],hour[0]
    out['hour_mode'] = mode(hour)
    out['workDuration'] = work_duration(data)
    out['workDuration_ratio'] = out['workDuration'] / totalTime
    
    # area
    out['convexArea'],_ = convex_hull(data[['x','y']].values)
    
    # keep valid data
    #data = data[data['速度']>0]
    isValid = data['速度']>0
    out['valid_num'] = isValid.sum()
    if len(data)>2:
        # velocity
        v = data['速度']
        out['v_mean'] = data['速度'].mean()
        out['v_min'],out['v_25'],out['v_50'],out['v_75'],out['v_max'] = v.min(),np.quantile(v,0.25),np.median(v),np.quantile(v,0.75),v.max()
        out['v_std'] = data['速度'].std()
        out['v_max_min'] = out['v_max']-out['v_min']
        out['v_75_25'] = out['v_75']-out['v_25']
        out['v_skew'],out['v_kurt'] = skew_kurt(data['速度'])
        out['v_mode'] = mode(v)
        out = bins_fea(out, data['速度'], 10, name='count_v', includeLeft=True, xmin=v_min, xmax=v_max)
        out['count_vn'] = (data['速度']>10).sum() / len(data)
        
        # total distance and mean velocity
        total_dist = np.sqrt((data[['x','y']].diff(-1).values[:-1]**2).sum(axis=1)).sum()
        out['total_v_mean'] = total_dist / totalTime
        # acceleration
        a = data['速度'].diff(-1).values[:-1] / difftime
        out['count_a_'] = (a<=a_min).sum() / len(data)
        out = bins_fea(out, a, 6, name='count_a', includeLeft=False, xmin=a_min, xmax=a_max)
        out['count_an'] = (a>a_max).sum() / len(data)
        # turn direction velocity
        tdv = data['方向'].diff(-1).values[:-1] / difftime
        out['tdv_mean'] = tdv.mean()
        out['tdv_min'],out['tdv_25'],out['tdv_50'],out['tdv_75'],out['tdv_max'] = tdv.min(),np.quantile(tdv,0.25),np.median(tdv),np.quantile(tdv,0.75),tdv.max()
        out['tdv_max_min'] = out['tdv_max']-out['tdv_min']
        out['tdv_75_25'] = out['tdv_75']-out['tdv_25']
        out['tdv_std'] = tdv.std()
        out['tdv_skew'],out['tdv_kurt'] = skew_kurt(tdv)
        out['tdv_mode'] = mode(tdv)
        
        out['v_tdv_pearson'] = np.corrcoef((v.values[1:]+v.values[:-1])/2,tdv)[0,1]
        # curvature
        k = ((data['y'].diff(-1))/(data['x'].diff(-1) + 1e-10)).values[:-1] + 1e-10
        x0,y0 = (data['x'].values[1:] + data['x'].values[:-1])/2,(data['y'].values[1:] + data['y'].values[:-1])/2
        xo = (x0[1:]/(k[1:]+1e-10) - x0[:-1]/(k[:-1]+1e-10) + y0[1:] - y0[:-1]) / (1/k[1:] - 1/k[:-1] + 1e-10)
        yo = -1/(k[:-1]+1e-10) * (xo - x0[:-1]) + y0[:-1]

        rho = np.log2(np.sqrt((data['y'][1:-1]-yo)**2 + (data['x'][1:-1]-xo)**2) + 1e-10)
        out['rho_mean'] = rho.mean()
        out['rho_min'],out['rho_25'],out['rho_50'],out['rho_75'],out['rho_max'] = rho.min(),np.quantile(rho,0.25),np.median(rho),np.quantile(rho,0.75),rho.max()
        out['rho_std'] = rho.std()
        out['rho_mode'] = mode(rho)
        out = bins_fea(out, rho, 5, name='count_rho', includeLeft=True, xmin=rho_min, xmax=rho_max)
        out['count_rhon'] = (rho>rho_max).sum() / len(data)
        
    return out

train2Out = pd.DataFrame(columns=cols)
for i,file in enumerate(tqdm(os.listdir(train2Path))):
    data = pd.read_csv(os.path.join(train2Path, file))
    # fill zero
    lat,lon = data['lat'],data['lon']
    lat[lat==0],lon[lon==0] = np.nan,np.nan
    lat,lon = lat.fillna(method='bfill').fillna(method='ffill'),lon.fillna(method='bfill').fillna(method='ffill')
    data['lat'],data['lon'] = lat,lon
    
    train2Out = train2Out.append( fea_extract(data),sort=False )
train2Out.to_csv('./data/lgbTrain2.csv', index=None)

test2Out = pd.DataFrame(columns=cols)
for i,file in enumerate(tqdm(os.listdir(test2Path))):
    data = pd.read_csv(os.path.join(test2Path, file))
    # fill zero
    lat,lon = data['lat'],data['lon']
    lat[lat==0],lon[lon==0] = np.nan,np.nan
    lat,lon = lat.fillna(method='bfill').fillna(method='ffill'),lon.fillna(method='bfill').fillna(method='ffill')
    data['lat'],data['lon'] = lat,lon
    
    test2Out = test2Out.append( fea_extract(data),sort=False )

test2Out.to_csv('./data/lgbTest2.csv', index=None)