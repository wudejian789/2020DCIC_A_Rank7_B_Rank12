import pickle,os
import numpy as np
import pandas as pd
from lib.MarineTargetsAnalyze import *
from collections import Counter

ensemblePre = []
map_location = torch.device('cpu')

# lgb
with open('./model/lgb/cv10_lgb.pkl', 'rb') as f:
    lgbs = pickle.load(f)
X_t = pd.read_csv('./data/lgbTest2.csv')

useCols = ['lat_mean', 'lat_min', 'lat_25', 'lat_50', 'lat_75',
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

X_t = X_t[useCols]
for lgb in lgbs:
    ensemblePre.append(lgb.predict(X_t))

# nn
rootPath = './model/nn/'
with open('./data/nnTest2.pkl', 'rb') as f:
    X_t = pickle.load(f)
for path in os.listdir(rootPath):
    nn = StateAnalyzer(f'{rootPath}{path}', xydiff=500, feaSize=224, filterNum=256)
    ensemblePre.append(nn.predict(X_t))

# vote
ensemblePre = np.array(ensemblePre).swapaxes(0,1).swapaxes(1,2)

res = []
for pre in ensemblePre:
    lab = pre.argmax(axis=0)
    ensembleCounter = Counter(lab).most_common(3)
    if True:#len(ensembleCounter)>1 and ensembleCounter[0][1] == ensembleCounter[1][1]:
        res.append(pre.mean(axis=1).argmax())
    else:
        res.append(ensembleCounter[0][0])
ensembleRes = np.array(res)


test2Path = '/tcdata/hy_round2_testB_20200312'

# output results.csv
id2lab = ['拖网', '围网', '刺网']
with open('result.csv','w', encoding='utf8') as f:
    for i,lab in zip(os.listdir(test2Path),ensembleRes):
        f.write(f'{i[:-4]},{id2lab[lab]}\n')
print(Counter(ensembleRes))