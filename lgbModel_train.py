import pandas as pd
import numpy as np

lab2id = {'拖网':0, '围网':1, '刺网':2}
id2lab = ['拖网', '围网', '刺网']
train = pd.read_csv('./data/lgbTrain2.csv')
X,Y = train.drop('label', axis=1),train['label'].map(lab2id)

import lightgbm as lgb
from lib.metrics import *
from sklearn.model_selection import KFold,StratifiedKFold,train_test_split
from collections import Counter
from matplotlib import pyplot as plt
from  scipy.stats import chi2_contingency
import pickle,torch,random,os

seed = 20200315
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

from lib.metrics import *
params = { 'boosting_type': 'gbdt',
           'objective': 'multiclass',
           'num_class': 3,
           'num_leaves': 30,
           'max_depth': -1,
           'min_data_in_leaf': 1,
           'learning_rate': 0.2,
           'is_unbalance': 'true',
           'metric':'None',
}

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

metrictor = Metrictor(3)
kf = StratifiedKFold(n_splits=10, random_state=seed, shuffle=True)
models = []
Fs = []

X,Y = train[list(useCols)],train['label'].map(lab2id)
for train_index, test_index in kf.split(X, Y):
    X_train, X_valid = X.values[train_index],X.values[test_index]
    Y_train, Y_valid = Y.values[train_index],Y.values[test_index]
    lgb_train = lgb.Dataset(X_train, Y_train)
    lgb_valid = lgb.Dataset(X_valid, Y_valid,reference=lgb_train)
    models.append(lgb.train(params, lgb_train, valid_sets=[lgb_valid], feval=lgb_MaF, 
                  num_boost_round=5000, early_stopping_rounds=500, verbose_eval=0))
    metrictor.set_data(models[-1].predict(X.iloc[test_index]), Y.iloc[test_index])
    res = metrictor(['ACC', 'MaF'])
    Fi = metrictor.each_class_indictor_show(id2lab)
    Fs.append(Fi[0])
s = []
for model in models:
    s.append( model.best_score['valid_0']['macro_f1'] )
print(s)
print(f'{np.mean(s):.3f} ± {np.std(s):.3f}')
print(np.mean(Fs))

with open('./model/lgb/cv10_lgb.pkl', 'wb') as f:
    pickle.dump(models, f)