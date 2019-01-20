# -*- coding: utf-8 -*-
"""
Created on Sun Dec 23 23:25:12 2018

@author: DJF
"""
import time
import numpy  as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm

#start time
St = time.time()

#read data from csv file
path = ''

#1. fund return
fr_test  = pd.read_csv(path + 'test_fund_return.csv')
fr_train = pd.read_csv(path + 'train_fund_return.csv')
fr = pd.merge(fr_train, fr_test, how='left')
fr = pd.concat([fr["Unnamed: 0"], fr.drop(columns = "Unnamed: 0")*10000], axis=1)

#2. fund benchmark return
fbr_test  = pd.read_csv(path + 'test_fund_return.csv')
fbr_train = pd.read_csv(path + 'train_fund_return.csv')
fbr = pd.merge(fbr_train, fbr_test, how='left')
fbr = pd.concat([fbr["Unnamed: 0"], fbr.drop(columns = "Unnamed: 0")*10000], axis=1)

#3. index return
ir_test  = pd.read_csv(path + 'test_index_return.csv', encoding='GBK', index_col=0)
ir_train =  pd.read_csv(path + 'train_index_return.csv', encoding='GBK', index_col=0) 
ir = pd.concat([ir_train, ir_test], axis=1)*10000

#4. correlation
correlation_test  = pd.read_csv(path + 'test_correlation.csv')
correlation_train = pd.read_csv(path + 'train_correlation.csv')
correlation = pd.merge(correlation_train, correlation_test, how='left')
correlation = pd.concat([correlation["Unnamed: 0"], correlation.drop(columns="Unnamed: 0")*10000], axis=1)

#training target: correlation between fundA ans fundB
ID = correlation["Unnamed: 0"]
ID = pd.concat([ID.map(lambda x:x.split('-')[0]), ID.map(lambda x:x.split('-')[1])], axis=1)
ID.columns = ['fundA', 'fundB']

#model evaluate 
from sklearn.metrics import mean_absolute_error
def ModelEva(label, y):
    MAE = mean_absolute_error(label, y)
    TMAPE = cnt = 0
    for i in range(len(label)):
        TMAPE += abs((y[i] - label[i])/(1.5 - label[i]))
        cnt   += 1
    TMAPE = TMAPE / cnt
    score = (2 / (2 + MAE + TMAPE))**2
    print("Model Score: ", score)
    return score

#construct 1st layer train set
##data from t1 to t2
def TrainData(dataset, t1, t2): #(DataFrame, column time, column time)
    data = pd.concat([dataset[dataset.columns[0]], dataset[dataset.columns[t1:t2]]], axis=1)
    #data1: fundA col
    data.rename(columns={data.columns[0]:"fundA"}, inplace=True)
    data1 = pd.merge(ID, data, how='left')
    data1 = data1[data1.columns[2:]]
    data1.columns = range(0, data1.shape[1])
    #data2: fundB col
    data.rename(columns={data.columns[0]:"fundB"}, inplace=True)
    data2 = pd.merge(ID, data, on='fundB', how='left')
    data2 = data2[data2.columns[2:]]
    data2.columns = range(0, data2.shape[1])
    return data1, data2

##feature project
###construct feature for training set using data from t1 to t2
def TrainFeature(t1, t2):
    #feature 1: fundA-fundB correlation
    data1, data2 = TrainData(fr, t1, t2)
    fr_cor = data1.corrwith(data2, axis=1)
    #feature 2: fundA-fundB benchmark return correlation
    data1, data2 = TrainData(fbr, t1, t2)
    fbr_cor = data1.corrwith(data2, axis=1)
    
    return np.vstack([fr_cor, fbr_cor]).T

###stack feature from t1 to t2 many times to enlarge train set
def StackFeature(t1, t2, times):
    #feature 3: fundA-fundB test-correlation mean and quantile
    data = correlation.drop(columns="Unnamed: 0")
    fea1 = data.mean(axis=1)
    fea2 = data.quantile(0.25, axis=1)
    fea3 = data.quantile(0.5 , axis=1)
    fea4 = data.quantile(0.75, axis=1)
    f2 = np.vstack([fea1, fea2, fea3, fea4]).T
    for i in tqdm(range(times)):
        #f1 = TrainFeature(t1 - 20*i, t2)
        if i == 0:
            f1 = TrainFeature(t1, t2)
            xtrain = np.hstack([f2, f1])
        else:
            f1 = TrainFeature(t1 - 20*(i+1), t2)
            xtrain = np.hstack([xtrain, f1])
    return xtrain

###train set
for i in range(15):
    tsetx = StackFeature(-82, -62, 20)
    tsety = correlation[correlation.columns[-2-i]]
    if i == 0:
        xtrain = tsetx
        ytrain = tsety
    else:
        xtrain = np.vstack([xtrain, tsetx])
        ytrain = np.hstack([ytrain, tsety])

#validation set
xval = StackFeature(-81, -61, 20)
yval = correlation[correlation.columns[-1]]

#test set for predict
xtest = StackFeature(-20, None, 20)

#Model
def XGB(xtrain, label, val, xtest, params):
    trainM = xgb.DMatrix(np.array(xtrain), label)
    valM   = xgb.DMatrix(np.array(val))
    testM  = xgb.DMatrix(np.array(xtest))
    
    model = xgb.train(params, trainM, params['nrounds'])
    return model.predict(valM), model.predict(testM)

def LGB(xtrain, label, val, xtest, params):
    trainM = lgb.Dataset(np.array(xtrain), label)
    
    model = lgb.train(params, trainM, params['nrounds'])
    return model.predict(val), model.predict(xtest)

##lgb
lgb_params = {
    'application':'regression_l1',
    'metric':'mae',
    'seed': 0,
    'learning_rate':0.04,
    'max_depth':1,
    'feature_fraction':0.7,
    'lambda_l1':2,
    'nrounds':900
}
lgbval, lgby = LGB(xtrain, ytrain, xval, xtest, lgb_params)

##xgb
xgb_params = {
    'objective':'reg:linear',
    'learning_rate':0.3,
    'max_depth':1,
    'subsample':1,
    'colsample_bytree':0.06,
    'alpha':50,
    'lambda':5,
    'nrounds':1800
}
###Naive 5-fold
rows = xtrain.shape[0]
piece = int(rows/5)
xtrain_1 = xtrain[0:(piece)*4]
ytrain_1 = ytrain[0:(piece)*4]
xval_1   = xtrain[(piece)*4:]

xtrain_2 = np.vstack([xtrain[0:(piece)*3], xtrain[(piece)*4:]])
ytrain_2 = np.hstack([ytrain[0:(piece)*3], ytrain[(piece)*4:]])
xval_2   = xtrain[(piece)*3:(piece)*4]

xtrain_3 = np.vstack([xtrain[0:(piece)*2], xtrain[(piece)*3:]])
ytrain_3 = np.hstack([ytrain[0:(piece)*2], ytrain[(piece)*3:]])
xval_3   = xtrain[(piece)*2:(piece)*3]

xtrain_4 = np.vstack([xtrain[0:(piece)*1], xtrain[(piece)*2:]])
ytrain_4 = np.hstack([ytrain[0:(piece)*1], ytrain[(piece)*2:]])
xval_4   = xtrain[(piece)*1:(piece)*2]

xtrain_5 = xtrain[(piece)*1:]
ytrain_5 = ytrain[(piece)*1:]
xval_5   = xtrain[0:(piece)*1]

xgbval1, xgby1 = XGB(xtrain_1, ytrain_1, xval, xtest, xgb_params)
xgbval2, xgby2 = XGB(xtrain_2, ytrain_2, xval, xtest, xgb_params)
xgbval3, xgby3 = XGB(xtrain_3, ytrain_3, xval, xtest, xgb_params)
xgbval4, xgby4 = XGB(xtrain_4, ytrain_4, xval, xtest, xgb_params)
xgbval5, xgby5 = XGB(xtrain_5, ytrain_5, xval, xtest, xgb_params)

xgby = np.vstack([xgby1, xgby2, xgby3, xgby4, xgby5])
print(np.shape(xgby))
xgby = np.mean(xgby, axis=0)
print(np.shape(xgby))
xgbval = np.vstack([xgbval1, xgbval2, xgbval3, xgbval4, xgbval5])
xgbval = np.mean(xgbval, axis=0)
print(np.shape(xgbval))

#second layer
##second layer train set
xtrain2 = np.vstack([xgbval, lgbval])
date = [5, 30, 60, 90]
for i in tqdm(date):
    data1, data2 = TrainData(fr, -61-i, -61)
    #feature: fundA-fundB sum distance
    fea = abs(data1 - data2).sum(axis=1)
    xtrain2 = np.vstack([xtrain2, fea])
xtrain2 = xtrain2.T
ytrain2 = correlation[correlation.columns[-1]]

##second layer test set for predict
xtest2  = np.vstack([lgby, lgby])
for i in tqdm(date):
    data1, data2 = TrainData(fr, -i, None)
    #feature: fundA-fundB sum distance
    fea = abs(data1 - data2).sum(axis=1)
    xtest2 = np.vstack([xtest2, fea])
xtest2 = xtest2.T

#lgb train
lgbs_params = {
    'application':'regression_l1',
    'seed':0,
    'learning_rate': 0.01,
    'max_depth':1,
    'feature_fraction':0.8,
    'nrounds':1800
}

yval2, ypredict = LGB(xtrain2, ytrain2, xtrain2, xtest2, lgbs_params)

#Model evaluate
ModelEva(yval/10000, yval2/10000)
ModelEva(yval/10000, ypredict/10000)

#save predict result
res = pd.DataFrame({"ID":correlation["Unnamed: 0"], "value":ypredict/10000})
res.to_csv("MLP.csv", index=None)

#train finish
Ed = time.time()
print("The train takes " + str(Ed - St) + "second")