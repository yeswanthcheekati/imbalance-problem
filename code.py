# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 07:28:42 2018

@author: cheekati
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,precision_score,recall_score,f1_score
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,BaggingClassifier
from imblearn import over_sampling,under_sampling,combine
from sklearn.svm import SVC
import pickle
from sklearn.externals import joblib

pd.options.mode.chained_assignment = None

from sklearn.model_selection import train_test_split

train = pd.read_csv('train_ZoGVYWq.csv',index_col=0)
test = pd.read_csv('test_66516Ee.csv',index_col=0)

train['Income_per_age'] = train['Income']/train['age_in_days']
test['Income_per_age'] = test['Income']/test['age_in_days']

train['total_premium'] = train['premium']*train['no_of_premiums_paid']
test['total_premium'] = test['premium']*test['no_of_premiums_paid']

train['premium_cash_credit'] = train['total_premium']*train['perc_premium_paid_by_cash_credit']
test['premium_cash_credit'] = test['total_premium']*test['perc_premium_paid_by_cash_credit']

desc = train.describe().loc[['mean','std']]
desc.drop('renewal',inplace=True,axis=1)

#train
for col in desc.columns:
    mean = desc[col]['mean']
    std = desc[col]['std']
    train[col] = (train[col]-mean)/std
#test
for col in desc.columns:
    mean = desc[col]['mean']
    std = desc[col]['std']
    test[col] = (test[col]-mean)/std

unique_channel = list(sorted(set(train.sourcing_channel)))
unique_res = list(sorted(set(train.residence_area_type)))

def encode_feature(feature_values,unique_list,column):
    encoded_list = []
    index = feature_values.index
    feature_values = feature_values.values
    for element in feature_values:
        encoded_vec = [0]*len(unique_list)
        encoded_vec[unique_list.index(element)] = 1
        encoded_list.append(encoded_vec)
    return pd.DataFrame(encoded_list,index=index,columns=[column + element for element in unique_list])

#train
temp_channel = encode_feature(train.sourcing_channel,unique_channel,'sourcing_channel')
temp_res = encode_feature(train.residence_area_type,unique_res,'residence_area_type')
#test
test_temp_channel = encode_feature(test.sourcing_channel,unique_channel,'sourcing_channel')
test_temp_res = encode_feature(test.residence_area_type,unique_res,'residence_area_type')

#train
train.drop(['sourcing_channel','residence_area_type'],inplace=True,axis=1)
#test
test.drop(['sourcing_channel','residence_area_type'],inplace=True,axis=1)

#train
train = pd.concat([train,temp_channel,temp_res],axis=1)
del temp_channel,temp_res
#test
test = pd.concat([test,test_temp_channel,test_temp_res],axis=1)
del test_temp_channel,test_temp_res

train_nan = train[(train.isnull().sum(axis=1)>0).values]
#test
test_nan = test[(test.isnull().sum(axis=1)>0).values]

assert train.shape[0] == train_nan.shape[0] + train.dropna().shape[0]
assert test.shape[0] == test_nan.shape[0] + test.dropna().shape[0]

train.dropna(inplace=True)
test.dropna(inplace=True)

def impute(i,train,data_nan):
    row = data_nan.loc[i]
    temp_col = row[pd.isnull(row)==False].index
    fill_col = row[pd.isnull(row)==True].index
    mag = np.linalg.norm(train[temp_col].values - row[temp_col].values,axis=1)
    index = np.where(mag == mag.min())[0][0]
    id_ = train.iloc[index].name
    return fill_col,id_

#train
for i in train_nan.index:
    fill_col,id_ = impute(i,train,train_nan)
    train_nan.loc[i,fill_col] = train.loc[id_,fill_col]
#test
for i in test_nan.index:
    fill_col,id_ = impute(i,train,test_nan)
    test_nan.loc[i,fill_col] = train.loc[id_,fill_col]

train = pd.concat([train,train_nan],axis=0)
del train_nan
#test
test = pd.concat([test,test_nan],axis=0)
del test_nan

y = train['renewal']
x = train.drop('renewal',axis=1)

ros = over_sampling.ADASYN()
rus = under_sampling.NearMiss()
rcs = combine.SMOTEENN()       
rcs2 = combine.SMOTETomek()

log = BaggingClassifier(LogisticRegressionCV(Cs=6))
rf = BaggingClassifier(RandomForestClassifier())
gbc = BaggingClassifier(GradientBoostingClassifier(n_estimators=250,learning_rate=0.01))
sv = SVC(C=0.8,probability=True)
for sample,sample_name in zip([rcs2,ros,rus,rcs,rcs2],['rcs2','ros','rus','rcs']):
    print(sample_name)
    x_rs,y_rs = sample.fit_sample(x,y)
    for model,model_name in zip([log,rf,gbc],['log','rf','gbc']):
        model.fit(x_rs,y_rs)
        filename = 'C:/Users/cheekati/Desktop/ml/AV Mck/' + str(model_name) + str(sample_name) + '.pkl'
        f =  open(filename, 'wb')
        pickle.dump(model, f)
        print('model complete')
        
    