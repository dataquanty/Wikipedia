#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 18:17:56 2017

@author: dataquanty
"""

import pandas as pd, numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import matplotlib.pyplot as plt
import random
from sklearn.externals import joblib


def SMAPE(y_true, y_pred): 
    
    return np.mean(SMAPEu(y_true,y_pred)) 

def SMAPEu(y_true,y_pred):
    y_true = y_true.astype('float')
    y_pred = y_pred.astype('float')
    return (np.abs(y_true - y_pred)) /((y_true+y_pred)/2 +((y_pred==0.0)&(y_true==0.0)*1))




mat = pd.read_csv('train_transf_v4.csv')
#mat = pd.read_csv('predict_transf_out.csv')

wikicrawl = pd.read_csv('train_wikicrawl_v2.csv')
mat = mat.merge(wikicrawl,how='left',left_on='page',right_on='Page')

#mat[np.isnan(mat['images'])].groupby('country').count()['page']/mat.groupby('country').count()['page']
mat.drop('Page',axis=1,inplace=True)
mat = mat.drop(['page'],axis=1)
mat['access'] = mat['access'].apply(lambda x: 0 if x == 'all-access' else 1 if x=='mobile-web' else 2)
mat['access0']=mat['access'].apply(lambda x: 1 if x==0 else 0)
mat['access1']=mat['access'].apply(lambda x : 1 if x==1 else 0)
mat.drop('access',axis=1,inplace=True)

mat['accesstype'] = mat['accesstype'].apply(lambda x : 0 if x == 'spider' else 1 )

countrlst = list(np.unique(mat['country']))
countrlst = ['co', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'ww', 'zh']
for c in countrlst[1:]:
    mat['countr_'+ c]=mat['country'].apply(lambda x : 1 if x==c else 0)

mat.drop('country',axis=1,inplace=True)
#mat = mat[(np.isnan(mat['images']))==False]

mat = mat.fillna(0)
mat = mat.replace(np.inf, 0)

#mat = mat[((mat['y']>10000) | (mat['y']<0) | (mat['median']<1))==False]
mat['id']=range(len(mat))
matres = np.array(mat['id'])

med = mat['median']
med = med.apply(lambda x: 1 if x==0 else x)

def sigmoid(x):
    return 1/(1+np.exp(-x))



X = mat.drop(['y','id'],axis=1)
Y = mat['y']/med
#Y = np.log(0.0001+Y)
#Y = mat['y']

#Y = mat[((mat['y']>10000) | (mat['y']<0))==False]['y']/med
#Y = np.log1p(Y)

#scaler = StandardScaler()
#scaler.fit(X)

#joblib.dump(scaler, 'sklean_scaler1.pkl',compress=True)
#X = scaler.transform(X)


for i in range(14,25):
    X, Y, med, matres = shuffle(X,Y,med, matres)
    #offset2 = 100
    #XKern = X[-offset2:]
    
    #gammarand = random.random()/2
    #X2 = rbf_kernel(X,XKern,gamma=gammarand)
    
    
    offset = int(X.shape[0] * 0.2)
    X_train, y_train = X[:offset], Y[:offset]
    X_test, y_test = X[offset:], Y[offset:]
    
    X_train, y_train, med_train = X[:offset], Y[:offset],med[:offset]
    X_test, y_test,med_test = X[offset:], Y[offset:],med[offset:]
    
    
    n_est = 80
    params = { 'loss':'lad',
            'n_estimators': n_est, 
              'max_depth': 8, 
              'min_samples_split': 2,
              'learning_rate': 0.05, 
    #          'subsample':0.7,
    #          'max_features':'sqrt'
              }
    
    clf = GradientBoostingRegressor(**params)
    clf.fit(X_train, y_train)
    joblib.dump(clf, 'sklean_GBMv2' + str(i) + '.pkl',compress=True)
    print SMAPE(y_test*med_test,(clf.predict(X_test)>0)*1.0*clf.predict(X_test)*med_test)
    
    Xpreds = np.round((clf.predict(X)>0)*1.0*clf.predict(X)*med).astype(np.int)
    matres = np.vstack((matres.T,Xpreds)).T

resout = pd.DataFrame(matres)
resout.columns=['id']+[str(i) for i in resout.columns[1:] ]
resout['pred']=0
for c in resout.columns[1:(len(resout.columns)-1)]:
    #print c
    resout['pred']+=resout[c]
resout['pred'] = np.round(resout['pred']/(len(resout.columns)-2))

resout = resout.merge(mat[['y','id']],how='left', left_on='id',right_on='id')
SMAPE(resout['y'],resout['pred'])
SMAPE(resout['y'],np.median(resout[resout.columns[1:12]],axis=1))
SMAPE(resout['y'],np.mean(resout[resout.columns[1:12]],axis=1))

resout.to_csv('resout2.csv',index=False)
resout1 = pd.read_csv('resout.csv')

cols = [c for c in resout1.columns if c not in ['y_y','y_x','id']]
resout1 = resout1.merge(resout, how='left',left_on='id',right_on='id')
SMAPE(resout1['y_x'],np.median(resout1[cols],axis=1))

resout1['median']=np.median(resout1[cols],axis=1)
med = resout1['median'].apply(lambda x: 1 if x<1 else x)


X3 = resout1.drop(['id','y_x','y_y'],axis=1)
Y = resout1['y_x']/med

X3, Y = shuffle(X3,Y)
offset = int(X3.shape[0] * 0.2)
X_train, y_train, med_train = X3[:offset], Y[:offset], med[:offset]
X_test, y_test, med_test = X3[offset:], Y[offset:], med[offset:]

#X_train, y_train = X3[:offset], Y[:offset]
#X_test, y_test = X3[offset:], Y[offset:]


n_est = 20
params = { 'loss':'lad',
        'n_estimators': n_est, 
          'max_depth': 5, 
          'min_samples_split': 2,
          'learning_rate': 0.05, 
#          'subsample':0.7,
#          'max_features':'sqrt'
          }

clf = GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)
print SMAPE(y_test*med_test,clf.predict(X_test)*med_test)



test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')


