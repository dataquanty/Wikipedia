#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 09:54:05 2017

@author: dataquanty
"""
################### BULLSHIT TRIAL #####################

import pandas as pd, numpy as np
import csv
import scipy
from sklearn.utils import shuffle
from operator import itemgetter
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.kernel_approximation import RBFSampler


def SMAPE(y_true, y_pred): 
    
    return np.mean(2*np.abs(y_true - y_pred) /(1+y_true+(y_pred>0)*y_pred)) 


mat = pd.read_csv('train_transf_v4.csv')
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
mat = mat[(np.isnan(mat['images']))==False]

mat = mat.fillna(0)
mat = mat.replace(np.inf, 0)

"""
cols = ['ave7', 'ave14', 'ave28', 'aveAll', 'varAll', 'regdeg1','regdeg2']
for c in cols:
    mat[c]=mat[c].apply(lambda x: 0 if x<0 else x)
    #mat[c]=np.log1p(mat[c])
"""

cols = mat.drop('y',axis=1).columns


for c in cols:
    mat[c] = pd.cut(mat[c],256,labels=False)


for c in cols:
    try:
        mat[c]=mat[c].astype('int')
    except:
        print c

med = (mat['median'])
med = med.apply(lambda x: 1 if x==0 else x)

cols = ['ave7','ave28','median','ave14','aveAll']
for c in cols:
    mat[c]=mat[c]/med

colstolearn = ['yeardate','dayofweek','accesstype','summary','cat','images','links','refs','access0','access1','countr_de', 'countr_en', u'countr_es', u'countr_fr', u'countr_ja',
       'countr_ru', 'countr_ww', 'countr_zh']

colstolearn2 = ['days', 'ave7', 'ave14', 'ave28', 'aveAll', 'sigma_mu', 
                'median', 'pctsup1sigma', 'pctsup2sigma', 'pctsup3sigma', 'regdeg1', 'regdeg2', 'yeardate', 'dayofweek','predictWiki']


X = mat[((mat['y']>10000) | (mat['y']<0))==False].drop('y',axis=1)
X = mat[colstolearn]
X = mat[colstolearn2]
X = mat.drop(['y'],axis=1)
Y = mat['y']/med
Y = mat['y']
Y = mat[((mat['y']>10000) | (mat['y']<0))==False]['y']/med
Y = np.log1p(Y)



scaler = StandardScaler()
scaler.fit(X)
joblib.dump(scaler, 'sklean_scaler1.pkl',compress=True)

X = scaler.transform(X)

rbf = RBFSampler(gamma=0.05,n_components=100)
rbf.fit(X)
X = rbf.transform(X)


X, Y,med = shuffle(X,Y,med)

offset = int(X.shape[0] * 0.2)
X_train, y_train = X[:offset], Y[:offset]
X_test, y_test = X[offset:], Y[offset:]



X_test, y_test,med_test = X[offset:], Y[offset:],med[offset:]




n_est = 80
params = { 'loss':'lad',
        'n_estimators': n_est, 
          'max_depth': 8, 
          'min_samples_split': 2,
          'learning_rate': 0.05, 
#          'subsample':0.7,
          'max_features':'sqrt'
          }

clf = GradientBoostingRegressor(**params)
clf.fit(X_train, y_train)
print SMAPE(y_test*med_test,clf.predict(X_test)*med_test)

#print SMAPE(y_test,clf.predict(X_test))
print SMAPE(np.expm1(y_test),np.expm1(clf.predict(X_test)))

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

mat['predictWiki'] = np.round(clf.predict(X))


#cols = np.array(mat.drop('y',axis=1).columns)
cols = colstolearn2
importance = clf.feature_importances_

#plt.figure()
featImport = pd.concat((pd.DataFrame(cols),pd.DataFrame(importance)),axis=1)
featImport.columns=['f','v']
featImport.sort_values('v',ascending=False,inplace=True)
featImport.set_index('f',inplace=True)

featImport.plot(kind='bar')
plt.subplots_adjust(bottom = 0.3)
plt.show()










def report(grid_scores, n_top=5):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")
        
        

paramDist = {'n_estimators': [20],
#             'criterion': ['gini'],
             'max_features':['auto'],
             #'max_depth': [10,11,12,13],
             'max_depth': scipy.stats.expon(scale=12),
             'min_samples_split':[2],
             'min_samples_leaf': [2]}


paramDist = {'n_estimators': scipy.stats.randint(30,40),
             'learning_rate': [0.1],
             'max_features':['auto'],
             'max_depth' : scipy.stats.randint(5,12),
#             'loss' : ['lad'],
#             'max_depth': scipy.stats.expon(scale=10),
#             'min_samples_split':scipy.stats.expon(scale=2),
             'min_samples_leaf':scipy.stats.randint(1,4)}

Reg = LogisticRegression(solver='sag')
Reg = LinearRegression()
Reg = Ridge(alpha=1000)

Rforest = RandomForestRegressor(criterion='mae')
Gradboost = GradientBoostingRegressor(loss='lad')
grid_search = RandomizedSearchCV(Gradboost,cv=3,param_distributions=paramDist,n_iter=8,n_jobs=8, scoring='neg_mean_absolute_error')
grid_search = RandomizedSearchCV(Rforest,param_distributions=paramDist,n_iter=4,n_jobs=4,cv=3, scoring='neg_mean_absolute_error')

Reg.fit(X_train,y_train)
grid_search.fit(X_train, y_train)
#svmr.fit(X_train,y_train)

scoresGrid = grid_search.grid_scores_
print grid_search.best_score_
print grid_search.best_estimator_
report(grid_search.grid_scores_)


cols = np.array(mat.drop('y',axis=1).columns)

importance = grid_search.best_estimator_.feature_importances_

#plt.figure()
featImport = pd.concat((pd.DataFrame(cols),pd.DataFrame(importance)),axis=1)
featImport.columns=['f','v']
featImport.sort_values('v',ascending=False,inplace=True)
featImport.set_index('f',inplace=True)

featImport.plot(kind='bar')
plt.subplots_adjust(bottom = 0.3)
plt.show()





y_pred = Reg.predict(X_test)
y_pred = grid_search.best_estimator_.predict(X_test)

SMAPE(np.expm1(y_test),np.expm1(y_pred))
SMAPE(y_test,y_pred)
SMAPE(y_test*med_test,y_pred*med_test)

joblib.dump(grid_search.best_estimator_, 'sklean_RandFor1.pkl',compress=True)

joblib.dump(clf, 'sklearn_GBM1.pkl',compress=True)
