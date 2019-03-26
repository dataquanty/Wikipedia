#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:41:01 2017

@author: dataquanty
"""

import pandas as pd
import numpy as np
import csv
from sklearn.externals import joblib


cols = ['page', 'days', 'lents', 'daysofweeksin', 'daysofweekcos', 'yeardatecos', 'yeardatesin', 'year', 'median', 
        'country', 'access', 'accesstype', 'd0', 'd1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9', 'd10', 'd11', 
        'd12', 'd13', 'd14', 'd15', 'd16', 'd17', 'd18', 'd19', 'd20', 'd21', 'd22', 'd23', 'd24', 'd25', 'd26', 
        'd27', 'd28', 'd29', 'd30', 'd31', 'd32', 'd33', 'd34', 'd35', 'd36', 'd37', 'd38', 'd39', 'd40', 'd41', 
        'd42', 'd43', 'd44', 'd45', 'd46', 'd47', 'd48', 'd49', 'd50', 'd51', 'd52', 'd53', 'd54', 'd55', 'd56', 
        'd57', 'd58', 'd59', 'y']


fnames = ['xaa','xab','xac','xad','xae','xaf','xag','xah','xai']
for f in fnames:
    if f=='xaa':
        mat = pd.read_csv(f)
    else:
        mat = pd.read_csv(f,header=None,names=cols)


    keys = pd.read_csv('key_2.csv')
    
    mat = mat.merge(keys,left_on='page',right_on='Page',how='left')
    mat.drop('Page',inplace=True,axis=1)
    
    wikicrawl = pd.read_csv('train_wikicrawl_v2.csv')
    mat = mat.merge(wikicrawl,how='left',left_on='page',right_on='Page')
    
    mat.drop(['page','Page'],inplace=True,axis=1)
    
    
    
    
    mat['access'] = mat['access'].apply(lambda x: 0 if x == 'all-access' else 1 if x=='mobile-web' else 2)
    mat['access0']=mat['access'].apply(lambda x: 1 if x==0 else 0)
    mat['access1']=mat['access'].apply(lambda x : 1 if x==1 else 0)
    mat.drop('access',axis=1,inplace=True)
    
    mat['accesstype'] = mat['accesstype'].apply(lambda x : 0 if x == 'spider' else 1 )
    
    
    countrlst = ['co', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'ww', 'zh']
    for c in countrlst[1:]:
        mat['countr_'+ c]=mat['country'].apply(lambda x : 1 if x==c else 0)
    
    mat.drop('country',axis=1,inplace=True)
    mat = mat.fillna(0)
    mat = mat.replace(np.inf, 0)
    
    med = (mat['median'])
    med = med.apply(lambda x: 1 if x==0 else x)
    
    X = mat.drop(['y','Id'],axis=1)
    
    preds = np.empty((len(X)))
    for i in range(0,25):
        jobname = 'sklean_GBMv2' + str(i) + '.pkl'
        model = joblib.load(jobname)
        y_pred = model.predict(X)
        y_pred = np.round((y_pred>0)*1.0*y_pred*med).astype(np.int)
        preds = np.vstack((preds.T,y_pred)).T
    
    y_pred = np.median(preds,axis=1)
    
    
    
    mat['Visits']=y_pred
    mat = mat[['Id','Visits']]
    mat['Visits']=mat['Visits'].apply(lambda x: 0 if x<0 else round(x))
    mat['Visits']=mat['Visits'].astype(int)
    
    mat.to_csv('o'+f,index=False, header=False)
    print 'Done'