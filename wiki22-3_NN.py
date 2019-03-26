#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 14:17:10 2017

@author: dataquanty
"""

import pandas as pd, numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def SMAPE(y_true, y_pred): 
    
    return np.mean(SMAPEu(y_true,y_pred)) 

def SMAPEu(y_true,y_pred):
    y_true = y_true.astype('float')
    y_pred = y_pred.astype('float')
    return (np.abs(y_true - y_pred)) /((y_true+y_pred)/2 +((y_pred==0.0)*(y_true==0.0)*1))



mat = pd.read_csv('train_transf_v3.csv')

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

mat = mat[((mat['y']>10000) | (mat['y']<0) | (mat['median']<1))==False]


med = mat['median']
med = med.apply(lambda x: 1 if x==0 else x)

def sigmoid(x):
    return 1/(1+np.exp(-x))


X = mat.drop(['y'],axis=1)
Y = mat['y']/med
Y = np.log(0.0001+Y)
#Y = mat['y']

#Y = np.log1p(Y)
Ymax = np.max(np.abs(Y))
#Y = Y/Ymax
Y = Y/Ymax

#Y = mat[((mat['y']>10000) | (mat['y']<0))==False]['y']/med
#Y = np.log1p(Y)

scaler = StandardScaler()
scaler.fit(X)
#joblib.dump(scaler, 'sklean_scaler1.pkl',compress=True)

X = scaler.transform(X)
X = np.array(X,dtype='float')

X, Y, med = shuffle(X,Y,med)

offset = int(X.shape[0] * 0.8)
X_train, y_train = X[:offset], Y[:offset]
X_test, y_test = X[offset:], Y[offset:]

X_train, y_train, med_train = X[:offset], Y[:offset],med[:offset]
X_test, y_test,med_test = X[offset:], Y[offset:],med[offset:]

def trOut_old(preds):
    return (1+preds)*Ymax/2


def trOut(preds):
    return (np.exp(preds*Ymax)-0.0001)


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))


n_features = X_train.shape[1] #input units

layers = [n_features,100,100,10,100,100,1]


batch_size = 1000
n_epochs = 15



tx = tf.placeholder(tf.float32, (None, X_train.shape[1]))
ty = tf.placeholder(tf.float32, (None, ))
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

w_h = []
for i in range(len(layers)-1):
    w_h.append(init_weights([layers[i],layers[i+1]]))

biases = []
for i in range(1,len(layers)):
    biases.append(tf.Variable(tf.random_normal([layers[i]],stddev=0.1)))


h = tx
for i in range(1,len(layers)-1):
    h = tf.add(tf.matmul(h, w_h[i-1]),biases[i-1])
    h = tf.contrib.layers.batch_norm(h,center=True,scale=True,is_training=is_training)
    h = tf.nn.tanh(h)
    h = tf.nn.dropout(h,keep_prob)
    
y_predict = tf.nn.tanh(tf.matmul(h, w_h[len(w_h)-1])+biases[len(biases)-1])



#tError = tf.reduce_mean(tf.square(tf.subtract(y_predict,ty)) + \
#                        0.0 *sum([tf.nn.l2_loss(x) for x in w_h]))

#tError = tf.reduce_mean(2*tf.divide(tf.abs(tf.subtract(trOut(y_predict),trOut(ty))),tf.add(tf.abs(trOut(y_predict)),tf.abs(trOut(ty)))) + \
#                        0.0*sum([tf.nn.l2_loss(x) for x in w_h]))

def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)

#tError = tf.reduce_mean(huber_loss(ty,y_predict,delta=1.0) + \
#                        0.00001 *sum([tf.nn.l2_loss(x) for x in w_h]))



tError = tf.reduce_mean(tf.abs(tf.subtract(y_predict,ty)) + \
                        0.01 *sum([tf.nn.l2_loss(x) for x in w_h]))




tOptimize = tf.train.GradientDescentOptimizer(0.1).minimize(tError)
#tOptimize = tf.train.AdamOptimizer(learning_rate=0.5,beta1=0.9,beta2=0.999).minimize(tError)
#tOptimize = tf.train.AdadeltaOptimizer(learning_rate=0.5,rho=0.95).minimize(tError)





it = []
res = []
train_preds = []
test_preds = []
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    it = 0 
    for i_e in range(n_epochs):
        for i in range(0, X_train.shape[0], batch_size):    
            batch_X = X_train[i:i + batch_size, ...]
            batch_y = y_train[i:i + batch_size]
                    
            _, loss = sess.run([tOptimize, tError], feed_dict={tx: batch_X, ty: batch_y,keep_prob : 0.5, is_training:1})
            #print i,loss
            
            res.append(loss)
            
            
                
                

            if it%3 == 0 :
                #v = w_h[0].eval()
                #print(v)
                #a = sess.run(y_predict, feed_dict={tx: X_train, ty: y_train, keep_prob:1.0,is_training:0})
                #e_train = SMAPE(trOut(y_train)*med_train, np.round(trOut(a[:,0])*med_train))
                #train_preds.append(SMAPE(((1+y_train)*Ymax/2), np.round(((1+a[:,0])*Ymax/2))))
                #train_preds.append(SMAPE(np.expm1((1+y_train)*Ymax/2)*med_train, np.round(np.expm1((1+a[:,0])*Ymax/2))*med_train))
                #train_preds.append(SMAPE(y_train*med_train,np.round(a[:,0]*med_train)))
                b = sess.run(y_predict, feed_dict={tx: X_test, ty: y_test, keep_prob:1.0,is_training:0})
                #valid = SMAPE(trOut(y_test), np.round(trOut(1+b[:,0])))
                e_valid = SMAPE(np.round(trOut(y_test)*med_test), np.round(trOut(b[:,0])*med_test))
                #valid = SMAPE(np.expm1((1+y_test)*Ymax/2)*med_test, np.round(np.expm1((1+b[:,0])*Ymax/2))*med_test)
                #valid = SMAPE(y_test*med_test,np.round(b[:,0]*med_test))
                #train_preds.append(e_train)
                test_preds.append(e_valid)
                print i,loss,e_valid


#SMAPE(y_test*med_test,np.round(b[:,0]*med_test))


fig = plt.figure()
plt.subplot(1,2,1)
plt.title('Loss')
plt.plot(res,'b-',label='Training loss')
#plt.plot(hist['val_acc'],'r-', label = 'Validation accuracy')
#plt.ylim( (0.5, 1) )
plt.legend(loc='upper right')
plt.subplot(1,2,2)
plt.title('SMAPE')
plt.plot(train_preds,'b-',label='Training SMAPE')
plt.plot(test_preds, 'r-',label = 'Validation SMAPE')
#plt.ylim( (0, 0.5) )
plt.legend(loc='upper right')





