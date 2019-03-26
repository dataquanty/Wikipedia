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
    
    return np.mean(2*np.abs(y_true - y_pred) /(1+y_true+(y_pred>0)*y_pred)) 



mat = pd.read_csv('train_transf.csv')
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


med = (mat['median']+mat['ave7'])/2
med = med.apply(lambda x: 1 if x==0 else x)

#X = mat[((mat['y']>10000) | (mat['y']<0))==False].drop('y',axis=1)
X = mat.drop(['y'],axis=1)
Y = mat['y']/med
#Y = mat['y']
#Y = mat[((mat['y']>10000) | (mat['y']<0))==False]['y']/med
#Y = np.log1p(Y)

scaler = StandardScaler()
scaler.fit(X)
#joblib.dump(scaler, 'sklean_scaler1.pkl',compress=True)

X = scaler.transform(X)


X, Y, med = shuffle(X,Y,med)

offset = int(X.shape[0] * 0.4)
X_train, y_train = X[:offset], Y[:offset]
X_test, y_test = X[offset:], Y[offset:]

X_train, y_train, med_train = X[:offset], Y[:offset],med[:offset]
X_test, y_test,med_test = X[offset:], Y[offset:],med[offset:]



def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.1))



n_features = X_train.shape[1] #input units
first_layer = 200
second_layer = 20
third_layer = 200
batch_size = 300
n_epochs = 1



tx = tf.placeholder(tf.float32, (None, X_train.shape[1]))
ty = tf.placeholder(tf.float32, (None, ))
keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)

w_h1 = init_weights([n_features, first_layer])
w_h2 = init_weights([first_layer, second_layer])
w_h3 = init_weights([second_layer, third_layer])
w_o = init_weights([third_layer, 1])

biases = {
    'b1': tf.Variable(tf.random_normal([first_layer])),
    'b2': tf.Variable(tf.random_normal([second_layer])),
    'b3': tf.Variable(tf.random_normal([third_layer])),
    'bout': tf.Variable(tf.random_normal([1]))
}


h1 = tf.add(tf.matmul(tx, w_h1),biases['b1'])
h1 = tf.contrib.layers.batch_norm(h1,center=True,scale=True,is_training=is_training)
h1 = tf.nn.tanh(h1)
h1 = tf.nn.dropout(h1,keep_prob)

h2 = tf.add(tf.matmul(h1, w_h2),biases['b2'])
h2 = tf.contrib.layers.batch_norm(h2,center=True,scale=True,is_training=is_training)
h2 = tf.nn.tanh(h2)
h2 = tf.nn.dropout(h2,keep_prob)


h3 = tf.add(tf.matmul(h2, w_h3),biases['b3'])
h3 = tf.contrib.layers.batch_norm(h3,center=True,scale=True,is_training=is_training)
h3 = tf.nn.tanh(h3)
h3 = tf.nn.dropout(h3,keep_prob)

y_predict = tf.nn.tanh(tf.matmul(h3, w_o))+biases['bout']

tError = tf.reduce_mean(tf.abs(tf.subtract(y_predict,ty)) + \
                        0.01 * (tf.nn.l2_loss(w_h1) +tf.nn.l2_loss(w_h2) +tf.nn.l2_loss(w_h3) + tf.nn.l2_loss(w_o) ))

tOptimize = tf.train.GradientDescentOptimizer(0.01).minimize(tError)


it = []
res = []
train_preds = []
test_preds = []
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i_e in range(n_epochs):
        for i in range(0, X_train.shape[0], batch_size):    
            batch_X = X_train[i:i + batch_size, ...]
            batch_y = y_train[i:i + batch_size]
                    
            _, loss = sess.run([tOptimize, tError], feed_dict={tx: batch_X, ty: batch_y,keep_prob : 0.6, is_training:1})
            #print i,loss
            it.append(i)
            res.append(loss)



            a = sess.run(y_predict, feed_dict={tx: X_train, ty: y_train, keep_prob:1.0,is_training:0})
            train_preds.append(SMAPE(y_train*med_train,np.round(a[:,0]*med_train)))
            b = sess.run(y_predict, feed_dict={tx: X_test, ty: y_test, keep_prob:1.0,is_training:0})
            valid = SMAPE(y_test*med_test,np.round(b[:,0]*med_test))
            test_preds.append(valid)
            print i,loss,valid


SMAPE(y_test*med_test,np.round(b[:,0]*med_test))


fig = plt.figure()
plt.subplot(1,2,1)
plt.title('Loss')
plt.plot(res,'b-',label='Training loss')
#plt.plot(hist['val_acc'],'r-', label = 'Validation accuracy')
#plt.ylim( (0.5, 1) )
plt.legend(loc='lower right')
plt.subplot(1,2,2)
plt.title('SMAPE')
plt.plot(train_preds,'b-',label='Training SMAPE')
plt.plot(test_preds, 'r-',label = 'Validation SMAPE')
#plt.ylim( (0, 0.5) )
plt.legend(loc='upper right')





