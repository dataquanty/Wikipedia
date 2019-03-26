#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:27:48 2017

@author: dataquanty
"""

import pandas as pd
import numpy as np
from random import randint
import datetime
import sys
from csv import reader

dateslist = ["2017-01-01","2017-01-02","2017-01-03","2017-01-04","2017-01-05","2017-01-06","2017-01-07",
"2017-01-08","2017-01-09","2017-01-10","2017-01-11","2017-01-12","2017-01-13","2017-01-14","2017-01-15",
"2017-01-16","2017-01-17","2017-01-18","2017-01-19","2017-01-20","2017-01-21","2017-01-22","2017-01-23",
"2017-01-24","2017-01-25","2017-01-26","2017-01-27","2017-01-28","2017-01-29","2017-01-30","2017-01-31",
"2017-02-01","2017-02-02","2017-02-03","2017-02-04","2017-02-05","2017-02-06","2017-02-07","2017-02-08",
"2017-02-09","2017-02-10","2017-02-11","2017-02-12","2017-02-13","2017-02-14","2017-02-15","2017-02-16",
"2017-02-17","2017-02-18","2017-02-19","2017-02-20","2017-02-21","2017-02-22","2017-02-23","2017-02-24",
"2017-02-25","2017-02-26","2017-02-27","2017-02-28","2017-03-01"]





i = 0
fout = open('predict_transf.csv','w')
with open('train_1.csv','r') as f:
    ff = reader(f, delimiter=',', quotechar='"')
    for ll in ff:
        if i == 0:
            #headers = f.readline().rstrip('\n').split(',')
            headers = ll
            headersout = ['page','days','ave7','ave14','ave28','aveAll','sigma_mu','median','pctsup1sigma','pctsup2sigma','pctsup3sigma',
                          'regdeg1','regdeg2','yeardate','dayofweek','country','access','accesstype','y']
            fout.write(','.join(headersout)+'\n')
            i+=1

        else:
#            try:
            #line = ll.rstrip('\n').split(',')
            
            line = ll
            #print line
            page = line[0]
            
            ts = [0 if x == '' else x for x in line[1:] ]
            try:
                ts = [int(x) for x in ts]
            except:
                try:
                    ts = [int(x.replace('1e+05','100000')) for x in ts]
                except:
                    print i,ts
                    ts = [0 for x in ts]
                
            #print ts
            tsfiltered = ts
            lents = len(tsfiltered)
            xi = range(len(tsfiltered))
            ave7 = int(round(np.mean(tsfiltered[-7:])))
            ave14 = int(round(np.mean(tsfiltered[-14:])))
            ave28 = int(round(np.mean(tsfiltered[-28:])))
            aveAll = int(round(np.mean(tsfiltered)))
            sigma = np.sqrt(np.var(tsfiltered))
            sigma_mu = sigma/aveAll
            median = int(round(np.median(tsfiltered[-49:])))
            pctsup1sigma = float(sum([1 for x in tsfiltered if x > (median+sigma)]))/lents
            pctsup2sigma = float(sum([1 for x in tsfiltered if x > (median+2*sigma)]))/lents
            pctsup3sigma = float(sum([1 for x in tsfiltered if x > (median+3*sigma)]))/lents
            
            y = 0
            
            pagelmt = page.rsplit('_',3)
            country = pagelmt[1][:2]
            
            try:
                p1 = np.poly1d(np.polyfit(xi[-7:], tsfiltered[-7:], 1))
                p2 = np.poly1d(np.polyfit(xi, tsfiltered,2))
            except:
                regdeg1 = 0
                regdeg2 = 0
                print i,ts,tsfiltered
            
            
            for k in range(60):
            
                datei = len(ts) + k
                datestr = dateslist[k]
                days = k+1
                
                
                try:
                    regdeg1 = int(np.round(p1(datei)))
                    regdeg2 = int(np.round(p2(datei)))
                except:
                    regdeg1 = 0
                    regdeg2 = 0
                    print i,ts,tsfiltered
                
                
                yeardate = int(datestr[5:7])*30+int(datestr[8:]) #should be (month-1)*30 but same result
                dayofweek = datetime.date(*(int(s) for s in datestr.split('-'))).isoweekday()
                
                
                # if pagelmt[2]==all-access else mobile-web else desktop
                # if pagelmt[3]==spider 1 else  0
                
                lststrout = ['\"' + str(page).replace('\"','\"\"') +'_'+ datestr + '\"']
                lststrout.append(str(days))
                lststrout.append(str(ave7))
                lststrout.append(str(ave14))
                lststrout.append(str(ave28))
                lststrout.append(str(aveAll))
                lststrout.append(str(sigma_mu))
                lststrout.append(str(median))
                lststrout.append(str(pctsup1sigma))
                lststrout.append(str(pctsup2sigma))
                lststrout.append(str(pctsup3sigma))
                lststrout.append(str(regdeg1))
                lststrout.append(str(regdeg2))
                lststrout.append(str(yeardate))
                lststrout.append(str(dayofweek))
                lststrout.append(str(country))
                lststrout.append(pagelmt[2])
                lststrout.append((pagelmt[3]))
                lststrout.append(str(y))
                strout = ','.join(lststrout)
                fout.write(strout+'\n')
            i+=1
            if i%1000==0:
                print i+'  \r',
"""
            except:
                print line
                sys.exit()
                i+=1
                    
"""                 


fout.close()
f.close()
print 'done'
#cat predict_transf.csv | sed -e 's/"Awaken,_My_Love!"/""Awaken,_My_Love!""/g' > predict_transf2.csv