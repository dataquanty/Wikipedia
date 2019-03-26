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
from math import sin,cos,pi

dateslist = ["2017-09-13","2017-09-14","2017-09-15","2017-09-16","2017-09-17","2017-09-18",
             "2017-09-19","2017-09-20","2017-09-21","2017-09-22","2017-09-23","2017-09-24",
             "2017-09-25","2017-09-26","2017-09-27","2017-09-28","2017-09-29","2017-09-30",
             "2017-10-01","2017-10-02","2017-10-03","2017-10-04","2017-10-05","2017-10-06",
             "2017-10-07","2017-10-08","2017-10-09","2017-10-10","2017-10-11","2017-10-12",
             "2017-10-13","2017-10-14","2017-10-15","2017-10-16","2017-10-17","2017-10-18",
             "2017-10-19","2017-10-20","2017-10-21","2017-10-22","2017-10-23","2017-10-24",
             "2017-10-25","2017-10-26","2017-10-27","2017-10-28","2017-10-29","2017-10-30",
             "2017-10-31","2017-11-01","2017-11-02","2017-11-03","2017-11-04","2017-11-05",
             "2017-11-06","2017-11-07","2017-11-08","2017-11-09","2017-11-10","2017-11-11",
             "2017-11-12","2017-11-13"]






i = 0
fout = open('predict_transf_out.csv','w')
with open('train_2.csv','r') as f:
    ff = reader(f, delimiter=',', quotechar='"')
    for ll in ff:
        if i == 0:
            #headers = f.readline().rstrip('\n').split(',')
            headers = ll
            headersout = ['page','days','lents',
              'daysofweeksin','daysofweekcos','yeardatecos','yeardatesin','year','median','country','access','accesstype']
            headersout = headersout + ['d'+str(x) for x in range(60)] + ['y']
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
            
            
            median = int(round(np.median(tsfiltered)))
            y = ''
            
            
            lents = len(tsfiltered)
            
            for k in range(len(dateslist)):
                datei = len(ts) + k
                datestr = dateslist[k]
                days = k+1
            
                dateOfdate = datetime.date(*(int(s) for s in datestr.split('-')))
                dayofweek = dateOfdate.isoweekday()
                dayofweeksin = sin((dayofweek-1)*2*pi/6)
                dayofweekcos = cos((dayofweek-1)*2*pi/6)
                
                yeardatecos = cos(float((dateOfdate-datetime.date(dateOfdate.year,1,1)).days)*2*pi/(365))
                yeardatesin = sin(float((dateOfdate-datetime.date(dateOfdate.year,1,1)).days)*2*pi/(365))
                year = dateOfdate.year
               
                pagelmt = page.rsplit('_',3)
                
                
                country = pagelmt[1][:2]
                # if pagelmt[2]==all-access else mobile-web else desktop
                # if pagelmt[3]==spider 1 else  0
                
                lststrout = ['\"' + str(page).replace('\"','\"\"') +'_'+ datestr + '\"']
                lststrout.append(str(days))
                lststrout.append(str(lents))
                lststrout.append(str(round(dayofweeksin,5)))
                lststrout.append(str(round(dayofweekcos,5)))
                lststrout.append(str(round(yeardatecos,5)))
                lststrout.append(str(round(yeardatesin)))
                lststrout.append(str(year))
                lststrout.append(str(median))
                
                lststrout.append(str(country))
                lststrout.append(pagelmt[2])
                lststrout.append((pagelmt[3]))
                mediandiv = (median<1)*1+median
                lststrout = lststrout + ['0']*(60-len(tsfiltered)) + [str(x) for x in tsfiltered[-60:]] 
                lststrout.append(str(y))
                #lststrout.append(Xdatestr)
                #lststrout.append(datestr)
                strout = ','.join(lststrout)
                fout.write(strout+'\n')
                i+=1
                if i%10==0:
                    sys.stdout.write(str(100*round(float(i)/145065,3)) + "%     \r")
                    sys.stdout.flush()
                
"""
            if i>2:
                break
"""
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