#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 15:43:47 2017

@author: dataquanty
"""

import pandas as pd
import numpy as np
from random import randint
import datetime
import sys
from csv import reader
from math import sin,cos,pi
"""
mat = pd.read_csv('train_1.csv')

mat[mat['Page'].apply(lambda x: 'Category:Nude_women' in x)]

"https://www.wikidata.org/w/api.php?action=wbgetentities&sites=dewiki&titles=Nadja_Abd_el_Farrag&normalize=1"

mat = mat.melt(id_vars=['Page'])
mat['variable']=mat['variable'].astype(np.datetime64)
mat['value']=mat['value'].fillna(0)  #remove NA's ?
mat['value'] = mat['value'].astype(np.int)
mat['month']=mat['variable'].apply(lambda x: x.month+x.day/31)
"""

"""
for each line
for 3 specific dates 
    for 3 limits (1-60) : 
        number of days past from last known figure (limit 60 days)
        take before that limit the last know figure, the 7 day moving average, 14 days moving average, 28 days moving average
        take before that limit the global trend (=linear regression)
        add feature to the specific date (month+day, day of week)
        add features to the article : number of links, number of categories, number of images ...
        dump to file

"""
i = 0
fout = open('train_transf_v4.csv','w')
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
            
            for k in range(3):
                for n in range(3):
                    datei = randint(90,len(ts)-1)
                    datestr = headers[datei+1].replace('\"','')
                    days = randint(1,62)
                    
                    tsfiltered = ts[:(datei-days+1)]
                    
                    median = int(round(np.median(tsfiltered)))
                    y = ts[datei]
                    
                    
                    lents = len(tsfiltered)
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
                    
                    lststrout = ['\"' + str(page).replace('\"','\"\"') +'\"']
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
            if i==5:
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
