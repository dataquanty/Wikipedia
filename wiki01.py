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
fout = open('train_transf.csv','w')
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
            
            for k in range(3):
                for n in range(3):
                    datei = randint(90,len(ts)-1)
                    datestr = headers[datei+1].replace('\"','')
                    days = randint(1,60)
                    
                    tsfiltered = ts[:(datei-days+1)]
                    y = ts[datei]
                    
                    lents = len(tsfiltered)
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
                    
                    
                    xi = range(len(tsfiltered))
                    
                    try:
                        p1 = np.poly1d(np.polyfit(xi[-7:], tsfiltered[-7:], 1))
                        p2 = np.poly1d(np.polyfit(xi, tsfiltered,2))
                        regdeg1 = int(np.round(p1(datei)))
                        regdeg2 = int(np.round(p2(datei)))
                    except:
                        regdeg1 = 0
                        regdeg2 = 0
                        print i,ts,tsfiltered
                    
                    
                    yeardate = int(datestr[5:7])*30+int(datestr[8:])
                    dayofweek = datetime.date(*(int(s) for s in datestr.split('-'))).isoweekday()
                    
                    pagelmt = page.rsplit('_',3)
                    
                    
                    country = pagelmt[1][:2]
                    # if pagelmt[2]==all-access else mobile-web else desktop
                    # if pagelmt[3]==spider 1 else  0
                    
                    lststrout = ['\"' + str(page).replace('\"','\"\"') +'\"']
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
