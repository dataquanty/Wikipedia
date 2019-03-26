#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:32:00 2017

@author: dataquanty
"""

import wikipedia
from csv import reader
import time
import threading
import sys
import warnings


def getWikipediaStats(page):
    
    pagelmt = page.rsplit('_',3)
    try:
        wikipedia.set_lang(pagelmt[1][:2])
        wikipage = wikipedia.page(pagelmt[0],preload=False)
        properties = ['\"' + str(page).replace('\"','\"\"') + '\"']
        properties.append(str(len(wikipage.summary)))
        properties.append(str(len(wikipage.categories)))
        properties.append(str(len(wikipage.images)))
        properties.append(str(len(wikipage.links)))
        properties.append(str(len(wikipage.references)))
        #properties.append(str(len(wikipage.sections)))
    except:
        properties = ['\"' + str(page).replace('\"','\"\"') + '\"']
        properties = properties + ['','','','','']
    strout = ','.join(properties) + '\n'
    sys.stdout.write(strout)
    
t1 = time.time()

pages = []
with open('train_1_trail64.csv','r') as f:
    i = 0
    ff = reader(f, delimiter=',', quotechar='"')
    for l in ff:
        i+=1
        if i==1:
            sys.stdout.write('Page,summary,cat,images,links,refs\n')
        else:
            pages.append(l[0])
            
            
            if i%100==0:
                
                threads = [threading.Thread(target=getWikipediaStats, args=(page,)) for page in pages]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                
                pages = []
        
        if i%5000==0:
            warnings.warn(str(i) + ' in ' + str(round((time.time()-t1)/3600,2)))
        


threads = [threading.Thread(target=getWikipediaStats, args=(page,)) for page in pages]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()

f.close()
#print round(time.time()-t1)







