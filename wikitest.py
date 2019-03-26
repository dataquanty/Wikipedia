#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:12:45 2017

@author: dataquanty
"""

from csv import reader

with open('wikitest.csv','r') as f:
    ff = reader(f, delimiter=',')
    for ll in ff:
        print ll[0].replace('\"','\"\"')