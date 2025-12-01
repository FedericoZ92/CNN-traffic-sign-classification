#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 13:24:17 2016

@author: fede
"""

import os, shutil, glob
homedir = '/home/fede/Desktop/cross_fold_segmented/' #root folder from which all subfolders cancel pngs
homedir2 = 

num_of_classes = 43


for classes in range(0,num_of_classes): 

    for file in glob.glob(homedir + str(classes) + '/' + str(classes) + '_train/*.ppm'):
                 #print('classes index:' + str(classes)) 
                 #print('TEST DIRECTORY' + homedir + str(classes) + '_train/*.ppm')
                 
                 im_temp = Image.open(file)                 
                 file = file[-15:-4]
                 #print(str(file))
                 im_temp.save( homedir + str(classes) + '/' + file + '.ppm')
                
                 
    for file in glob.glob(homedir + str(classes) + '/' + str(classes) + '_test/*.ppm'):
                 #print('classes index:' + str(classes)) 
                 #print('TEST DIRECTORY' + homedir + str(classes) + '_train/*.ppm')
                 
                 im_temp = Image.open(file)                 
                 file = file[-15:-4]
                 im_temp.save( homedir + str(classes) + '/' + file + '.ppm')