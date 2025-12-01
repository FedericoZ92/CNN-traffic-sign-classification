# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 10:09:00 2016

@author: zanetti
"""

#Build histogram of image dimensions


from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from io import BytesIO
import glob, os, sys, time, pdb

import lasagne
from lasagne.layers import InputLayer, DenseLayer, DropoutLayer
from lasagne.layers import Conv2DLayer as ConvLayer  

from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as NormLayer
from lasagne.utils import floatX

from PIL import Image
import scipy.misc
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
from pylab import *
from numpy import *
import theano
import theano.tensor as T
import lasagne
import io
import skimage.transform
from skimage import novice
from skimage import data, io, filters

import matplotlib
import matplotlib.cm as cm
from urllib import urlretrieve
import cPickle as pickle
import gzip
from lasagne import layers
from lasagne.updates import nesterov_momentum
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import pickle
import scipy.misc

#layername = 'conv1'
matplotlib.rcParams.update({'font.size': 7})

#GLOBAL SETTINGS
homedir = '/home/zanetti/Scrivania/My_Programs/GTSRB_4_tests/'
architecture_type = 'CNN2' #Possibilities: MLP, CNN1, CNN2, CNN3 ,ResNet1
png_converter = 1
num_epochs_ = 1
batch_size = 16
image_size = 32
crop_size = 16
c_2 = crop_size / 2
num_of_classes = 2
learning_rate_ = 0.005 # now it's possible to change l.r. on each ite., just check below
momentum_ = 0.9
toLoadWeights = False
simulation_name = 'default' #To be changed every time architecture or data changes to save the weights. 
show_filters = 0

hist_range = 230


def load_dataset(png_needed):    
    train_images = []
    test_images = []
    #val_images = []
    train_labels = []
    test_labels = []
    #val_labels = []

    # We define functions for loading MNIST images and labels.
    import gzip
    ##########################################
    def load_train_images(png_needed):
  
        classes = 1   
        for classes in range(0,num_of_classes): 
                  
           if png_needed == 1:          
             for file in glob.glob(homedir + str(classes) + '/' + str(classes) + '_train/*.ppm'):
                 print('classes index:' + str(classes)) 
                 print('TEST DIRECTORY' + homedir + str(classes) + '_train/*.ppm')
                 
                 im_temp = Image.open(file)                 
             #    file = file[:-4]
             #    im_temp.save(file + '.png')
             #for file in glob.glob(homedir + str(classes) + '/' + str(classes) + '_train/*.png'):
             #    im_temp = Image.open(file)
             #    #im_temp = prep_image_for_test_A(file)
             #    file = file[:-4]

             #   scipy.misc.toimage(im_temp).save(file + '.png')

           #for file in glob.glob(homedir + str(classes) + '/' + str(classes) + '_train/*.png'):    
           #    im_temp = Image.open(file)

           #    rawim, im = prep_image_for_test_B(file, 0)
             
           #    file_name = file
               
                 train_images.append(file) 
           #    train_labels.append(classes)                

        return train_images
        
    #############################################
    def load_test_images(png_needed):
  
        for classes in range(0,num_of_classes):          
           if png_needed == 1:
             for file in glob.glob(homedir + str(classes) + '/' + str(classes) + '_test/*.ppm'):
                 im_temp = Image.open(file)                 
                 #file = file[:-4]
                 #im_temp.save(file + '.png')#('im10.png')
             #for file in glob.glob(homedir + str(classes) + '/' + str(classes) + '_test/*.png'):
             #    im_temp = Image.open(file)
                 #im_temp = prep_image_for_test_A(file)
                 #file = file[:-4]
                 #scipy.misc.toimage(im_temp).save(file + '.png')
               
           #for file in glob.glob(homedir + str(classes) + '/' + str(classes) + '_test/*.png'):    
           #    im_temp = Image.open(file)#('im10.png') #images[0] = mpimg.imread('im1.png') #Image.open("im1.png")

           #    rawim, im = prep_image_for_test_B(file, 0)
               
           #    file_name = file
               
                 test_images.append(file) 
           #    test_labels.append(classes)
               
        return test_images


    def load_labels(case):
        if case == "train":
            return train_labels
        if case == "test":
           return test_labels
        #if case == "val":
        #   return val_labels

    # We can now download and read the training and test set images and labels.
    X_train = load_train_images(png_needed)    
    X_test = load_test_images(png_needed)
    #X_val = load_val_images(png_needed)
    y_train = load_labels('train')
    y_test = load_labels('test')
    #y_val = load_labels('val')
   
    return X_train, y_train, X_test, y_test
    
    

print("============PART-3============")
a = 17
b=3
print(str(a%b))    
    
print("============PART-2============")          
    
x = 5    
file = open("test.txt","a")
file.write(" - %f" %(x))
file.close()    
    
    
print("============PART-1============")       
a = 1
b = 0.9999999999
x = T.eq(a,b)
print("value of x: ")
print(x.eval())

print("============PART0============")
   
jj = []
jj.append(1)
jj.append(2)
jj.append(3)
print(jj.index(max(jj)))
   
   
    
print("============PART1============")
    
    
pic = np.random.random([16,16]) 
piclist = np.random.random([1,64,16,16])   



class testxx(object):
    def __init__(self, input):
        self.input = input
        self.output = input#T.sum(input)
a = T.tensor4(dtype=theano.config.floatX)
classfier = testxx(a)
outxx = classfier.output
f = theano.function([a], outxx)
picc = np.random.random([2,2,2,2]) 
outxx = f(picc)
print(outxx)


print("============PART2============")
pic2 = np.random.random([16,16,3]) 
#pic2_32 = np.float32(pic2)
print(str(type(pic2)))
np.squeeze(pic2)
#print("pic_32 shape:")
#print(pic2.shape)             
#rawpic = np.copy(pic)
conv1 = []
numfilters = 32
ysize = 2
xsize = 2
ratio = numfilters/8
for x_x in range (0,numfilters):
   pic3 = np.random.random([3,ysize,xsize])
   conv1.append(pic3)
   #print(conv1[x_x])

conv2 = conv1[3:5][0]
print(conv2)
   
#print(pic3.shape)              
#imshow(pic3, cmap = 'Greys_r')
#print(pic3)
print("===============t================")#str((conv1[0])))

fig, ax = plt.subplots(nrows=ratio, ncols=8, sharex='col', sharey='row', figsize = (8,ratio)) #sharex=True, sharey=False
print('\n' + '==============Layer ' + str(layer_n_feats) + ' filters==============')
          
for i in xrange(0,numfilters): 
           ax[(i)/8][(i)%8].imshow(conv1[i],interpolation = 'nearest', 
                            cmap = 'Greys_r', aspect = '1')#prev.: aspect = 'auto'
           #ax[(i-1)/8][(i-1)%8].autoscale(False)
           ax[(i)/8][(i)%8].set_ylim([0,ysize]) #prev from 0 to 2
           
          
           ####plt.title( "Conv1" + ' layer filters',horizontalalignment='left', verticalalignment='bottom', x= -2, y=-2,fontsize=16)
           #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
           plt.xlim(0,xsize); plt.ylim(0,ysize)
           #plt.figure(figsize=(2,2))
           #fig = plt.figure()
           #fig.title('Layer ' + str(layer_n_feats) + ' filters')
           #fig.show()
           font = {'family' : 'normal',
                   'weight' : 'bold',
                   'size'   : 0}

           matplotlib.rc('font', **font)
          
          
           #plt.show()
           #counter = counter + 1    


def main(model='fede', num_epochs= num_epochs_):
           #pic = skimage.novice.Picture.from_size((3,3), color=(100, 0, 0))
           

           
           gag = "test gag"
           print(gag)
           #  
           # 
           #for i in range(0,5):
           #    for j in range(0,5):   
           #        for k in range(0,2): 
           #             pic[i,j] = 0
           #
           #rawpic = np.copy(pic)
           #from skimage.viewer import ImageViewer
           #imshow(rawpic)
 
           
           
   
if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)   
