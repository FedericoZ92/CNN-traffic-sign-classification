
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
from skimage.viewer import ImageViewer
from skimage import exposure

import matplotlib
import matplotlib.cm as cm
from urllib import urlretrieve
import cPickle as pickle
import gzip
from lasagne import layers
from lasagne.updates import nesterov_momentum
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from numpy import unravel_index
from tempfile import TemporaryFile

import pickle
import scipy.misc

#layername = 'conv1'
matplotlib.rcParams.update({'font.size': 7})

#GLOBAL SETTINGS
homedir = '/home/fede/Desktop/DATABASE_segmented-24_/'#Full_set1/'#3fold_Full_set1/'#Main directory of dataset
architecture_type = 'CNN3' #Possibilities: MLP, CNN2, CNN3 ,ResNet1. Standard use: CNN2
png_converter = 1 #Set to 1 for image preprocessing and png conversion. To 0 if done before.
num_epochs_ = 0 #Times all images are forward-propagated.
batch_size = 128
image_size = 64
crop_size = 32 #Images are re-sized to crop_size x crop_size.
c_2 = crop_size / 2 #Leave like this for central cropping.
num_of_classes = 43 #The program will load the first num_of_classes image folders for training.
learning_rate_ = 0.005 #Kepp between 0.001 and 0.020
momentum_ = 0.9 #Keep close to 0.9
toLoadWeights = True #To resume training from previous state.
save_weights = 1
simulation_name = '43cl._99ite_patch'#'43cl._1ite_segm'#'std_43cl_10ite_32x32_halveall_fcl256' #To be changed every time architecture or data changes to save the weights. 
show_features = 0 #1 to display features at layer_n_feats_arr layers, 0 not to.
layer_n_feats_arr = [3]#,2,3,4,5,6,8]#1,2,3,4,5,6,8]
show_examples = 0 #1 to classify sample images, 0 not to.
show_filters = 0 #1 to display filters at layer_n_feats_arr layers, 0 not to.
layer_n_filters_arr = [1]
show_confusion_matrix = 1
eval_cat_cross = 0
eval_sq_err = 0
eval_hinge_l = 0
do_only_testing = 0


test_images2 = []
test_labels2 = []

def update_learn_rate(i,learn_rate):
    
    #define function to update learning rate    
    learn_rate = learn_rate

def prep_image_for_test_A(img):

    im = plt.imread(img) 
    
    # Resize so smallest dim = 256, preserving aspect ratio
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (image_size, w*image_size/h))
    else:
        im = skimage.transform.resize(im, (h*image_size/w, image_size))
 
    h, w, _ = im.shape
    #print(str(h) + str(w))
    im = im[h//2-c_2:h//2+c_2, w//2-c_2:w//2+c_2]
    im = exposure.rescale_intensity(im)
    #im = exposure.equalize_hist(im)
    #im = exposure.equalize_adapthist(im, ntiles_x = 4, ntiles_y = 4)
    return im  
       
############################################################################       
       
def prep_image_for_test_B(img, for_test): 
    
    if for_test == 0:
        im = plt.imread(img)  
    else: im = img 
    
    rawim = np.copy(im)
        
    #HERE im VALUES ARE BETWEEN 0 AND 1
    for i in range (0,crop_size):
       for j in range (0,crop_size):                      
               
               #currval = float32(255*255*(0.2990*im[i,j,0] + 0.5870*im[i,j,1] + 0.1440*im[i,j,2]))              
               im[i,j,0] = 255*255*im[i,j,0]
               im[i,j,1] = 255*255*im[i,j,1]
               im[i,j,2] = 255*255*im[i,j,2]
       
    # Shuffle axes to c01
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)

    # Convert to BGR
    im = im[::-1, :, :]

    return rawim, im

############################################################################     

def prep_image_for_training(imlist_, mean_):    
    #print(mean_.shape)
    #print(imlist_[0].shape)
    #print("ffffff")
    #x = imlist_[0]
    #print(x[1,1,1])

    #for element in imlist_:#for e in range(0,len(imlist_)):# 
    #    for i in range(0,3):#len(mean_[0])-1):
    #       for j in range(0,len(mean_[1])):
    #          for k in range(0,len(mean_[2])):
    #              #print(i)#element[i,j,k])
    #              #print(j)#mean_[i,j,k])
    #              #print(k)
    #              #print("-")
    #              #print(element[i,j,k])
    #              element[i,j,k] = element[i,j,k] - mean_[i,j,k]
    #              #print(element[i,j,k])
                  
                  
    #x = imlist_[0]
    #print(mean_[1,1,1])
    #print(x[1,1,1])          
    #print("=============================")
    #print(imlist_[0])   
                  
                  
    #for element in imlist_:   
        #element = element / np.float32(256)
    return imlist_ 
  
############################################################################ 
 
def compute_mean_image(imlist_):

      w = crop_size
      h = crop_size
      N=len(imlist_)

      # Create a numpy array of floats to store the average (assume RGB images)
      arr=np.zeros((3,h,w),np.float)

      for i in range(0,N) :
          imarr=np.array(imlist_[i],dtype=np.float)
          arr=arr+imarr/N
      # Round values in array and cast as 8-bit integer
      arr=np.array(np.round(arr),dtype=np.uint8)
      #print("sample from mean image below: ")

      #arr = 256*arr
      return arr#floatX(arr)
             

############################################################################

def prepare_one_image_for_test(img, mean_): #img must be a floatX (np.array ), in [0,255]
    #img = img - mean_
    #img = img / np.float32(256)
    
    img = (img[np.newaxis])
    return img

############################################################################

def load_dataset(png_needed):    
    train_images = []
    test_images = []
    
    train_labels = []
    test_labels = []    

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
                 file = file[:-4]
                 im_temp.save(file + '.png')
             for file in glob.glob(homedir + str(classes) + '/' + str(classes) + '_train/*.png'):
                 im_temp = Image.open(file)
                 im_temp = prep_image_for_test_A(file)
                 file = file[:-4]

                 scipy.misc.toimage(im_temp).save(file + '.png')

           for file in glob.glob(homedir + str(classes) + '/' + str(classes) + '_train/*.png'):    
               im_temp = Image.open(file)

               rawim, im = prep_image_for_test_B(file, 0)
             
               file_name = file
               
               train_images.append(im) 
               train_labels.append(classes)                

        return train_images
        
    #############################################
    def load_test_images(png_needed):
  
        for classes in range(0,num_of_classes):          
           if png_needed == 1:
             for file in glob.glob(homedir + str(classes) + '/' + str(classes) + '_test/*.ppm'):
                 im_temp = Image.open(file)                 
                 file = file[:-4]
                 im_temp.save(file + '.png')
             for file in glob.glob(homedir + str(classes) + '/' + str(classes) + '_test/*.png'):
                 im_temp = Image.open(file)
                 
                 im = plt.imread(file)
                 test_images2.append(im) 
                 im_temp = prep_image_for_test_A(file)
                 file = file[:-4]
                 scipy.misc.toimage(im_temp).save(file + '.png')
               
           for file in glob.glob(homedir + str(classes) + '/' + str(classes) + '_test/*.png'):    
               im_temp = Image.open(file)

               im = plt.imread(file)
               test_images2.append(im) 
               #test_labels2.append(classes) 
               rawim, im = prep_image_for_test_B(file, 0)
               
               file_name = file
               
               test_images.append(im) 
               test_labels.append(classes)
               
        return test_images
               

               
        return test_images


    def load_labels(case):
        if case == "train":
            return train_labels
        if case == "test":
           return test_labels

    # We can now download and read the training and test set images and labels.
    X_train = load_train_images(png_needed)    
    X_test = load_test_images(png_needed)
    
    y_train = load_labels('train')
    y_test = load_labels('test')
   
    return X_train, y_train, X_test, y_test

##############################################################################

def categorical_accuracy(predictions, targets, top_k=1):
    """Computes the categorical accuracy between predictions and targets.
    .. math:: L_i = \\mathbb{I}(t_i = \\operatorname{argmax}_c p_{i,c})
    Can be relaxed to allow matches among the top :math:`k` predictions:
    .. math::
        L_i = \\mathbb{I}(t_i \\in \\operatorname{argsort}_c (-p_{i,c})_{:k})
    Parameters
    ----------
    predictions : Theano 2D tensor
        Predictions in (0, 1), such as softmax output of a neural network,
        with data points in rows and class probabilities in columns.
    targets : Theano 2D tensor or 1D tensor
        Either a vector of int giving the correct class index per data point
        or a 2D tensor of 1 hot encoding of the correct class in the same
        layout as predictions
    top_k : int
        Regard a prediction to be correct if the target class is among the
        `top_k` largest class probabilities. For the default value of 1, a
        prediction is correct only if the target class is the most probable.
    Returns
    -------
    Theano 1D tensor
        An expression for the item-wise categorical accuracy in {0, 1}
    Notes
    -----
    This is a strictly non differential function as it includes an argmax.
    This objective function should never be used with a gradient calculation.
    It is intended as a convenience for validation and testing not training.
    To obtain the average accuracy, call :func:`theano.tensor.mean()` on the
    result, passing ``dtype=theano.config.floatX`` to compute the mean on GPU.
    """
    if targets.ndim == predictions.ndim:
        targets = theano.tensor.argmax(targets, axis=-1)
    elif targets.ndim != predictions.ndim - 1:
        raise TypeError('rank mismatch between targets and predictions')

    if top_k == 1:
        # standard categorical accuracy
        top = theano.tensor.argmax(predictions, axis=-1)
        return theano.tensor.eq(top, targets)
    else:
        # top-k accuracy
        top = theano.tensor.argsort(predictions, axis=-1)
        # (Theano cannot index with [..., -top_k:], we need to simulate that)
        top = top[[slice(None) for _ in range(top.ndim - 1)] +
                  [slice(-top_k, None)]]
        targets = theano.tensor.shape_padaxis(targets, axis=-1)
        return theano.tensor.any(theano.tensor.eq(top, targets), axis=-1)



# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)

    #My option
    inputs_shuf = []
    targets_shuf = []
    index_shuf = range(len(inputs))
    np.random.shuffle(index_shuf)
    for i in index_shuf:
        inputs_shuf.append(inputs[i])
        targets_shuf.append(targets[i]) 
    
       
    #print("Shuffled data")
    #for i in range (0,3):
    #   print(targets_shuf[i])      
          
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = slice(start_idx, start_idx + batchsize)
                             
        yield inputs_shuf[excerpt], targets_shuf[excerpt]                  
    
    
    
############################################################################         
############################################################################
    
class NormalizeLayer(lasagne.layers.Layer):

    def __init__(self, incoming, axes=None, epsilon=1e-10, alpha='single_pass',
                 return_stats=False, stat_indices=None,
                 **kwargs):
        """
        This layer is a modified version of code originally written by
        Jan Schluter.

        Instantiates a layer performing batch normalization of its inputs [1]_.

        Params
        ------
        incoming: `Layer` instance or expected input shape

        axes: int or tuple of int denoting the axes to normalize over;
            defaults to all axes except for the second if omitted (this will
            do the correct thing for dense layers and convolutional layers)

        epsilon: small constant added to the standard deviation before
            dividing by it, to avoid numeric problems

        alpha: coefficient for the exponential moving average of
            batch-wise means and standard deviations computed during training;
            the larger, the more it will depend on the last batches seen
            If alpha is none we'll assume that the entire training set
            is passed through in one batch.

        return_stats: return mean and std

        stat_indices if slice object only calc stats for these indices. Used
            semisupervsed learning

        Notes
        -----
        This layer accepts the keyword collect=True when get_output is
        called. Before evaluation you should collect the batchnormalizatino
        statistics by running all you data through a function with

        collect=True and deterministic=True

        Then you can evaluate.

        References
        ----------
        .. [1] Ioffe, Sergey, and Christian Szegedy. "Batch normalization:
               Accelerating deep network training by reducing internal
               covariate shift."
               arXiv preprint arXiv:1502.03167 (2015).

        """
        super(NormalizeLayer, self).__init__(incoming, **kwargs)
        self.return_stats = return_stats
        self.stat_indices = stat_indices
        if axes is None:
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        self.epsilon = epsilon
        self.alpha = alpha
        shape = list(self.input_shape)
        broadcast = [False] * len(shape)
        for axis in self.axes:
            shape[axis] = 1
            broadcast[axis] = True
        if any(size is None for size in shape):
            raise ValueError("NormalizeLayer needs specified input sizes for "
                             "all dimensions/axes not normalized over.")
        self.mean = self.add_param(lasagne.init.Constant(0), shape, 'mean',
                                   trainable=False, regularizable=False)
        self.var = self.add_param(lasagne.init.Constant(1), shape, 'var',
                                  trainable=False, regularizable=False)

    def get_output_for(self, input, deterministic=False, collect=False,
                       **kwargs):

        if collect:
            # use this batch's mean and var
            if self.stat_indices is None:
                mean = input.mean(self.axes, keepdims=True)
                var = input.var(self.axes, keepdims=True)
            else:
                mean = input[self.stat_indices].mean(self.axes, keepdims=True)
                var = input[self.stat_indices].var(self.axes, keepdims=True)
            # and update the stored mean and var:
            # we create (memory-aliased) clones of the stored mean and var
            running_mean = theano.clone(self.mean, share_inputs=False)
            running_var = theano.clone(self.var, share_inputs=False)
            # set a default update for them

            if self.alpha is not 'single_pass':
                running_mean.default_update = (
                    (1 - self.alpha) * running_mean + self.alpha * mean)
                running_var.default_update = (
                    (1 - self.alpha) * running_var + self.alpha * var)
            else:
                print "Collecting using single pass..."
                # this is ugly figure out what can be safely removed...
                running_mean.default_update = (0 * running_mean + 1.0 * mean)
                running_var.default_update = (0 * running_var + 1.0 * var)

            # and include them in the graph so their default updates will be
            # applied (although the expressions will be optimized away later)
            mean += 0 * running_mean
            var += 0 * running_var

        elif deterministic:
            # use stored mean and var
            mean = self.mean
            var = self.var
        else:
            # use this batch's mean and var
            mean = input.mean(self.axes, keepdims=True)
            var = input.var(self.axes, keepdims=True)

        mean = T.addbroadcast(mean, *self.axes)
        var = T.addbroadcast(var, *self.axes)
        normalized = (input - mean) / T.sqrt(var + self.epsilon)

        if self.return_stats:
            return [normalized, mean, var]
        else:
            return normalized
            
class ScaleAndShiftLayer(lasagne.layers.Layer):
    """
    This layer is a modified version of code originally written by
    Jan Schluter.

    Used with the NormalizeLayer to construct a batchnormalization layer.

    Params
    ------
    incoming: `Layer` instance or expected input shape

    axes: int or tuple of int denoting the axes to normalize over;
        defaults to all axes except for the second if omitted (this will
        do the correct thing for dense layers and convolutional layers)
    """

    def __init__(self, incoming, axes=None, beta=lasagne.init.Constant(0), gamma=lasagne.init.Constant(1), **kwargs):
        super(ScaleAndShiftLayer, self).__init__(incoming, **kwargs)
        if axes is None:
            # default: normalize over all but the second axis
            axes = (0,) + tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes
        shape = list(self.input_shape)
        broadcast = [False] * len(shape)
        for axis in self.axes:
            shape[axis] = 1
            broadcast[axis] = True
        #if any(size is None for size in shape):
        #    raise ValueError("ScaleAndShiftLayer needs specified input sizes for "
        #                     "all dimensions/axes not normalized over.")
        self.beta = self.add_param(beta, shape, name='beta',
                                   trainable=True, regularizable=True)
        self.gamma = self.add_param(gamma, shape, name='gamma',
                                    trainable=True, regularizable=False)

    def get_output_for(self, input, deterministic=False, **kwargs):
        beta = T.addbroadcast(self.beta, *self.axes)
        gamma = T.addbroadcast(self.gamma, *self.axes)
        return input*gamma + beta

#########################################

#Abbreviations like in example 
conv = lasagne.layers.Conv2DLayer
maxpool = lasagne.layers.MaxPool2DLayer
pool = lasagne.layers.Pool2DLayer
sumlayer = lasagne.layers.ElemwiseSumLayer
nonlin = lasagne.layers.NonlinearityLayer

def convLayer(l, num_filters, filter_size=(3,3), stride=(1,1), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.Uniform(), b=lasagne.init.He, pad_ = 0):
    l = conv(l, filter_size=filter_size, num_filters=num_filters,
		    stride=stride, nonlinearity=None, pad=pad_)
    #l = NormalizeLayer(l)
    l = lasagne.layers.LocalResponseNormalization2DLayer(l, alpha=0.0001)
    l = ScaleAndShiftLayer(l)
    l = nonlin(l, nonlinearity=nonlinearity)
    return l

def bottleneck(l, num_filters, stride=(1,1)):
    l = convLayer(l, num_filters=num_filters, filter_size=(1,1), stride=stride)#pad is 1 (see above)
    l = convLayer(l, num_filters=num_filters, filter_size=(3,3), stride=stride, pad_ = 1)
    l = convLayer(l, num_filters=num_filters*4, filter_size=(1,1), stride=stride)
    return l

def build_ResNet1(input_var=None):
    print("Building ResNet1... ")
    
    l_in = lasagne.layers.InputLayer(shape=(None, 3, crop_size, crop_size),
                                        input_var=input_var)
    l1 = convLayer(l_in, num_filters=16*4, pad_ = 1) #Needs a starting layer, l_in doesnt have dimensionality

    l1_a = sumlayer([bottleneck(l1, num_filters=16), l1])
    l1_b = sumlayer([bottleneck(l1_a, num_filters=16), l1_a])
    l1_c = sumlayer([bottleneck(l1_b, num_filters=16), l1_b])
    l1_c = maxpool(l1_c, pool_size=(2, 2))
    l1_c_residual = convLayer(l1_c, num_filters=32*4, filter_size=(3,3), stride= 1, pad_ =1) #should these also be batch norm?
    
    #l2_a = sumlayer([bottleneck(l1_c, num_filters=32), l1_c_residual])
    #l2_b = sumlayer([bottleneck(l2_a, num_filters=32), l2_a])
    #l2_c = sumlayer([bottleneck(l2_b, num_filters=32), l2_b])
    #l2_c = maxpool(l2_c, pool_size=(2, 2))
    #l2_c_residual = convLayer(l2_c, num_filters=64*4) #should these also be batch norm?
    
    #l3_a = sumlayer([bottleneck(l2_c, num_filters=64), l2_c_residual])
    #l3_b = sumlayer([bottleneck(l3_a, num_filters=64), l3_a])
    #l3_c = sumlayer([bottleneck(l3_b, num_filters=64), l3_b])

    l_out = lasagne.layers.DenseLayer(
                #lasagne.layers.dropout(l1_c, p=.5),
                l1_c,
                num_units=num_of_classes,
                nonlinearity=lasagne.nonlinearities.softmax)

    return l_out
    
    
def build_ResNet2(input_var=None):
    print("Building ResNet1...")
    
    l_in = lasagne.layers.InputLayer(shape=(None, 3, crop_size, crop_size),
                                        input_var=input_var)
    l1 = convLayer(l_in, num_filters=16*4) #Needs a starting layer, l_in doesnt have dimensionality

    l1_a = sumlayer([bottleneck(l1, num_filters=16), l1])
    l1_b = sumlayer([bottleneck(l1_a, num_filters=16), l1_a])
    l1_c = sumlayer([bottleneck(l1_b, num_filters=16*2), l1_b])
    l1_c = maxpool(l1_c, pool_size=(2, 2))
    l1_c_residual = convLayer(l1_c, num_filters=16*8, filter_size=(3,3), stride= 1, pad_ =1) #should these also be batch norm?
    
    #l2_a = sumlayer([bottleneck(l1_c, num_filters=32), l1_c_residual])
    #l2_b = sumlayer([bottleneck(l2_a, num_filters=32), l2_a])
    #l2_c = sumlayer([bottleneck(l2_b, num_filters=32), l2_b])
    #l2_c = maxpool(l2_c, pool_size=(2, 2))
    #l2_c_residual = convLayer(l2_c, num_filters=64*4) #should these also be batch norm?
    
    #l3_a = sumlayer([bottleneck(l2_c, num_filters=64), l2_c_residual])
    #l3_b = sumlayer([bottleneck(l3_a, num_filters=64), l3_a])
    #l3_c = sumlayer([bottleneck(l3_b, num_filters=64), l3_b])

    l_out = lasagne.layers.DenseLayer(
                #lasagne.layers.dropout(l1_c, p=.5),
                l1_c,
                num_units=num_of_classes,
                nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

############################################################################         
############################################################################


def build_cnn2(input_var=None):
    print("building CNN2...")

    # Input layer, as usual:
    #0
    network[0] = lasagne.layers.InputLayer(shape=(None, 3, crop_size, crop_size),
                                        input_var=input_var)   
    #1
    network[1] = lasagne.layers.Conv2DLayer(
              network[0], num_filters=32, filter_size=(3, 3), stride = 1, pad = 1,
              nonlinearity=lasagne.nonlinearities.rectify,
              W=lasagne.init.GlorotNormal(), b=lasagne.init.Uniform())
    #2
    network[2] = lasagne.layers.LocalResponseNormalization2DLayer(network[1], alpha=0.0001)
    
    #3
    network[3] = lasagne.layers.Conv2DLayer(
              network[2], num_filters=64, filter_size=(3,3), pad = 1,
              nonlinearity=lasagne.nonlinearities.rectify,
              W=lasagne.init.GlorotNormal(),b=lasagne.init.Uniform())
    #4: officially not used
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=2, stride=1, ignore_border=False)

    #5
    network[4] = lasagne.layers.LocalResponseNormalization2DLayer(network[3], alpha=0.0001)        
    
    #6
    network[5] = lasagne.layers.Conv2DLayer(#512
              network[4], num_filters=128, filter_size=(3,3), pad=1, 
              nonlinearity=lasagne.nonlinearities.rectify,
              W=lasagne.init.GlorotNormal(),b=lasagne.init.Uniform())
              
    #7
    network[6] = lasagne.layers.Conv2DLayer(
              network[5], num_filters=256, filter_size=3, pad=1, 
              nonlinearity=lasagne.nonlinearities.rectify,
              W=lasagne.init.GlorotNormal(),b=lasagne.init.Uniform())

    #8
    network[7] = lasagne.layers.MaxPool2DLayer(network[6], pool_size=2, stride=2, ignore_border=False)

    #9
    network[8] = lasagne.layers.LocalResponseNormalization2DLayer(network[7], alpha=0.0001) 

    #10
    network[9] = DropoutLayer(network[8], p=0.5)
    
    #11
    network[10] = lasagne.layers.DenseLayer(
              network[9], num_units=256,  #4096
              nonlinearity=lasagne.nonlinearities.rectify)
    
    #12          
    network[11] = DropoutLayer(network[10], p=0.5)          

    #13
    outlayer = lasagne.layers.DenseLayer(
              network[11], num_units=num_of_classes,
              nonlinearity=lasagne.nonlinearities.softmax)

    return network, outlayer

############################################################################

def build_cnn3(input_var=None):
    print("building CNN3...")

    # Input layer, as usual:
    #0
    network[0] = lasagne.layers.InputLayer(shape=(None, 3, crop_size, crop_size),
                                        input_var=input_var)   
    #
    network[1] = lasagne.layers.Conv2DLayer(
              network[0], num_filters=32, filter_size=(3, 3), stride = 1, pad = 1,
              nonlinearity=lasagne.nonlinearities.rectify,
              W=lasagne.init.GlorotNormal(), b=lasagne.init.Uniform())
    #
    network[2] = lasagne.layers.LocalResponseNormalization2DLayer(network[1], alpha=0.0001)
    
    #
    network[3] = lasagne.layers.Conv2DLayer(
              network[2], num_filters=64, filter_size=(3,3), pad = 1,
              nonlinearity=lasagne.nonlinearities.rectify,
              W=lasagne.init.GlorotNormal(),b=lasagne.init.Uniform())
    #4: officially not used
    #network = lasagne.layers.MaxPool2DLayer(network, pool_size=2, stride=1, ignore_border=False)

    #
    network[4] = lasagne.layers.MaxPool2DLayer(network[3], pool_size=2, stride=2, ignore_border=False)
    
    network[5] = lasagne.layers.LocalResponseNormalization2DLayer(network[4], alpha=0.0001)        
    
    #
    network[6] = lasagne.layers.Conv2DLayer(#512
              network[5], num_filters=128, filter_size=(3,3), pad=1, 
              nonlinearity=lasagne.nonlinearities.rectify,
              W=lasagne.init.GlorotNormal(),b=lasagne.init.Uniform())
              
    #
    network[7] = lasagne.layers.Conv2DLayer(
              network[6], num_filters=256, filter_size=3, pad=1, 
              nonlinearity=lasagne.nonlinearities.rectify,
              W=lasagne.init.GlorotNormal(),b=lasagne.init.Uniform())

    #
    network[8] = lasagne.layers.MaxPool2DLayer(network[7], pool_size=2, stride=2, ignore_border=False)

    #
    network[9] = lasagne.layers.LocalResponseNormalization2DLayer(network[8], alpha=0.0001) 

    #
    network[10] = DropoutLayer(network[9], p=0.5)
    
    #
    network[11] = lasagne.layers.DenseLayer(
              network[10], num_units=256,  #4096
              nonlinearity=lasagne.nonlinearities.rectify)
    
    #          
    network[12] = DropoutLayer(network[11], p=0.5)          

    #
    outlayer = lasagne.layers.DenseLayer(
              network[12], num_units=num_of_classes,
              nonlinearity=lasagne.nonlinearities.softmax)

    return network, outlayer

############################################################################

def build_mlp(input_var=None):
    print("building MLP...")
    l_in = lasagne.layers.InputLayer(shape=(None, 3, crop_size, crop_size),
                                     input_var=input_var)

    # Apply 50% dropout to the input data:
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.5)

    # Add a fully-connected layer of 800 units, using the linear rectifier, and
    # initializing weights with Glorot's scheme (which is the default anyway):
    l_hid1 = lasagne.layers.DenseLayer(
            l_in_drop, num_units=400,
            #nonlinearity=lasagne.nonlinearitieqs.rectify,
            W=lasagne.init.GlorotUniform())

    # We'll now add dropout of 50%:
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    # Another 800-unit layer:
    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=400,
            nonlinearity=lasagne.nonlinearities.rectify)

    # 50% dropout again:
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    # Finally, we'll add the fully-connected output layer, of 10 softmax units:
    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=3,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out
    
############################################################################
    
def multiclass_hinge_loss(predictions, targets, delta=1):
    """Computes the multi-class hinge loss between predictions and targets.
    .. math:: L_i = \\max_{j \\not = p_i} (0, t_j - t_{p_i} + \\delta)
    Parameters
    ----------
    predictions : Theano 2D tensor
        Predictions in (0, 1), such as softmax output of a neural network,
        with data points in rows and class probabilities in columns.
    targets : Theano 2D tensor or 1D tensor
        Either a vector of int giving the correct class index per data point
        or a 2D tensor of one-hot encoding of the correct class in the same
        layout as predictions (non-binary targets in [0, 1] do not work!)
    delta : scalar, default 1
        The hinge loss margin
    Returns
    -------
    Theano 1D tensor
        An expression for the item-wise multi-class hinge loss
    Notes
    -----
    This is an alternative to the categorical cross-entropy loss for
    multi-class classification problems
    """
    num_cls = predictions.shape[1]
    if targets.ndim == predictions.ndim - 1:
        targets = theano.tensor.extra_ops.to_one_hot(targets, num_cls)
    elif targets.ndim != predictions.ndim:
        raise TypeError('rank mismatch between targets and predictions')
    corrects = predictions[targets.nonzero()]
    rest = theano.tensor.reshape(predictions[(1-targets).nonzero()],
                                 (-1, num_cls-1))
    rest = theano.tensor.max(rest, axis=1)
    return theano.tensor.nnet.relu(rest - corrects + delta)

############################################################################


def main(model='fede', num_epochs= num_epochs_):
    global network 
    # Load the dataset
    print("Loading data...")
    X_train = []
    y_train = []

    X_test = [] 
    y_test = []
    X_train, y_train, X_test, y_test = load_dataset(png_converter)
    #print(X_train[0])
    mean = compute_mean_image(X_train)  
    #print("mean:")
    #print(mean)
    
    #X_train = prep_image_for_training(X_train, mean)
    #X_test = prep_image_for_training(X_test, mean)
    
    y_train = np.array(y_train).astype('uint8')
    y_test = np.array(y_test).astype('uint8')    
    
    print("just entered the main.")
    
    #print("len x train :" , len(X_train))
    #print("len y train :" , len(y_train))
    #print("len x test :" , len(X_test))
    #print("len y test :" , len(y_test))   
    
    print("BELOW X_train's length:")
    print(len(X_train))
    x = X_train[0]
    print(type(x[0,0,0]))
    #print(X_train[0])
    
    #print("BELOW y_train's sample")
    #for i in range (0,10):
    #      print(y_train[i])
       
    #print("BELOW y_test's sample")
    #for i in range (0,10):
    #      print(y_test[i]) 
    

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs') 
    target_var = T.ivector('targets')

    architecture = { }
    network = { }
    #Possibilities: MLP, CNN2, CNN3, ResNet1
    ##architecture_type = 'CNN1' 
    if architecture_type == 'MLP': network = build_mpl(input_var)
    #if architecture_type == 'CNN1': network = build_cnn1(input_var)
    if architecture_type == 'CNN2': architecture, network = build_cnn2(input_var)
    if architecture_type == 'CNN3': architecture, network = build_cnn3(input_var)
    if architecture_type == 'ResNet1': network = build_ResNet1(input_var)
    global architecture, network
    
    print("===========================================")           
    
    if (toLoadWeights is True):
        print("loading weights...")
        try:
            with np.load('model_' + simulation_name + '.npz') as f:
                 param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(network, param_values)                                     
            
            print ("Done.")
        except:
            print ("Error loading Weights!")                 
   
    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs): 
        global learning_rate_
        learning_rate_ = 0.005 #Could use: update_learn_rate(epoch, learning_rate_)
        print("Learning rate: " + str(learning_rate_))
   
        prediction = lasagne.layers.get_output(network, deterministic = False)
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var) #prev was categorical
        loss = loss.mean()
        #print("loss: ", loss)
        # Could add some weight decay as well here, see lasagne.regularization.

        # Create update expressions for training, i.e., how to modify the
        # parameters at each training step. Here, we'll use Stochastic Gradient
        # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(#nesterov_momentum(
                loss, params, learning_rate=learning_rate_ )
                #, momentum=momentum_) 

        # Create a loss expression for validation/testing. The crucial difference
        # here is that we do a deterministic forward pass through the network,
        # disabling dropout layers.
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
        test_loss2 = lasagne.objectives.squared_error(test_prediction,
                                                            target_var)
        test_loss3 = multiclass_hinge_loss(test_prediction, target_var,1)
        test_loss = test_loss.mean()
        test_loss2 = test_loss.mean()
        test_loss3 = test_loss.mean()
        # As a bonus, also create an expression for the classification accuracy:
        #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
        #                  dtype=theano.config.floatX) 
        test_acc = categorical_accuracy(test_prediction, target_var)
        test_acc = test_acc.mean()
         
        # Compile a function performing a training step on a mini-batch (by giving
        # the updates dictionary) and returning the corresponding training loss:
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        # Compile a second function computing the validation loss and accuracy:
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
        val_fn2 = theano.function([input_var, target_var], [test_loss2, test_acc])
        val_fn3 = theano.function([input_var, target_var], [test_loss3, test_acc])
        #val_fn4 = theano.function([input_var, target_var], [test_loss4, test_acc])

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True): 
            inputs, targets = batch   #iterate_minibatches function yields inputs[excerpt], targets[excerpt]
                                                              
            upgrade = train_fn(inputs, targets)
            train_err = train_err + upgrade
            
            #print(upgrade)#print("train_fn : " , upgrade)  
                            
            train_batches += 1
   
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=True):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("train_err: " + str(train_err))
        print("train_err: " + str(val_err))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
        file = open('model_' + simulation_name +'.txt',"a")
        file.write("tl %f , vl %f , ta %f -" %(train_err / train_batches , val_err / val_batches , val_acc / val_batches * 100))
        file.close()    

    if do_only_testing == 1:                
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
        test_acc = categorical_accuracy(test_prediction, target_var)
        test_acc = test_acc.mean()
        val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=True):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            #val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        #print("Epoch {} of {} took {:.3f}s".format(
        #    epoch + 1, num_epochs, time.time() - start_time))
        #print("train_err: " + str(train_err))
        #print("train_err: " + str(val_err))
        #print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        #print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))

    if save_weights == 1:
        np.savez('model_' + simulation_name + '.npz', *lasagne.layers.get_all_param_values(network))


    # After training, we compute and print the test error:   
    if num_epochs != 0:
      if eval_cat_cross == 1: 
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=True):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results (categorical_crossentropy):")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))
      if eval_sq_err == 1: 
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=True):
            inputs, targets = batch
            err, acc = val_fn2(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results (squared_error):")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))
      if eval_hinge_l == 1:  
        test_err = 0
        test_acc = 0
        test_batches = 0
        for batch in iterate_minibatches(X_test, y_test, batch_size, shuffle=True):
            inputs, targets = batch
            err, acc = val_fn3(inputs, targets)
            test_err += err
            test_acc += acc
            test_batches += 1
        print("Final results (multiclass_hinge_loss):")
        print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
        print("  test accuracy:\t\t{:.2f} %".format(
            test_acc / test_batches * 100))
            
    
    
       
############################################################################## 
    #Confusion matrix
    if show_confusion_matrix == 1:
       error_count = 0
       print("Building confusion matrix...")
       confusion_array = zeros((43,43), float) #should benum_of_classes
       for i in range(0,len(X_test)):#len(X_test)):  #(start,end,step)
           arr = X_test[i]#np.random.random([1,3,16,16])
           #arr = 256* arr                            
           
           #print(arr.shape())
           #print(type(arr))
           arr = arr[np.newaxis]
           net_output = lasagne.layers.get_output(network, arr, deterministic = True)#, deterministic = True)
           arr2 = net_output.eval(#print(net_output.eval())                             
           )
           decision_array = np.argmax(arr2)#arr2.argmax(axis = 1)#index(max(arr2))#unravel_index(net_output.argmax(), net_output.shape())
           #print(arr2)
           maxv = np.float64(0)
           decision = 0
           arr2.astype(float64)
          
           for t in range(num_of_classes):
               #print(maxv) 
               #print(arr2[0,t])                            
               if (arr2[0,t] > maxv): 
                   #print(maxv)
                   maxv = arr2[0,t]
                   decision = t  
           
           if decision != y_test[i]:
              
              #print(arr.shape)
              #arr = scipy.delete(arr, 0, 0)
              ##wrong_im = zeros((crop_size,crop_size,3),dtype=np.uint8)
              ##for e in range(0,2):       
              ##    for q in range (0,crop_size):
              ##        for w in range (0,crop_size): 
                      
              ##            wrong_im[q,w,e] = (arr[0,e,q,w] / 255)           
              #need to revert:                        
              #arr = np.swapaxes(np.swapaxes(arr, 0, 1), 1, 2)              
              #wrong_im = wrong_im[:, :, ::-1]             
              scipy.misc.imsave('errors_' + simulation_name + '/' + 'error_' + str(error_count) + 'cl' + str(y_test[i]) + '.png', test_images2[i])
              error_count = error_count + 1                        
                           
           print("i: " + str(i) + "dec.:" + str(decision))           
           confusion_array[y_test[i], decision] = confusion_array[y_test[i], decision] + 1
       np.set_printoptions(threshold=np.inf) 
       print(np.around(confusion_array, decimals = 2))

       #outfile = TemporaryFile()
       np.savetxt(simulation_name + "_confusion_matrix", confusion_array)
       
       for j in range(num_of_classes):
           row_sum = 0
           for k in range(num_of_classes):
               row_sum = row_sum + confusion_array[j,k]
           for k in range(num_of_classes):
               if row_sum != 0:
                   confusion_array[j,k] = confusion_array[j,k] / row_sum 
       np.set_printoptions(threshold=np.inf)                  
       print(np.around(confusion_array, decimals = 3))  
       print(type(confusion_array[0,0]))

       
  
    # SHOW OUTPUTS - SHOW THE FEATURES OF SPECIFIED LAYERS
    if show_features == 1:
       counter = 0
       for file in glob.glob(homedir + "*.ppm"):
           print("found " + str(counter))
           im_temp = Image.open(file)

           im_temp.save(homedir + 'im' + str(counter) + '.png')
           #im_temp = Image.open('im' + str(counter) + '.png')
           #fpath = BytesIO() 
           #im_temp.save(fpath,'png')       
        
           counter = counter + 1
          
       counter = 0
       for file in glob.glob(homedir + "*.png"):
           for i in range(0,len(layer_n_feats_arr)):
               layer_n_feats = layer_n_feats_arr[i]
               #if counter < num_images:
               print("=========================================")
               print("Now final testing images", counter ) 
               print(file)
          
               im = prep_image_for_test_A(file)
               rawim, im = prep_image_for_test_B(im, 1)    
               plt.imshow(rawim)
               show()
           
               im = prepare_one_image_for_test(im, mean)             
           
               print("Now eval features:")
               net_output = lasagne.layers.get_output(architecture[layer_n_feats], im)
               print(str(type(net_output)))
               shape_net_output = lasagne.layers.get_output_shape(architecture[layer_n_feats])
               print( "Shape of network output: %s"%(shape_net_output,)) 
               x_eval = net_output.eval()
               ratio = shape_net_output[1]/8
           
               conv1 = []
               for x_x  in range(0, shape_net_output[1]):                                                                                                          
                   pic = np.random.random([shape_net_output[2],shape_net_output[3]])           
                   for i in range(0,shape_net_output[2]):
                       for j in range(0,shape_net_output[3]):                                          
                           pic[i,j] = x_eval[0,x_x,i,j]              
                   conv1.append(pic) 
    
               fig, ax = plt.subplots(ratio, ncols=8, sharex='col', sharey='row', figsize = (8,ratio)) #sharex=True, sharey=False
               print('\n' + '==============Layer ' + str(layer_n_feats) + ' features==============')
               for i in xrange(0,shape_net_output[1]):                   
                      ax[(i)/8][(i)%8].imshow(conv1[i],interpolation = 'nearest', 
                                cmap = 'Greys_r', aspect = '1')#prev.: aspect = 'auto'
                      
                      ax[(i)/8][(i)%8].set_ylim([0,shape_net_output[2]]) #prev from 0 to 2
                      plt.xlim(0,shape_net_output[3]); plt.ylim(0,shape_net_output[2])
                              
               font = {'family' : 'normal',
                       'weight' : 'bold',
                       'size'   : 0}

               matplotlib.rc('font', **font)
               counter = counter + 1
    
    
    # SHOW OUTPUTS - CLASSIFY AN IMAGE
    if show_examples == 1:
       counter = 0
       for file in glob.glob(homedir + "*.ppm"):
           
           im_temp = Image.open(file)

           im_temp.save('im' + str(counter) + '.png')
           im_temp = Image.open('im' + str(counter) + '.png')

           fpath = BytesIO() 
           im_temp.save(fpath,'png')       
        
           counter = counter + 1
          
       counter = 0
       for file in glob.glob(homedir + "*.png"):
           #if counter < num_images:
           print("===========================================")
           print("Now displaying filters", counter ) 
           print(file)
          
           im = prep_image_for_test_A(file)
           rawim, im = prep_image_for_test_B(im, 1)    
           plt.imshow(rawim)
           show()
           
           im = prepare_one_image_for_test(im, mean)             
           
           print("Now eval output:")
           net_output = lasagne.layers.get_output(network, im, deterministic = True)
           print( net_output.eval()) 
           counter = counter + 1
           
        
    # SHOW THE FILTERS OF A SPECIFIED LAYER       
    if show_filters == 1:       
        for i in range(0,50):#len(layer_n_filters_arr)):
           layer_n_filters = layer_n_filters_arr[i] 
           f = open('layers' + '.weights' + str(layer_n_filters),'wb')#open('layer' + str(layercounter) + '.weights','wb')
           print('checkpoint C')
           weights = architecture[layer_n_filters].W.get_value()
           
           print("weights: " + str(weights.shape[0]) + " " + str(weights.shape[1]) + " " + str(weights.shape[2]) + " " + str(weights.shape[3]) )
           # assuming W shape is (num_filters, 3, height, width):
           weights = weights.reshape(weights.shape[0]*weights.shape[1],weights.shape[2],weights.shape[3]) #
           #weights[0]
           #for i in range(weights.shape[0]):
           #    wmin = float(weights[i].min())
           #    wmax = float(weights[i].max())
           
               
           np.save(f, weights)
           f.close()          

           with open('layers.weights' + str(layer_n_filters), 'rb') as f:
               conv1 = np.load(f)
               shape_net_output = lasagne.layers.get_output_shape(architecture[layer_n_filters])
               print(shape_net_output)
               ratio = shape_net_output[1]/16
               
               fig, ax = plt.subplots(ratio*3, ncols=16, sharex=True, sharey=False) #first conv layer has 96 filters
               
               for i in xrange(0,shape_net_output[1]*3):#shape_net_output[1]):                   
                   ax[(i)/16][(i)%16].imshow(conv1[i])
                   ax[(i)/16][(i)%16].set_ylim([0,2])                    
          
               plt.xlim(-1,3); plt.ylim(-1,3)   
               
               plt.gca().xaxis.set_major_locator(plt.NullLocator())
               plt.gca().yaxis.set_major_locator(plt.NullLocator())
               #plt.tick_params(
               #           axis = 'both',
               #           length = 0,
               #           which = 'both',
               #           bottom = 'off',
               #           top = 'off'
               #                  )               
               #plt.tick_params(
               #    axis = 'y',
               #    which = 'both',
               #    bottom = 'off',
               #    top = 'off',
               #               )               
                
               font = {'family' : 'normal',
                           'weight' : 'bold',
                           'size'   : 0}

               matplotlib.rc('font', **font)
                
               plt.show()
               f.close()

############################################################################

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

############################################################################


