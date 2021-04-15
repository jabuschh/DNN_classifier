#!/usr/bin/env python

import numpy as np
from numpy import inf
import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from IPython.display import FileLink, FileLinks
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical, plot_model
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import math
import pickle
from Training import *
from Plotting import *
from GetInputs import *
from RankNetworks import *
from PredictExternal import *
from functions import *
#from TrainModelOnPredictions import *
#from TrainSecondNetwork import *
#from TrainThirdNetwork import *
from ExportModel import *

# ignore weird Warnings: "These warnings are visible whenever you import scipy
# (or another package) that was compiled against an older numpy than is installed."
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
# ignore weird Warnings: "These warnings are visible whenever you import scipy
# (or another package) that was compiled against an older numpy than is installed."
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)




# values for grid scan over the parameters
jobid = int(os.getenv('NR')) # = 0, 1, 2, ... (as given by HTCondor)
print "jobid: " + str(jobid)

values_layers        = [[128, 128], [256, 256], [512, 512]]
values_batchsize     = [32768, 131072, 524288] # 2^15, 2^17, 2^19
values_regrate       = [0.3, 0.4, 0.5]
values_epochs        = [500]
values_learningrate  = [0.0003, 0.0005, 0.0007]
values_runonfraction = [1.00]
values_eqweight      = [False]
values_preprocess    = ['MinMaxScaler', 'StandardScaler']

index_layers        =  (jobid) %  len(values_layers)
index_batchsize     = ((jobid) / (len(values_layers))) % len(values_batchsize)
index_regrate       = ((jobid) / (len(values_layers))  / (len(values_batchsize))) % len(values_regrate)
index_epochs        = ((jobid) / (len(values_layers))  / (len(values_batchsize))  / (len(values_regrate))) % len(values_epochs)
index_learningrate  = ((jobid) / (len(values_layers))  / (len(values_batchsize))  / (len(values_regrate))  / (len(values_epochs))) % len(values_learningrate)
index_runonfraction = ((jobid) / (len(values_layers))  / (len(values_batchsize))  / (len(values_regrate))  / (len(values_epochs))  / (len(values_learningrate))) % len(values_runonfraction)
index_eqweight      = ((jobid) / (len(values_layers))  / (len(values_batchsize))  / (len(values_regrate))  / (len(values_epochs))  / (len(values_learningrate))  / (len(values_runonfraction))) % len(values_eqweight)
index_preprocess    = ((jobid) / (len(values_layers))  / (len(values_batchsize))  / (len(values_regrate))  / (len(values_epochs))  / (len(values_learningrate))  / (len(values_runonfraction))  / (len(values_eqweight))) % len(values_preprocess)

variations = ['NOMINAL'] # 'NOMINAL','JEC_up','JEC_down','JER_up','JER_down'
merged_str = 'Merged'
parameters = {
    'layers'       : values_layers[index_layers], # Ksenia: 512, 512
    'batchsize'    : values_batchsize[index_batchsize], # 2^17 = 131072
    'classes'      : {0: ['TTbar'], 1: ['ST'], 2:['WJets','DY']}, # signal-agnostic + multiclass, Diboson as single category is not learned -> leave out
    'regmethod'    : 'dropout',
    'regrate'      : values_regrate[index_regrate], # dropout rate: Ksenia: 0.5
    'batchnorm'    : False,
    'epochs'       : values_epochs[index_epochs], # 500 are enough?
    'learningrate' : values_learningrate[index_learningrate],
    'runonfraction': values_runonfraction[index_runonfraction], # run on 100% of the MC samples with: 60% training, 20% validation, 20% testing (see GetInputs.py)
    'eqweight'     : values_eqweight[index_eqweight], #
    'preprocess'   : values_preprocess[index_preprocess], # MinMaxScaler or StandardScaler
    'sigma'        : 1.0, #sigma for Gaussian prior (BNN only)
    'inputdir'     : '/nfs/dust/cms/user/jabuschh/NonResonantTTbar/RunII_102X_v2/MLinputs_numpy_myDNN_2018muon/', # path to input files: inputdir + systvar + inputsubdir
    'inputsubdir'  : 'MLInput/', #path to input files: inputdir + systvar + inputsubdir
    'prepreprocess': 'RAW' #for inputs with systematics don't do preprocessing before merging all inputs on one,     #FixME: add prepreprocessing in case one does not need to merge inputs
}

tag = dict_to_str(parameters)
classtag = get_classes_tag(parameters)

########## GetInputs ########
for ivars in range(len(variations)):
     merged_str = merged_str+'__'+variations[ivars]
     parameters['systvar'] = variations[ivars]
     # # # # # # Get all the inputs
     # # # # # # # # # ==================
     inputfolder = parameters['inputdir']+parameters['inputsubdir']+parameters['systvar']+'/'+parameters['prepreprocess']+'/'+ classtag
     GetInputs(parameters)
     PlotInputs(parameters, inputfolder=inputfolder, filepostfix='', plotfolder='Plots/'+parameters['prepreprocess']+'/InputDistributions/'+parameters['systvar']+'/' + classtag)

MixInputs(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, variations=variations, filepostfix='')
SplitInputs(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='')
FitPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='')
ApplyPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='train')
ApplyPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='test')
ApplyPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='',setid='val')
ApplySignalPrepocessing(parameters, outputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag, filepostfix='')

inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/' + classtag
outputfolder='output/'+parameters['preprocess']+'/'+merged_str+'/' + classtag+'/DNN_'+tag
plotfolder = 'Plots/'+parameters['preprocess']
PlotInputs(parameters, inputfolder=inputfolder, filepostfix='', plotfolder=plotfolder+'/InputDistributions/'+merged_str+'/' + classtag)

########

# # DNN
TrainNetwork(parameters, inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+tag)
PredictExternal(parameters, inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+tag, filepostfix='')
PlotPerformance(parameters, inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+merged_str+'/'+classtag, outputfolder='output/'+parameters['preprocess']+'/DNN_'+tag, filepostfix='', plotfolder='Plots/'+parameters['preprocess']+'/DNN_'+tag, use_best_model=True, usesignals=[2,4])


##Test training on one and prediction on another set
#input_var = 'Merged__NOMINAL'
#training_var = 'Merged__NOMINAL'
#inputfolder=parameters['inputdir']+parameters['preprocess']+'/'+input_var+'/' + classtag
#outputfolder='output/'+parameters['preprocess']+'/'+training_var+'/' + classtag+'/DNN_'+tag
#plotfolder = 'Plots/'+parameters['preprocess']
#PredictExternal(parameters, inputfolder=inputfolder, outputfolder=outputfolder, filepostfix='')
#PlotPerformance(parameters, inputfolder=inputfolder, outputfolder=outputfolder, filepostfix='', plotfolder=plotfolder+'/Output/Input_'+input_var+'_Training_'+training_var+'/'+'/DNN_'+tag, use_best_model=True, usesignals=[2,4])
