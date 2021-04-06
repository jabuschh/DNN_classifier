import numpy as np
from numpy import inf
import keras
import matplotlib
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from IPython.display import FileLink, FileLinks
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical, plot_model
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
import pickle
import os
from functions import *
from shutil import copyfile

import pandas as pd
import gzip

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def GetInputs(parameters):

    prepreprocess = 'RAW' #for inputs with systematics don't do preprocessing before merging all inputs on one
    #FixME: add prepreprocessing in case one does not need to merge inputs
    # Get parameters
    classes = parameters['classes']
    eqweight = parameters['eqweight']
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    classtag = get_classes_tag(parameters)
    inputdir = parameters['inputdir']
    systvar = parameters['systvar']
    inputsubdir = parameters['inputsubdir']
    #path to input files: inputdir + systvar + inputsubdir

    if os.path.isdir(inputdir+inputsubdir+systvar+'/'+prepreprocess+'/'+ classtag):
        if os.path.isfile(inputdir+inputsubdir+systvar+'/'+prepreprocess+'/'+ classtag+ '/input_' + fraction + '_val.npy'):
            # print 'These inputfiles already exist, go on to next function.'
            # return
            pass
        else:
            pass
    else:
        os.makedirs(inputdir+inputsubdir+systvar+'/'+prepreprocess+'/'+ classtag)

    maxfiles_per_sample = {'TTbar': -1, 'WJets': -1,'DY': -1,'ST': -1, 'Diboson': -1}

    # Find initial file for each class
#    inputfiles = os.listdir('input/MLInput')
#    inputfiles = os.listdir(inputdir+systvar+inputsubdir)
    inputfiles = os.listdir(inputdir+inputsubdir+systvar)

    # list of numpy.array, containing the inputs for all classes. Will have len() = number of classes = len(classes)
    all_inputs = {}
    all_labels = {}
    all_eventweights = {}
    for cl in classes.keys():
        first = True

        # Get list of input files for this class, it's a list of lists --> one list per sample belonging to this class
        lists_of_inputfiles = []
        for i in range(len(classes[cl])):
            tmp = []
            sample = classes[cl][i]
            idx = 0
            for j in range(len(inputfiles)):
                if classes[cl][i]+'_' in inputfiles[j] and not 'Weights_' in inputfiles[j] and '.npy' in inputfiles[j] and (idx<maxfiles_per_sample[sample] or maxfiles_per_sample[sample]<0):
                    tmp.append(inputfiles[j])
                    idx += 1
            lists_of_inputfiles.append(tmp)
        print lists_of_inputfiles

        # Read files for this class
#        print 'Read files from: ',inputdir+systvar+inputsubdir
        print 'Read files from: ',inputdir+inputsubdir+systvar
        for i in range(len(lists_of_inputfiles)):
            print '\nNow starting with sample %s' % (classes[cl][i])
            for j in range(len(lists_of_inputfiles[i])):
                print 'At file no. %i out of %i.' % (j+1, len(lists_of_inputfiles[i]))
                if first:
#                    thisinput = np.load(inputdir+systvar+inputsubdir + lists_of_inputfiles[i][j])
#                    thiseventweight = np.load(inputdir+systvar+inputsubdir +'Weights_' + lists_of_inputfiles[i][j])
                    thisinput = np.load(inputdir+inputsubdir+systvar +'/'+ lists_of_inputfiles[i][j])
                    thiseventweight = np.load(inputdir+inputsubdir+systvar+'/'+'Weights_' + lists_of_inputfiles[i][j])

                    first = False
                else:
                    # thisinput = np.concatenate((thisinput, np.load(inputdir+systvar+inputsubdir + lists_of_inputfiles[i][j])))
                    # thiseventweight = np.concatenate((thiseventweight, np.load(inputdir+systvar+inputsubdir+'Weights_' + lists_of_inputfiles[i][j])))
                    thisinput = np.concatenate((thisinput, np.load(inputdir+inputsubdir+systvar+'/' + lists_of_inputfiles[i][j])))
                    thiseventweight = np.concatenate((thiseventweight, np.load(inputdir+inputsubdir+systvar+'/'+'Weights_' + lists_of_inputfiles[i][j])))

        # thisinput = thisinput.astype(np.float32)
        # thiseventweight = thiseventweight.astype(np.float32)
        all_inputs[cl] = thisinput
        all_eventweights[cl] = thiseventweight

        # Fill the class i with label i
        thislabel = np.empty(thisinput.shape[0])
        thislabel.fill(cl)
        thislabel = thislabel.astype(np.int8)
        all_labels[cl] = thislabel

#    # now read in signal
#    signal_masses = [1000, 2000, 3000, 4000, 5000, 6000]
#    signal_identifiers = ['RSGluon_All']
#    for mass in signal_masses:
#        signal_identifiers.append('RSGluon_M' + str(mass))
    all_signals = {}
    all_signal_eventweights = {}
    lists_of_inputfiles_sig = []
#    for i in range(len(signal_identifiers)):
#        tmp = []
#        sample = signal_identifiers[i]
#        idx = 0
#        for j in range(len(inputfiles)):
#            if signal_identifiers[i]+'_' in inputfiles[j] and not 'Weights_' in inputfiles[j] and '.npy' in inputfiles[j]:
#                tmp.append(inputfiles[j])
#                idx += 1
#        lists_of_inputfiles_sig.append(tmp)
#    print lists_of_inputfiles_sig

    # Read files for this class
#    for i in range(len(lists_of_inputfiles_sig)):
#        print '\nNow starting with sample %s' % (signal_identifiers[i])
#        first = True
#        for j in range(len(lists_of_inputfiles_sig[i])):
#            print 'At file no. %i out of %i.' % (j+1, len(lists_of_inputfiles_sig[i]))
#            if first:
#                # thisinput = np.load(inputdir+systvar+inputsubdir+ lists_of_inputfiles_sig[i][j])
#                # thiseventweight = np.load(inputdir+systvar+inputsubdir+'Weights_' + lists_of_inputfiles_sig[i][j])
#                thisinput = np.load(inputdir+inputsubdir+systvar+'/'+ lists_of_inputfiles_sig[i][j])
#                thiseventweight = np.load(inputdir+inputsubdir+systvar+'/'+'Weights_' + lists_of_inputfiles_sig[i][j])
#
#                first = False
#            else:
#                # thisinput = np.concatenate((thisinput, np.load(inputdir+systvar+inputsubdir + lists_of_inputfiles_sig[i][j])))
#                # thiseventweight = np.concatenate((thiseventweight, np.load(inputdir+systvar+inputsubdir+ 'Weights_' + lists_of_inputfiles_sig[i][j])))
#                thisinput = np.concatenate((thisinput, np.load(inputdir+inputsubdir+systvar+'/'+ lists_of_inputfiles_sig[i][j])))
#                thiseventweight = np.concatenate((thiseventweight, np.load(inputdir+inputsubdir+systvar+'/'+ 'Weights_' + lists_of_inputfiles_sig[i][j])))
#
#        # thisinput = thisinput.astype(np.float32)
#        # thiseventweight = thiseventweight.astype(np.float32)
#        all_signals[i] = thisinput
#        all_signal_eventweights[i] = thiseventweight


    if len(all_inputs) != len(classes) or len(all_labels) != len(classes) or len(all_labels) != len(all_eventweights):
        raise ValueError('Number of input classes or labels or eventweights read in does not match number of classes defined in GetInputs().')


    # Here we're making sure to loop through all classes in the numeric order to avoid confusing the labels of inputs -- dict might be unordered, but the input matrix has to be ordered! Thanks god the class names correspond to the list indices from 0 to nclasses-1
    label_concatenated = np.concatenate((tuple([all_labels[i] for i in range(len(all_labels))])))
    input_total = np.concatenate((tuple([all_inputs[i] for i in range(len(all_inputs))])))
    eventweight_total = np.concatenate((tuple([all_eventweights[i] for i in range(len(all_eventweights))])))
    # signal_total = np.concatenate((tuple([all_signals[i] for i in range(len(all_signals))])))
    # signal_eventweight_total = np.concatenate((tuple([all_signal_eventweights[i] for i in range(len(all_signal_eventweights))])))

    # Now create matrix with labels, it's zero everywhere, only the column corresponding to the class the example belongs to has ones
    labels_total = np.zeros((label_concatenated.shape[0], len(classes)))
    for i in range(label_concatenated.shape[0]):
        label = label_concatenated[i]
        labels_total[i,label] = 1
    labels_total = labels_total.astype(np.int8)

    # Treat inf entries
    input_total[input_total == inf]    = 999999.
    input_total[input_total == -inf]   = -999999.
    input_total[np.isnan(input_total)] = 0.
    # signal_total[signal_total == inf]    = 999999.
    # signal_total[signal_total == -inf]   = -999999.
    # signal_total[np.isnan(signal_total)] = 0.

    # print input_total[labels_total[:,2]==1][0]

    shuffle = np.random.permutation(np.size(input_total, axis=0))
    input_total       = input_total[shuffle]
    labels_total      = labels_total[shuffle]
    eventweight_total = eventweight_total[shuffle]
    label_concatenated = label_concatenated[shuffle]
#    for i in all_signals.keys():
#        shuffle_signal = np.random.permutation(np.size(all_signals[i], axis=0))
#        all_signals[i]       = all_signals[i][shuffle_signal]
#        all_signal_eventweights[i] = all_signal_eventweights[i][shuffle_signal]

    # Cut off some events if not running on full sample
    # percentage = 0.01
    percentage = runonfraction
    frac_train = 0.6 * percentage
    frac_test  = 0.2 * percentage
    frac_val   = 0.2 * percentage

    sumweights = np.sum(eventweight_total, axis=0)
    print 'shape of all inputs: ', input_total.shape
    print 'shape and sum of event weights: ', eventweight_total.shape, sumweights
    cutoffweighted_train = float(sumweights)*float(frac_train)
    cutoffweighted_test  = float(sumweights)*float(frac_train + frac_test)
    cutoffweighted_val   = float(sumweights)*float(frac_train + frac_test + frac_val)

    currentsum = 0.
    takeupto_train = 0
    takeupto_test  = 0
    takeupto_val   = 0
    sumweights_classes = {}
    # initialize this dict
    for i in range(labels_total.shape[1]):
        sumweights_classes[i] = 0.

    for i in range(len(eventweight_total)):
        currentsum += eventweight_total[i,0]
        # if i%1000000 == 0: print i, currentsum
        if currentsum >= cutoffweighted_train and takeupto_train == 0:
            takeupto_train = i+1
        if currentsum >= cutoffweighted_test  and takeupto_test  == 0:
            takeupto_test = i+1
        if currentsum >= cutoffweighted_val  and takeupto_val  == 0:
            takeupto_val = i+1

        #find out which class this event belongs to
        thisclass = label_concatenated[i]
        sumweights_classes[thisclass] += eventweight_total[i,0]

    print 'takeupto_(train/test/val): ' , takeupto_train, takeupto_test, takeupto_val
    input_train = input_total[:takeupto_train]
    labels_train = labels_total[:takeupto_train]
    eventweight_train = eventweight_total[:takeupto_train]
    input_test = input_total[takeupto_train:takeupto_test]
    labels_test = labels_total[takeupto_train:takeupto_test]
    eventweight_test = eventweight_total[takeupto_train:takeupto_test]
    input_val = input_total[takeupto_test:takeupto_val]
    labels_val = labels_total[takeupto_test:takeupto_val]
    eventweight_val = eventweight_total[takeupto_test:takeupto_val]
    print 'shapes of inputs (train, test, val): ', input_train.shape, input_test.shape, input_val.shape

    # Calculate class weights such, that after weighting by class_weight all classes have the same number of weighted events,
    # where all events are ALSO weighted by eventweight --> total weight = class_weight * eventweight
    class_weights = {}
    # scale each class to the one with the smallest sum of weights
    minsum = sumweights_classes[0]
    for i in range(len(sumweights_classes)):
        if sumweights_classes[i] < minsum: minsum = sumweights_classes[i]
        # print sumweights_classes[i]

    for i in range(len(sumweights_classes)):
        weight = 1
        if sumweights_classes[i] != 0: weight = minsum/sumweights_classes[i]
        class_weights[i] = weight
        # print weight

    sample_weights_train_list = []
    sample_weights_test_list = []
    sample_weights_val_list = []
    for i in range(len(labels_train[:,0])):
        #loop over training examples i
        for j in range(len(labels_train[i,:])):
            #loop over possible classes j
            if labels_train[i,j] == 1:
                thisweight = class_weights[j] * eventweight_train[i]
                sample_weights_train_list.append(thisweight)
    for i in range(len(labels_test[:,0])):
        for j in range(len(labels_test[i,:])):
            if labels_test[i,j] == 1:
                thisweight = class_weights[j] * eventweight_test[i]
                sample_weights_test_list.append(thisweight)
    for i in range(len(labels_val[:,0])):
        for j in range(len(labels_val[i,:])):
            if labels_val[i,j] == 1:
                thisweight = class_weights[j] * eventweight_val[i]
                sample_weights_val_list.append(thisweight)


    # Test: sum val-sampleweights for each class, should be the same value for all classes
    sums = {0:0., 1:0., 2:0., 3:0., 4:0.}
    for i in range(len(labels_val[:,0])):
        #loop over training examples i
        for j in range(len(labels_val[i,:])):
            #loop over possible classes j
            if labels_val[i,j] == 1:
                sums[j] += sample_weights_val_list[i]

    sample_weights_train = np.asarray(sample_weights_train_list).ravel()
    sample_weights_test = np.asarray(sample_weights_test_list).ravel()
    sample_weights_val = np.asarray(sample_weights_val_list).ravel()

    eventweight_train = np.asarray(eventweight_train).ravel()
    eventweight_test  = np.asarray(eventweight_test).ravel()
    eventweight_val   = np.asarray(eventweight_val).ravel()
#    for i in all_signal_eventweights.keys():
#        all_signal_eventweights[i] = np.asarray(all_signal_eventweights[i]).ravel()

    classtag = get_classes_tag(parameters)
#    with open(inputdir+systvar+inputsubdir+ 'variable_names.pkl', 'r') as f:
    with open(inputdir+inputsubdir+systvar+'/'+ 'variable_names.pkl', 'r') as f:
        variable_names = pickle.load(f)

  #   #### TEST without scaling
#     # Scale features

#     #print "mean = ", np.mean(input_train, axis=0)[0]
#     #print "std = ", np.std(input_train, axis=0)[0]

#     #print "scaler.mean_ =", scaler.mean_[0]
#     #print "scaler.scale_ = ",scaler.scale_[0]

#     # scaler = preprocessing.StandardScaler()
#     # scaler.mean_ = np.mean(input_train, axis=0)
#     # scaler.scale_ = np.std(input_train, axis=0)

# #    scaler = preprocessing.StandardScaler().fit(input_train)
# #    scaler = preprocessing.QuantileTransformer(output_distribution='normal').fit(input_train)
#     if(parameters['preprocess'] == 'StandardScaler'):
#         print " === StandardScaler preprocessing ==="
#         scaler = preprocessing.StandardScaler().fit(input_train)
#     elif(parameters['preprocess'] == 'QuantileTransformerUniform'):
#         print " === QuantileTransformer(Uniform) preprocessing ==="
#         scaler = preprocessing.QuantileTransformer(output_distribution='uniform').fit(input_train)
#     elif(parameters['preprocess'] == 'MinMaxScaler'):
#         print " === MinMaxScaler preprocessing ==="
#         scaler = preprocessing.MinMaxScaler().fit(input_train)
#     else:
#         print("preprocess set to unknown value! going to use standart StandardScaler preprocessing")
#         scaler = preprocessing.StandardScaler().fit(input_train)

# #    scaler =  preprocessing.PowerTransformer(method='yeo-johnson').fit(input_train)
# #    scaler =  preprocessing.RobustScaler(quantile_range=(25, 75)).fit(input_train)

#     input_train = deepcopy(scaler.transform(input_train))
#     input_test = deepcopy(scaler.transform(input_test))
#     input_val = deepcopy(scaler.transform(input_val))
#     for i in all_signals.keys():
#         all_signals[i] = deepcopy(scaler.transform(all_signals[i]))

#     # Write out scaler info
#     with open(inputdir+systvar+inputsubdir+parameters['preprocess']+'/'+classtag+'/NormInfo.txt', 'w') as f:
#         #for i in range(scaler.mean_.shape[0]):
#         for i in range(np.mean(input_train, axis=0).shape[0]): #valid only for StandardScaler, placeholder for the rest
#             var = variable_names[i]
#             #mean = scaler.mean_[i]
#             #scale = scaler.scale_[i]
#             mean = np.mean(input_train, axis=0)[i] #valid only for StandardScaler, placeholder for the rest
#             scale = np.std(input_train, axis=0)[i] #valid only for StandardScaler, placeholder for the rest
#             line = var + ' StandardScaler ' + str(mean) + ' ' + str(scale) + '\n'
#             f.write(line)
#     ### END Scaling

    output_path = inputdir+inputsubdir+systvar+'/'+prepreprocess+'/'+classtag
    # output_path =inputdir+systvar+inputsubdir+prepreprocess+'/'+classtag
    print 'Store files in ',output_path
    with open(output_path+'/variable_names.pkl', 'w') as f:
        pickle.dump(variable_names, f)

    print "STORE: input_train[0] = ", input_train[0]
    np.save(output_path+'/input_'+fraction+'_train.npy'  , input_train)
    np.save(output_path+'/input_'+fraction+'_test.npy'   , input_test)
    np.save(output_path+'/input_'+fraction+'_val.npy'    , input_val)
    np.save(output_path+'/labels_'+fraction+'_train.npy' , labels_train)
    np.save(output_path+'/labels_'+fraction+'_test.npy'  , labels_test)
    np.save(output_path+'/labels_'+fraction+'_val.npy'   , labels_val)

    np.save(output_path+'/sample_weights_'+fraction+'_train.npy', sample_weights_train)
    np.save(output_path+'/eventweights_'+fraction+'_train.npy', eventweight_train)
    np.save(output_path+'/sample_weights_'+fraction+'_test.npy', sample_weights_test)
    np.save(output_path+'/eventweights_'+fraction+'_test.npy', eventweight_test)
    np.save(output_path+'/sample_weights_'+fraction+'_val.npy', sample_weights_val)
    np.save(output_path+'/eventweights_'+fraction+'_val.npy', eventweight_val)



#    for i in all_signals.keys():
#        np.save(output_path+'/'+signal_identifiers[i]+'.npy', all_signals[i])
#        np.save(output_path+'/'+signal_identifiers[i]+'_eventweight.npy', all_signal_eventweights[i])


def MixInputs(parameters, outputfolder, variations, filepostfix):
    print("****** MixInputs ******")
    # Get parameters
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    classtag = get_classes_tag(parameters)
    tag = dict_to_str(parameters)
    if not os.path.isdir(outputfolder):
        os.makedirs(outputfolder)

    #input_array_all = np.ones((input_train.shape[0]+input_test.shape[0]+input_val.shape[0], input_train.shape[1]+labels_sample.shape[1]+2))
    input_array_all = np.ones((1,1))
    input_train_shape1 = 0 #to split concatenated array
    labels_train_shape_1 = 0 #to split concatenated array

    #input_signal_array_all = np.ones((1,1,1))
    input_signal_array_all = {}
    print "bool(input_signal_array_all) = ",bool(input_signal_array_all)
    for isyst in range(len(variations)):
#        inputfolder = parameters['inputdir']+variations[isyst]+parameters['inputsubdir']+parameters['prepreprocess']+'/'+ classtag
        inputfolder = parameters['inputdir']+parameters['inputsubdir']+variations[isyst]+'/'+parameters['prepreprocess']+'/'+ classtag

        # Get inputs
        input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, eventweight_signals, normweight_signals = load_data(parameters, inputfolder=inputfolder, filepostfix=filepostfix)
        input_train_shape1 = input_train.shape[1]
        labels_train_shape_1 = labels_train.shape[1]
        print 'input_.shape: ',input_train.shape, input_test.shape, input_val.shape
        input_sample = np.concatenate((input_train,input_test,input_val), axis=0)
        labels_sample = np.concatenate((labels_train,labels_test,labels_val), axis=0)
        eventweights_sample = np.concatenate((eventweights_train,eventweights_test,eventweights_val), axis=0)
        eventweights_sample = eventweights_sample.reshape((eventweights_sample.shape[0], 1))

        #FixMe sample weigtht containts weight to equalize the samples and calculated per systematic variation
        #for merged samples it should be recalculated. However using equally weighted samples does not give advantage,
        #thus this variable is not going to be used in the nearest future and left it as it's now
        sample_weights_sample = np.concatenate((sample_weights_train,sample_weights_test,sample_weights_val), axis=0)
        sample_weights_sample = sample_weights_sample.reshape((sample_weights_sample.shape[0],1))
   #     input_array = np.concatenate((input_sample,labels_sample,sample_weights_sample,eventweights_sample), axis=1) #array with all backgrounds at one place

#        input_array = np.concatenate((input_sample,labels_sample,eventweights_sample), axis=1) #array with all backgrounds at one place
        input_array = np.concatenate((input_sample,labels_sample,eventweights_sample,sample_weights_sample), axis=1) #array with all backgrounds at one place
        if input_array_all[0,0]!=1:
            tmp_input_array = np.append(input_array_all, input_array,axis=0)
            input_array_all.resize(tmp_input_array.shape[0],tmp_input_array.shape[1])
            input_array_all = tmp_input_array.copy()
        else:
            input_array_all = np.asarray(input_array).copy()

            input_signal_array = {}
#            for i in signals.keys():
#                eventweight_signals[i] = eventweight_signals[i].reshape(eventweight_signals[i].shape[0],1)
#                normweight_signals[i] = normweight_signals[i].reshape(normweight_signals[i].shape[0],1)
#                # print signals[i].shape, eventweight_signals[i].shape, normweight_signals[i].shape
#                #input_signal_array[i] = np.concatenate((signals[i], eventweight_signals[i], normweight_signals[i]), axis=1) #array with all signals at one place
#                input_signal_array[i] = np.concatenate((signals[i], eventweight_signals[i],normweight_signals[i]), axis=1) #array with all signals at one place
#                if isyst>0:
#                    tmp_input_signal_array = np.append(input_signal_array_all[i], input_signal_array[i],axis=0)
#                    input_signal_array_all[i].resize(tmp_input_signal_array.shape[0],tmp_input_signal_array.shape[1])
#                    input_signal_array_all[i] = tmp_input_signal_array.copy()
#                else:
#                    input_signal_array_all[i] = np.asarray(input_signal_array[i]).copy()

        #cleaning
        del input_sample
        del labels_sample
        #del sample_weights_sample
        del eventweights_sample
        del input_array



#    for i in signals.keys():
#        print 'MIX input_signal_array_all[i].shape: ',input_signal_array_all[i].shape
#        #print'Before shuffle: ',input_signal_array_all[i][0,0]
#        np.random.shuffle(input_signal_array_all[i])
#        #print'After shuffle: ',input_signal_array_all[i][0,0]
#        np.save(outputfolder+'/'+signal_identifiers[i]+'_set_raw.npy', input_signal_array_all[i][:,0:-2])
#        np.save(outputfolder+'/'+signal_identifiers[i]+'_set_raw_eventweight.npy', input_signal_array_all[i][:,-2])
#        np.save(outputfolder+'/'+signal_identifiers[i]+'_set_raw_sample_weights.npy', input_signal_array_all[i][:,-1])
#    np.save(outputfolder+'/input_'+fraction+'_signal_dict_array_all.npy', input_signal_array_all) #signal stored as dictionary
#    del input_signal_array_all

    print 'input_array_all.shape: ',input_array_all.shape
    np.random.shuffle(input_array_all)
    np.save(outputfolder+'/input_'+fraction+'_bkg_array_all.npy', input_array_all) #background stored as array
    del input_array_all


def SplitInputs(parameters, outputfolder, filepostfix):
    print("****** SplitInputs ******")
    classtag = get_classes_tag(parameters)
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    tag = dict_to_str(parameters)
#    if not os.path.isdir(outputfolder):
#        os.makedirs(outputfolder)

#    inputfolder = parameters['inputdir']+'NOMINAL'+parameters['inputsubdir']+parameters['prepreprocess']+'/'+ classtag
    inputfolder = parameters['inputdir']+parameters['inputsubdir']+'NOMINAL/'+parameters['prepreprocess']+'/'+ classtag
    with open(inputfolder+ '/variable_names.pkl', 'r') as f:
        variable_names = pickle.load(f)
    copyfile(inputfolder+'/variable_names.pkl', outputfolder+'/variable_names.pkl') #variables stay the same
    input_train_shape1 = len(variable_names)

    input_array_all = np.load(outputfolder+'/input_'+fraction+'_bkg_array_all.npy')
    #train1_set, train2_set, test_set, val_set = np.array_split(input_array_all,4,axis=0) #FixME: split now 0.5,0.25,0.25 and not 0.66/0.16/0.16 as before
    train1_set, train2_set, train3_set, test_set, val_set = np.array_split(input_array_all,5,axis=0) #Split 0.6,0.2,0.2

    train_set = np.concatenate((train1_set, train2_set, train2_set), axis=0)


    f_train = gzip.GzipFile(outputfolder+'/input_'+fraction+'_train_set_raw.npy.gz', "w")
    np.save(file=f_train, arr=train_set[:,0:input_train_shape1])
    #np.save(file=f, arr=train_set)
    f_train.close()
    f_test = gzip.GzipFile(outputfolder+'/input_'+fraction+'_test_set_raw.npy.gz', "w")
    np.save(file=f_test, arr=test_set[:,0:input_train_shape1])
    #np.save(file=f, arr=test_set)
    f_test.close()
    f_val = gzip.GzipFile(outputfolder+'/input_'+fraction+'_val_set_raw.npy.gz', "w")
    np.save(file=f_val, arr=val_set[:,0:input_train_shape1])
#    np.save(file=f, arr=val_set)
    f_val.close()

    # np.save(outputfolder+'/input_'+fraction+'_train_set_raw.npy', train_set[:,0:input_train_shape1])
    # np.save(outputfolder+'/input_'+fraction+'_test_set_raw.npy', test_set[:,0:input_train_shape1])
    # np.save(outputfolder+'/input_'+fraction+'_val_set_raw.npy', val_set[:,0:input_train_shape1])
    print('Inputs stored!')

    labels_train_shape_1 = len(parameters['classes'])
    np.save(outputfolder+'/labels_'+fraction+'_train.npy', train_set[:,input_train_shape1:input_train_shape1+labels_train_shape_1])
    np.save(outputfolder+'/labels_'+fraction+'_test.npy', test_set[:,input_train_shape1:input_train_shape1+labels_train_shape_1])
    np.save(outputfolder+'/labels_'+fraction+'_val.npy', val_set[:,input_train_shape1:input_train_shape1+labels_train_shape_1])
    print('Labels stored!')
    np.save(outputfolder+'/eventweights_'+fraction+'_train.npy', train_set[:,-2])
    np.save(outputfolder+'/eventweights_'+fraction+'_val.npy', val_set[:,-2])
    np.save(outputfolder+'/eventweights_'+fraction+'_test.npy', test_set[:,-2])
    print('Weights test stored!')
    np.save(outputfolder+'/sample_weights_'+fraction+'_train.npy', train_set[:,-1])
    np.save(outputfolder+'/sample_weights_'+fraction+'_val.npy', val_set[:,-1])
    np.save(outputfolder+'/sample_weights_'+fraction+'_test.npy', test_set[:,-1])
    print('Sample Weights test stored!')


    print("Inputs are split into train/test/val datasets")

def FitPrepocessing(parameters, outputfolder, filepostfix):
    print("****** FitPrepocessing ******")
    classtag = get_classes_tag(parameters)
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    tag = dict_to_str(parameters)
    with open(outputfolder+ '/variable_names.pkl', 'r') as f:
        variable_names = pickle.load(f)
    input_train_shape1 = len(variable_names)
    f_train = gzip.GzipFile(outputfolder+'/input_'+fraction+'_train_set_raw.npy.gz', "r")
    train_set = np.load(f_train)

    #input_train_set = train_set[:,0:input_train_shape1]
    input_train_set = train_set
    print ("input_train_set.shape",input_train_set.shape)
#    subset_part = int(0.5*input_train_set.shape[0])
#    input_train_subset = input_train_set[0:subset_part,:]
    input_train_subset = input_train_set
    if(parameters['preprocess'] == 'StandardScaler'):
        print " === StandardScaler preprocessing ==="
        scaler = preprocessing.StandardScaler().fit(input_train_subset)
    elif(parameters['preprocess'] == 'QuantileTransformerUniform'):
        print " === QuantileTransformer(Uniform) preprocessing ==="
        scaler = preprocessing.QuantileTransformer(output_distribution='uniform').fit(input_train_subset)
    elif(parameters['preprocess'] == 'MinMaxScaler'):
        print " === MinMaxScaler preprocessing ==="
        scaler = preprocessing.MinMaxScaler().fit(input_train_subset)
    else:
        print("preprocess set to unknown value! going to use standard StandardScaler preprocessing")
        scaler = preprocessing.StandardScaler().fit(input_train_subset)

    # Write out scaler info
    with open(outputfolder+'/NormInfo.txt', 'w') as f:
        for i in range(np.mean(input_train_set, axis=0).shape[0]): #valid only for StandardScaler, placeholder for the rest
            var = variable_names[i]
            if(parameters['preprocess'] == 'StandardScaler'):
                mean = scaler.mean_[i] #valid only for StandardScaler, placeholder for the rest
                scale = scaler.std_[i] #valid only for StandardScaler, placeholder for the rest
                line = var + ' StandardScaler ' + str(mean) + ' ' + str(scale) + '\n'
            elif(parameters['preprocess'] == 'MinMaxScaler'):
                # minv = scaler.min_[i] #valid only for MinMaxScaler
                # scale = scaler.scale_[i] #valid only for MinMaxScaler
                # line = var + ' MinMaxScaler ' + str(minv) + ' ' + str(scale) + '\n'
                minv = scaler.data_min_[i] #valid only for MinMaxScaler
                maxv = scaler.data_max_[i] #valid only for MinMaxScaler
                line = var + ' MinMaxScaler ' + str(minv) + ' ' + str(maxv) + '\n'

            else:
                mean = scaler.mean_[i] #valid only for StandardScaler, placeholder for the rest
                scale = scaler.std_[i] #valid only for StandardScaler, placeholder for the rest
                line = var + ' StandardScaler ' + str(mean) + ' ' + str(scale) + '\n'
            f.write(line)
    if(parameters['preprocess'] == 'StandardScaler'):
        print("Implement storage of parameters in array!")
    elif(parameters['preprocess'] == 'MinMaxScaler'):
        scaler_data_ = np.array([scaler.data_min_, scaler.data_max_])
        np.save(outputfolder+'/NormInfo.npy', scaler_data_)

    print("Preprocessing parameters are prepared and stored in ", outputfolder)

def ApplyPrepocessing(parameters, outputfolder, filepostfix, setid):
    print("****** ApplyPrepocessing ******")
    classtag = get_classes_tag(parameters)
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    tag = dict_to_str(parameters)
    with open(outputfolder+ '/variable_names.pkl', 'r') as f:
        variable_names = pickle.load(f)
    input_train_shape1 = len(variable_names)

    f_train = gzip.GzipFile(outputfolder+'/input_'+fraction+'_'+setid+'_set_raw.npy.gz', "r")
    train_set = np.load(f_train)

    #train_set = np.load(outputfolder+'/input_'+fraction+'_'+setid+'_set_raw.npy.gz')

    #input_train_set = train_set[:,0:input_train_shape1]
    input_train_set = train_set

    if(parameters['preprocess'] == 'StandardScaler'):
        print("Implement storage of parameters in array!")
    elif(parameters['preprocess'] == 'MinMaxScaler'):
        scaler_data_ = np.load(outputfolder+'/NormInfo.npy')
        Xmin, Xmax = scaler_data_[0], scaler_data_[1]
        input_train_set = (input_train_set - Xmin) / (Xmax-Xmin)
        #train_set[:,0:input_train_shape1] = input_train_set.copy()

    print "STORE mixed inputs for training/test/validation with e.g train_set.shape = ", train_set.shape
    print "-10 last var = ",train_set[:,input_train_shape1-10]
    print 'store in path: ',outputfolder
    np.save(outputfolder+'/input_'+fraction+'_'+setid+'.npy'  , input_train_set)

def ApplySignalPrepocessing(parameters, outputfolder, filepostfix):
    print("****** ApplySignalPrepocessing ******")
    classtag = get_classes_tag(parameters)
    runonfraction = parameters['runonfraction']
    fraction = get_fraction(parameters)
    tag = dict_to_str(parameters)
    with open(outputfolder+ '/variable_names.pkl', 'r') as f:
        variable_names = pickle.load(f)
    input_train_shape1 = len(variable_names)

#    input_signal_array_all = np.load(outputfolder+'/input_'+fraction+'_signal_dict_array_all.npy',allow_pickle=True)
#    print("input_signal_array_all.shape=", input_signal_array_all.shape)
    #for i in input_signal_array_all.keys():

#    for i in range(len(signal_identifiers)):
#
##        input_signal_array = input_signal_array_all[()][i]
##        print("Preprocessing for input_signal_array.shape",input_signal_array.shape)
#        #print("input_signal_array_all[i].shape",input_signal_array_all[i].shape)
#        if(parameters['preprocess'] == 'StandardScaler'):
#            print("Implement storage of parameters in array!")
#        elif(parameters['preprocess'] == 'MinMaxScaler'):
#            scaler_data_ = np.load(outputfolder+'/NormInfo.npy')
#            Xmin, Xmax = scaler_data_[0], scaler_data_[1]
##            input_signal_array[:,0:input_train_shape1] = (input_signal_array[:,0:input_train_shape1] - Xmin) / (Xmax-Xmin)
#            #input_signal_array_all[()][i] = deepcopy(input_signal_array)
#        print("Store modified input for ",signal_identifiers[i])
#        np.save(outputfolder+'/'+signal_identifiers[i]+'.npy', input_signal_array[:,0:input_train_shape1])
#        np.save(outputfolder+'/'+signal_identifiers[i]+'_eventweight.npy', input_signal_array[:,-1])

#        np.save(outputfolder+'/'+signal_identifiers[i]+'.npy', input_signal_array_all[:,0:-1][i])
 #       np.save(outputfolder+'/'+signal_identifiers[i]+'_eventweight.npy', input_signal_array_all[:,-1][i])
