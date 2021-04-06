import numpy as np
from numpy import inf
import itertools
import keras
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils import check_consistent_length, assert_all_finite, column_or_1d, check_array
import scipy.optimize as opt
from scipy.optimize import fsolve
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import train_test_split
from IPython.display import FileLink, FileLinks
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical, plot_model
from keras.callbacks import History, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from copy import deepcopy

from ROOT import TCanvas, TFile, TH1F, TH2F, gROOT, kRed, kBlue, kGreen, kMagenta, kCyan, kOrange, gStyle
from ROOT import gErrorIgnoreLevel
from ROOT import kInfo, kWarning, kError

from constants import *

import math
import pickle
import sys
import os

# ignore weird Warnings: "These warnings are visible whenever you import scipy
# (or another package) that was compiled against an older numpy than is installed."
# https://stackoverflow.com/questions/40845304/runtimewarning-numpy-dtype-size-changed-may-indicate-binary-incompatibility
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def dict_to_str(parameters):
    layers_str = [str(parameters['layers'][i]) for i in range(len(parameters['layers']))]
    tag = 'layers_'
    for i in range(len(layers_str)):
        tag = tag + layers_str[i]
        if i < len(layers_str)-1: tag = tag + '_'
    tag = tag + '__batchsize_' + str(parameters['batchsize'])
    tag = tag + '__classes_' + str(len(parameters['classes'])) + '_'
    for i in range(len(parameters['classes'])):
        for j in range(len(parameters['classes'][i])):
            tag = tag + parameters['classes'][i][j]
            if j < len(parameters['classes'][i]) - 1:
                tag = tag + '+'
        if i < len(parameters['classes']) - 1:
            tag = tag + '_'

    tag = tag + '__regmethod_' + parameters['regmethod']
    tag = tag + '__regrate_' + '{num:06d}'.format(num=int(parameters['regrate']*100000.))
    tag = tag + '__batchnorm_' + str(parameters['batchnorm'])
    tag = tag + '__epochs_' + str(parameters['epochs'])
    tag = tag + '__learningrate_' + '{num:06d}'.format(num=int(parameters['learningrate']*100000.))
    tag = tag + '__runonfraction_' + '{num:03d}'.format(num=int(parameters['runonfraction']*100.))
    tag = tag + '__eqweight_' + str(parameters['eqweight'])
    tag = tag + '__preprocess_' + str(parameters['preprocess'])
    tag = tag + '__priorSigma_' + '{num:03d}'.format(num=int(parameters['sigma']*100.))
    #print("------ Sigma in TAG: ",parameters['sigma'])
    #if len(tag.split('__')) != len(parameters): raise ValueError('in dict_to_str: Number of parameters given in the dictionary does no longer match the prescription how to build the tag out of it.')
    return tag

def get_classes_tag(parameters):
    tag = 'classes_' + str(len(parameters['classes'])) + '_'
    for i in range(len(parameters['classes'])):
        for j in range(len(parameters['classes'][i])):
            tag = tag + parameters['classes'][i][j]
            if j < len(parameters['classes'][i]) - 1:
                tag = tag + '+'
        if i < len(parameters['classes']) - 1:
            tag = tag + '_'
    return tag

def get_classtitles(parameters):
    classes = parameters['classes']
    classtitles = {}
    for key in classes.keys():
        list = classes[key]
        title = ''
        for i in range(len(list)):
            title = title + list[i]
            if i < len(list)-1:
                title = title + '+'
        classtitles[key] = title
    return classtitles

def get_fraction(parameters):
    runonfraction = parameters['runonfraction']
    string = str('{num:03d}'.format(num=int(parameters['runonfraction']*100.)))
    return string




def load_data(parameters, inputfolder, filepostfix):

    print 'Loading data...'
    tag = dict_to_str(parameters)
    classtag = get_classes_tag(parameters)
    fraction = get_fraction(parameters)


    input_train = np.load(inputfolder+'/input_'+fraction+'_train'+filepostfix+'.npy').astype(np.float32)
    input_test = np.load(inputfolder+'/input_'+fraction+'_test'+filepostfix+'.npy').astype(np.float32)
    input_val = np.load(inputfolder+'/input_'+fraction+'_val'+filepostfix+'.npy').astype(np.float32)
    labels_train = np.load(inputfolder+'/labels_'+fraction+'_train'+filepostfix+'.npy')
    labels_test = np.load(inputfolder+'/labels_'+fraction+'_test'+filepostfix+'.npy')
    labels_val = np.load(inputfolder+'/labels_'+fraction+'_val'+filepostfix+'.npy')
    sample_weights_train = np.load(inputfolder+'/sample_weights_'+fraction+'_train'+filepostfix+'.npy').astype(np.float32)
    eventweights_train = np.load(inputfolder+'/eventweights_'+fraction+'_train'+filepostfix+'.npy').astype(np.float32)
    sample_weights_test = np.load(inputfolder+'/sample_weights_'+fraction+'_test'+filepostfix+'.npy').astype(np.float32)
    eventweights_test = np.load(inputfolder+'/eventweights_'+fraction+'_test'+filepostfix+'.npy').astype(np.float32)
    sample_weights_val = np.load(inputfolder+'/sample_weights_'+fraction+'_val'+filepostfix+'.npy').astype(np.float32)
    eventweights_val = np.load(inputfolder+'/eventweights_'+fraction+'_val'+filepostfix+'.npy').astype(np.float32)

    signal_identifiers = ['RSGluon_All', 'RSGluon_M1000', 'RSGluon_M2000', 'RSGluon_M3000', 'RSGluon_M4000', 'RSGluon_M5000', 'RSGluon_M6000']
    signals = {}
    signal_eventweights = {}
    signal_normweights = {}
#    for i in range(len(signal_identifiers)):
#        signals[i] = np.load(inputfolder+'/' + signal_identifiers[i] + filepostfix+'.npy').astype(np.float32)
#        signal_eventweights[i] = np.load(inputfolder+'/' + signal_identifiers[i] + '_eventweight'+filepostfix+'.npy').astype(np.float32)
#        sum_signal_eventweights = signal_eventweights[i].sum()
#        signal_normweights[i] = np.array([1./sum_signal_eventweights for j in range(signal_eventweights[i].shape[0])])
    return input_train, input_test, input_val, labels_train, labels_test, labels_val, sample_weights_train, sample_weights_test, sample_weights_val, eventweights_train, eventweights_test, eventweights_val, signals, signal_eventweights, signal_normweights



def load_predictions(outputfolder, filepostfix):

    print 'Loading predictions...'
#    signal_identifiers = ['RSGluon_All', 'RSGluon_M1000', 'RSGluon_M2000', 'RSGluon_M3000', 'RSGluon_M4000', 'RSGluon_M5000', 'RSGluon_M6000']

    # Load model prediction
    pred_signals = {}
    pred_train = np.load(outputfolder+'/prediction_train'+filepostfix+'.npy').astype(np.float32)
    pred_val = np.load(outputfolder+'/prediction_val'+filepostfix+'.npy').astype(np.float32)
    pred_test = np.load(outputfolder+'/prediction_test'+filepostfix+'.npy').astype(np.float32)
#    for i in range(len(signal_identifiers)):
#        pred_signals[i] = np.load(outputfolder+'/prediction_'+signal_identifiers[i]+''+filepostfix+'.npy').astype(np.float32)

    return pred_train, pred_test, pred_val, pred_signals




def binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    # Check to make sure y_true is valid
    y_type = type_of_target(y_true)
    if not (y_type == "binary" or
            (y_type == "multiclass" and pos_label is not None)):
        raise ValueError("{0} format is not supported".format(y_type))

    check_consistent_length(y_true, y_score, sample_weight)
    y_true = column_or_1d(y_true)
    y_score = column_or_1d(y_score)
    assert_all_finite(y_true)
    assert_all_finite(y_score)

    if sample_weight is not None:
        sample_weight = column_or_1d(sample_weight)

    # ensure binary classification if pos_label is not specified
    classes = np.unique(y_true)
    if (pos_label is None and
        not (np.array_equal(classes, [0, 1]) or
             np.array_equal(classes, [-1, 1]) or
             np.array_equal(classes, [0]) or
             np.array_equal(classes, [-1]) or
             np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    if sample_weight is not None:
        weight = sample_weight[desc_score_indices]
    else:
        weight = 1.

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    if sample_weight is not None:
        # express fps as a cumsum to ensure fps is increasing even in
        # the presence of floating point errors
        fps = stable_cumsum((1 - y_true) * weight)[threshold_idxs]
    else:
        fps = 1 + threshold_idxs - tps
    return fps, tps, y_score[threshold_idxs]


def roc_curve_own(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True):
    # Copied from https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/metrics/ranking.py#L535
    # Extended by purity-part

    fps, tps, thresholds = binary_clf_curve(y_true, y_score, pos_label=pos_label, sample_weight=sample_weight)
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if tps.size == 0 or fps[0] != 0 or tps[0] != 0:
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]

    # purity!
   # prt = 0
    #if((tps+fps).all()>0):
#    prt = tps/(tps+fps)
    prt = np.ones(tps.shape)
    np.divide(tps,tps+fps, out=prt, where=(tps+fps)!=0)
    return fpr, tpr, thresholds, prt


def get_fpr_tpr_thr_auc(parameters, pred_val, labels_val, weights_val):

    eqweight = parameters['eqweight']
    FalsePositiveRates = {}
    TruePositiveRates = {}
    Thresholds = {}
    SignalPuritys = {}
    aucs = {}

    for i in range(labels_val.shape[1]):
        FalsePositiveRates[i], TruePositiveRates[i], Thresholds[i], SignalPuritys[i] = roc_curve_own(labels_val[:,i], pred_val[:,i], sample_weight=weights_val)
        aucs[i] = np.trapz(TruePositiveRates[i], FalsePositiveRates[i])

    return (FalsePositiveRates, TruePositiveRates, Thresholds, aucs, SignalPuritys)

def get_cut_efficiencies(parameters, predictions, thresholds, weights):
    effs_list = []
    indices = []
    length = thresholds.shape[0]
    stepsize = length/10000

    for i in range(thresholds.shape[0]):
        if i%(stepsize)==0:
            effs_list.append(weights[predictions > thresholds[i]].sum())
            indices.append(i)
    sumweights = weights.sum()
    effs = np.array(effs_list)
    effs /= sumweights
    return np.array(effs), indices



def get_data_dictionaries(parameters, eventweights_train, sample_weights_train, pred_train, labels_train, eventweights_val, sample_weights_val, pred_val, labels_val, eventweights_test, sample_weights_test, pred_test, labels_test):
    classes = parameters['classes']
    eqweight = parameters['eqweight']
    pred_trains = {}
    pred_vals = {}
    pred_tests = {}
    weights_trains = {}
    weights_vals = {}
    weights_tests = {}
    normweights_trains = {}
    normweights_vals = {}
    normweights_tests = {}
    lumiweights_trains = {}
    lumiweights_vals = {}
    lumiweights_tests = {}

    for cl in classes.keys():
        pred_trains_thistrueclass = {}
        pred_vals_thistrueclass = {}
        pred_tests_thistrueclass = {}
        weights_trains_thistrueclass = {}
        weights_vals_thistrueclass = {}
        weights_tests_thistrueclass = {}
        normweights_trains_thistrueclass = {}
        normweights_vals_thistrueclass = {}
        normweights_tests_thistrueclass = {}
        lumiweights_trains_thistrueclass = {}
        lumiweights_vals_thistrueclass = {}
        lumiweights_tests_thistrueclass = {}
        for node in classes.keys():
            print "    ---!!!HEYHEY!!! node: ",node
            if not eqweight:
                weights_trains_thistrueclass[node] = eventweights_train[labels_train[:,cl] == 1]
                weights_vals_thistrueclass[node] = eventweights_val[labels_val[:,cl] == 1]
                weights_tests_thistrueclass[node] = eventweights_test[labels_test[:,cl] == 1]
            else:
                weights_trains_thistrueclass[node] = sample_weights_train[labels_train[:,cl] == 1]
                weights_vals_thistrueclass[node]   = sample_weights_val[labels_val[:,cl] == 1]
                weights_tests_thistrueclass[node]   = sample_weights_test[labels_test[:,cl] == 1]
            if(node=='TTbar'):
                print "labels_train[:,cl], pred_train[:,node][labels_train[:,cl] == 1]", labels_train[:,cl], pred_train[:,node][labels_train[:,cl] == 1]

            pred_trains_thistrueclass[node] = pred_train[:,node][labels_train[:,cl] == 1]
            pred_vals_thistrueclass[node] = pred_val[:,node][labels_val[:,cl] == 1]
            pred_tests_thistrueclass[node] = pred_test[:,node][labels_test[:,cl] == 1]
            lumiweights_trains_thistrueclass[node] = eventweights_train[labels_train[:,cl] == 1]
            lumiweights_vals_thistrueclass[node] = eventweights_val[labels_val[:,cl] == 1]
            lumiweights_tests_thistrueclass[node] = eventweights_test[labels_test[:,cl] == 1]
            sum_train = weights_trains_thistrueclass[node].sum()
            sum_val   = weights_vals_thistrueclass[node].sum()
            sum_test   = weights_tests_thistrueclass[node].sum()
            normweights_trains_thistrueclass[node] = np.array([1./sum_train for j in range(weights_trains_thistrueclass[node].shape[0])])
            normweights_vals_thistrueclass[node]   = np.array([1./sum_val for j in range(weights_vals_thistrueclass[node].shape[0])])
            normweights_tests_thistrueclass[node]   = np.array([1./sum_test for j in range(weights_tests_thistrueclass[node].shape[0])])

        pred_trains[cl] = pred_trains_thistrueclass
        pred_vals[cl] = pred_vals_thistrueclass
        pred_tests[cl] = pred_tests_thistrueclass
        weights_trains[cl] = weights_trains_thistrueclass
        weights_vals[cl] = weights_vals_thistrueclass
        weights_tests[cl] = weights_tests_thistrueclass
        normweights_trains[cl] = normweights_trains_thistrueclass
        normweights_vals[cl] = normweights_vals_thistrueclass
        normweights_tests[cl] = normweights_tests_thistrueclass
        lumiweights_trains[cl] = lumiweights_trains_thistrueclass
        lumiweights_vals[cl] = lumiweights_vals_thistrueclass
        lumiweights_tests[cl] = lumiweights_tests_thistrueclass

    return pred_trains, weights_trains, normweights_trains, lumiweights_trains, pred_vals, weights_vals, normweights_vals, lumiweights_vals, pred_tests, weights_tests, normweights_tests, lumiweights_tests


#create dictinaries for data (train, test or val) sample
def get_data_dictionaries_onesample(parameters, eventweights, sample_weights, pred, labels):
    classes = parameters['classes']
    eqweight = parameters['eqweight']
    pred_d = {}
    weights_d = {}
    normweights_d = {}
    lumiweights_d = {}


    for cl in classes.keys():
        print "cl = ",cl
        pred_thistrueclass = {}
        weights_thistrueclass = {}
        normweights_thistrueclass = {}
        lumiweights_thistrueclass = {}
        for node in classes.keys():
            #print "node = ",node
            if(node == 3):
                print "labels[:,cl] = ",labels[:,cl]
                print "pred[:,node][labels[:,cl] == 1] = ",pred[:,node][labels[:,cl] == 1]
            if not eqweight:
                weights_thistrueclass[node] = eventweights[labels[:,cl] == 1]
            else:
                weights_thistrueclass[node] = sample_weights[labels[:,cl] == 1]

            pred_thistrueclass[node] = pred[:,node][labels[:,cl] == 1]
            lumiweights_thistrueclass[node] = eventweights[labels[:,cl] == 1]
            sum_train = weights_thistrueclass[node].sum()
            normweights_thistrueclass[node] = np.array([1./sum_train for j in range(weights_thistrueclass[node].shape[0])])

        pred_d[cl] = pred_thistrueclass
        weights_d[cl] = weights_thistrueclass
        normweights_d[cl] = normweights_thistrueclass
        lumiweights_d[cl] = lumiweights_thistrueclass

    print("pred_d[0]",pred_d[0])
    return pred_d, weights_d, normweights_d, lumiweights_d


def get_indices_wrong_predictions(labels, preds):
    mask = []
    for i in range(labels.shape[0]):
        label = -1
        predclass = -1
        maxpred = -1
        for j in range(labels.shape[1]):
            if labels[i,j] == 1: label = j
            if maxpred < preds[i,j]:
                maxpred = preds[i,j]
                predclass = j
        if label != predclass:
            mask.append(i)

    return mask


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()


def conf_matrix(labels, predictions, weights):
    # will return a list of lists, shape=NxN
    # each list is for one true class
    if labels.shape != predictions.shape:
        raise ValueError('Labels and predictions do not have the same shape.')
    if labels.shape[0] != weights.shape[0]:
        raise ValueError('Labels and weights do not have the same length (.shape[0]).')

    # initialize confusion matrix
    matrix = np.zeros((labels.shape[1], labels.shape[1]))

    # format inputs
    for i in range(labels.shape[0]):
        label = -1
        predclass = -1
        maxpred = -1
        for j in range(labels.shape[1]):
            if labels[i,j] == 1: label = j
            if maxpred < predictions[i,j]:
                maxpred = predictions[i,j]
                predclass = j
        if label == -1: raise ValueError('For this event, the labels of all classes are 0, so the event doesn\'t have a class?')
        matrix[label,predclass] = matrix[label,predclass] + weights[i]

    return matrix



def plot_confusion_matrix(cm, classes,
                          normalize=False, axis='',
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize and axis == 'x':
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix along x")
    elif normalize and axis == 'y':
        cm = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
        print("Normalized confusion matrix along y")
    else:
        print('Confusion matrix, without normalization')

    if not axis == '' and not normalize: raise ValueError('a normalization axis was given but normalization was switched off!')
    if axis == '' and normalize: raise ValueError('no normalization axis was given but normalization was switched on!')

    plt.imshow(cm, interpolation='none', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    listofclasses = []
    for key in classes.keys():
        listofclasses.append(classes[key])
    plt.xticks(tick_marks, listofclasses, rotation=45)
    plt.yticks(tick_marks, listofclasses)

    # fmt = '.2f' if normalize else 'd'
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def log_model_performance(parameters, model_history, outputfolder):
    loss_train = model_history['loss']
    loss_val = model_history['val_loss']
#    print "loss_train[500] = ",loss_train[500]
#    print "loss_val[500] = ",loss_val[500]

    #print(model_history)
    #Fit Losses
    x, fitx, fitfunc, pars_train = fit_loss(model_history['loss'])
    x, fitx, fitfunc, pars_val = fit_loss(model_history['val_loss'])

    #build difference
    pars = [pars_train, pars_val]
    # def func_diff(fitx, mypars):
    #     return fitfunc(fitx, *mypars[0]) - fitfunc(fitx, *mypars[1])
    def func_diff(x, fitfunc, pars):
        return fitfunc(x, *(pars[0])) - fitfunc(x, *(pars[1]))

    #Find intersection between losses - aka roots of (f_train - f_val)
    # sol = opt.root_scalar(func_diff, (pars))
    roots = fsolve(func_diff, [100.], (fitfunc, pars))
    #print roots

    # Get closest integer numbers
    for root in roots:
        x_low = int(math.floor(root))
        x_high = int(math.ceil(root))

    # compare losses at these values, pick the one where difference of losses is smallest
        diff_xlow = math.fabs(fitfunc(x_low, *pars_train) - fitfunc(x_low, *pars_val))
        diff_xhigh = math.fabs(fitfunc(x_high, *pars_train) - fitfunc(x_high, *pars_val))
        bestloss_val = fitfunc(x_low, *pars_val)
        bestx = x_low
        if diff_xhigh < diff_xlow:
            bestloss_val = fitfunc(x_high, *pars_val)
            bestx = x_high
        if root < 10:
            bestloss_val = 999999
            bestx = 999999
        print "Validation loss in point of closest approach: %f, reached after %i epochs" % (bestloss_val, bestx)


    acc_train = model_history['categorical_accuracy']
    acc_val = model_history['val_categorical_accuracy']
    tag = dict_to_str(parameters)
    with open(outputfolder+'/ModelPerformance.txt', 'w') as f:
        f.write('\n\n====================\n')
        f.write('Tag: %s\n\n' % (tag))
        #f.write('Minimum validation loss reached after %i epochs\n' % (loss_val.index(min(loss_val))))
        f.write('Validation loss in point of closest approach: %2.3f, reached after %i epochs\n' % (bestloss_val, bestx))
        f.write('Performance: training loss (min, final) -- validation loss (min, final) -- training acc (min, final) -- validation acc (min, final)\n')
        f.write('                         ({0:2.3f}, {1:2.3f}) --               ({2:2.3f}, {3:2.3f}) --            ({4:1.3f}, {5:1.3f}) --              ({6:1.3f}, {7:1.3f})\n'.format(min(loss_train), loss_train[len(loss_train)-1], min(loss_val), loss_val[len(loss_val)-1], min(acc_train), acc_train[len(acc_train)-1], min(acc_val), acc_val[len(acc_val)-1]))


def plot_rocs(parameters, plotfolder, pred_val, labels_val, sample_weights_val, eventweights_val, pred_signals=None, eventweight_signals=None, usesignals=[0], use_best_model=False):

    tag = dict_to_str(parameters)
    classtitles = get_classtitles(parameters)
    classes = parameters['classes']
#    do_sig = (pred_signals is not None) and (eventweight_signals is not None)
    do_sig = (pred_signals is None) and (eventweight_signals is None)

    print 'plotting ROC curves'
    # Inclusive backgrounds
    # equallyweighted
#    print("HEY, in <plot_rocs> pred_val: ",pred_val[0:2,:])
 #   print("HEY, in <plot_rocs> labels_val: ",labels_val[0:2,:])
    FalsePositiveRates, TruePositiveRates, Thresholds, aucs, SignalPuritys = get_fpr_tpr_thr_auc(parameters=parameters, pred_val=pred_val, labels_val=labels_val, weights_val=sample_weights_val)
    # print 'eqweight: ', sample_weights_val[10:15]
    print 'fpr: ', FalsePositiveRates[1][10:15]
    print 'tpr: ', TruePositiveRates[1][10:15]


    plt.clf()
    fig = plt.figure()
    plt.xticks(np.arange(0.1,1.1,0.1))
    plt.grid(True, which='both')
    for i in range(len(FalsePositiveRates)):
        plt.semilogy(TruePositiveRates[i], FalsePositiveRates[i], label=classtitles[i] + ', AUC: '+str(round(aucs[i],3)), color=colorstr[i])
    plt.legend(loc='upper left')
    plt.ylim([0.0001, 1.05])
    plt.xlabel('Selection efficiency')
    plt.ylabel('Background efficiency')
    title = 'ROC_val_eqweight'
    if use_best_model: title += '_best'
    title += '.pdf'
    fig.savefig(plotfolder+'/'+title)
    plt.close()

    plt.clf()
    fig = plt.figure()
    for i in range(len(TruePositiveRates)):
        plt.plot(TruePositiveRates[i], SignalPuritys[i], label=classtitles[i], color=colorstr[i])
    plt.legend(loc='best')
    plt.ylim([0., 1.05])
    plt.xlim([0., 1.])
    plt.xticks(np.arange(0.,1.1,0.1))
    plt.yticks(np.arange(0.,1.1,0.1))
    plt.grid(True, which='both')
    plt.xlabel('Selection efficiency')
    plt.ylabel('Purity')
    title = 'EffVsPur_val_eqweight'
    if use_best_model: title += '_best'
    title += '.pdf'
    fig.savefig(plotfolder+'/'+title)
    plt.close()


    # lumiweighted
    FalsePositiveRates, TruePositiveRates, Thresholds, aucs, SignalPuritys = get_fpr_tpr_thr_auc(parameters=parameters, pred_val=pred_val, labels_val=labels_val, weights_val=eventweights_val)
    # print 'lumiweight: ', eventweights_val[10:15]
    # print 'fpr: ', FalsePositiveRates[1][10:15]
    plt.clf()
    fig = plt.figure()
    plt.xticks(np.arange(0.1,1.1,0.1))
    plt.grid(True, which='both')
    for i in range(len(FalsePositiveRates)):
        plt.semilogy(TruePositiveRates[i], FalsePositiveRates[i], label=classtitles[i] + ', AUC: '+str(round(aucs[i],3)), color=colorstr[i])
    plt.legend(loc='upper left')
    plt.ylim([0.0001, 1.05])
    plt.xlabel('Class selection efficiency')
    plt.ylabel('Class background efficiency')
    title = 'ROC_val_lumiweighted'
    if use_best_model: title += '_best'
    title += '.pdf'
    fig.savefig(plotfolder+'/'+title)
    plt.close()

    plt.clf()
    fig = plt.figure()
    for i in range(len(TruePositiveRates)):
        plt.plot(TruePositiveRates[i], SignalPuritys[i], label=classtitles[i], color=colorstr[i])
    plt.legend(loc='best')
    plt.ylim([0., 1.05])
    plt.xlim([0., 1.])
    plt.xticks(np.arange(0.,1.1,0.1))
    plt.yticks(np.arange(0.,1.1,0.1))
    plt.grid(True, which='both')
    plt.xlabel('Selection efficiency')
    plt.ylabel('Purity')
    title = 'EffVsPur_val_lumiweighted'
    if use_best_model: title += '_best'
    title += '.pdf'
    fig.savefig(plotfolder+'/'+title)
    plt.close()
    # now plot signal efficiency versus signal purity --> calculate things ourselves

    # Individual backgrounds: 1 plot per class, the other classes are individual curves. Therefore we only need to look at outputnode no. 'cl'
    for cl in classes.keys():
        #cl is the true class

        # Dictionaries to store the rocs against each individual background
        fprs_eq = {}
        tprs_eq = {}
        thrs_eq = {}
        aucss_eq = {}
        prts_eq = {}
        fprs_lum = {}
        tprs_lum = {}
        thrs_lum = {}
        aucss_lum = {}
        prts_lum = {}
#        eff_signals_lum = {}
#        auc_signals_lum = {}

        # Loop over all remaining classes, always keep predictions, labels, and weights for class 'cl' and this one background
        for i in classes.keys():
            # i is the index of the one background class
            if i == cl: continue
            mask = np.logical_or(labels_val[:,cl] == 1, labels_val[:,i] == 1)
            pred_this = pred_val[mask]
            labels_this = labels_val[mask]
            weights_sample = sample_weights_val[mask]
            weights_lum = eventweights_val[mask]
            pred_this = pred_this[:,[cl,i]]
            labels_this = labels_this[:,[cl,i]]
            fprs_eq[i], tprs_eq[i], thrs_eq[i], aucss_eq[i], prts_eq[i] = get_fpr_tpr_thr_auc(parameters=parameters, pred_val=pred_this, labels_val=labels_this, weights_val=weights_sample)
            fprs_lum[i], tprs_lum[i], thrs_lum[i], aucss_lum[i], prts_lum[i] = get_fpr_tpr_thr_auc(parameters=parameters, pred_val=pred_this, labels_val=labels_this, weights_val=weights_lum)

        # don't care, which tpr and thr we choose for class 'cl', we are calculating the singal efficiency for those values anyway ;)
        if do_sig:
            for key in pred_signals.keys():
                eff_signals_lum[key], indices = get_cut_efficiencies(parameters=parameters, predictions=pred_signals[key][:,cl], thresholds=thrs_lum[0 if cl > 0 else 1][0], weights=eventweight_signals[key])
                # print thrs_lum[0 if cl > 0 else 1][0]
                # print eff_signals_lum
                auc_signals_lum[key] = np.trapz(tprs_lum[0 if cl > 0 else 1][0][indices], eff_signals_lum[key])

        # Now just plot all 4 curves (equallyweighted)
        plt.clf()
        fig = plt.figure()
        plt.xticks(np.arange(0.1,1.1,0.1))
        plt.grid(True, which='both')
        for i in fprs_eq.keys():
            plt.semilogy(tprs_eq[i][0], fprs_eq[i][0], label='Bkg: ' + classtitles[i] + ', AUC: '+str(round(aucss_eq[i][0],3)), color=colorstr[i])
        plt.legend(loc='upper left')
        plt.ylim([0.0001, 1.05])
#        plt.ylim([1e-9, 1.05])
        plt.xlabel(classtitles[cl]+' selection efficiency')
        plt.ylabel('Class background efficiency')
        title = 'ROC_val_class'+str(cl)+'_eqweight'
        if use_best_model: title += '_best'
        title += '.pdf'
        fig.savefig(plotfolder+'/'+title)
        plt.close()

        plt.clf()
        fig = plt.figure()
        for i in fprs_eq.keys():
            plt.plot(tprs_eq[i][0], prts_eq[i][0], label='Bkg: ' + classtitles[i], color=colorstr[i])
        plt.legend(loc='best')
        plt.ylim([0., 1.05])
        plt.xlim([0., 1.])
        plt.xticks(np.arange(0.,1.1,0.1))
        plt.yticks(np.arange(0.,1.1,0.1))
        plt.grid(True, which='both')
        plt.xlabel(classtitles[cl]+' selection efficiency')
        plt.ylabel('Purity wrt. given background')
        title = 'EffVsPur_val_class'+str(cl)+'_eqweight'
        if use_best_model: title += '_best'
        title += '.pdf'
        fig.savefig(plotfolder+'/'+title)
        plt.close()

        # Now just plot all 4 curves (lumiweighted)
        plt.clf()
        fig = plt.figure()
        plt.xticks(np.arange(0.1,1.1,0.1))
        plt.grid(True, which='both')
        for i in fprs_lum.keys():
            plt.semilogy(tprs_lum[i][0], fprs_lum[i][0], label='Bkg: '+classtitles[i] + ', AUC: '+str(round(aucss_lum[i][0],3)), color=colorstr[i])
        if do_sig:
            for sigidx in range(len(usesignals)):
                plt.semilogy(tprs_lum[0 if cl > 0 else 1][0][indices], eff_signals_lum[usesignals[sigidx]], label='Signal (%s), AUC: %s' % (signalmasses[usesignals[sigidx]], str(round(auc_signals_lum[usesignals[sigidx]],3))), color='k', linestyle=signal_linestyles[sigidx])
        plt.legend(loc='upper left')
        plt.ylim([0.0001, 1.05])
#        plt.ylim([1e-9, 1.05])
        plt.xlabel(classtitles[cl]+' selection efficiency')
        plt.ylabel('Class background efficiency')
        title = 'ROC_val_class'+str(cl)+'_lumiweighted'
        if use_best_model: title += '_best'
        title += '.pdf'
        fig.savefig(plotfolder+'/'+title)
        plt.close()

        plt.clf()
        fig = plt.figure()
        for i in fprs_eq.keys():
            plt.plot(tprs_lum[i][0], prts_lum[i][0], label='Bkg: ' + classtitles[i], color=colorstr[i])
        plt.legend(loc='best')
        plt.ylim([0., 1.05])
        plt.xlim([0., 1.])
        plt.xticks(np.arange(0.,1.1,0.1))
        plt.yticks(np.arange(0.,1.1,0.1))
        plt.grid(True, which='both')
        plt.xlabel(classtitles[cl]+' selection efficiency')
        plt.ylabel('Purity wrt. given background')
        title = 'EffVsPur_val_class'+str(cl)+'_lumiweighted'
        if use_best_model: title += '_best'
        title += '.pdf'
        fig.savefig(plotfolder+'/'+title)
        plt.close()


def plot_loss(parameters, plotfolder, model_history):
    print 'Starting to plot Loss'
    eqweight = parameters['eqweight']

    # def fitfunc(x, a, b, c, d, e):
    #     return a + b/x + c*x + d*x*x + e/x/x

    tag = dict_to_str(parameters)
    plt.clf()
    fig = plt.figure()
    plt.grid()
    x, fitx, fitfunc, postfitpars_train = fit_loss(model_history['loss'])
    x, fitx, fitfunc, postfitpars_val = fit_loss(model_history['val_loss'])

    plt.plot(x, model_history['loss'], label = 'Training set')
    plt.plot(x, model_history['val_loss'], label = 'Validation set')
    #plt.plot(fitx, fitfunc(fitx, *postfitpars_train), label="Fit (training set)") #FixME for BNN
    #plt.plot(fitx, fitfunc(fitx, *postfitpars_val), label="Fit (validation set)") #FixME for BNN

    plt.legend(loc='upper right')
    #plt.ylim([0.1, 0.25])
    #plt.ylim([0.01,100])
    plt.ylim([0.001,1])
    plt.yscale("log")
    if eqweight:
#        plt.ylim([0.01, 0.06])
        #plt.ylim([0.1,100])
        plt.ylim([0.001,1])
        plt.yscale("log")
    plt.ylabel('Loss')
    plt.xlabel('Number of training epochs')
    fig.savefig(plotfolder+'/Loss.pdf')
    plt.close()

def fit_loss(losslist, maxfev=50000):

    def fitfunc(x, a, b, c, d, e):
        return a + b/x + c*x + d*x*x + e/x/x + d*x*x*x
    print("len(losslist)",len(losslist))
    x = range(len(losslist)+1)
    x = x[1:]
    x = np.array(x)
    #fitx = x[9:]
    #fity = losslist[9:]
    fitx = x[2:]
    fity = losslist[2:]

    postfitpars, cov = opt.curve_fit(fitfunc, fitx, fity, maxfev=maxfev,method='dogbox')
    print "postfitpars: ",postfitpars
#    postfitpars, cov = opt.curve_fit(fitfunc, fitx, fity) #TEST

    return x, fitx, fitfunc, postfitpars






def plot_accuracy(parameters, plotfolder, model_history):
    print 'Starting to plot accuracy'
    tag = dict_to_str(parameters)
    plt.clf()
    fig = plt.figure()
    plt.grid()
    x = range(len(model_history['categorical_accuracy'])+1)
    x = x[1:]
    plt.plot(x, model_history['categorical_accuracy'], label = 'Training set')
    plt.plot(x, model_history['val_categorical_accuracy'], label = 'Validation set')
    plt.legend(loc='lower right')
    plt.ylim([0., 1.05])
    plt.ylabel('Prediction accuracy')
    plt.xlabel('Number of training epochs')
    fig.savefig(plotfolder+'/Accuracy.pdf')


def plot_weight_updates(parameters, model, input_val):
    print 'Starting to plot weights'
    tag = dict_to_str(parameters)
    epochs = parameters['epochs']
    tmp = os.listdir('output/'+tag)
    weightfiles = []
    weights = []
    rel_updates = []
    updates = []
    activations = []
    for t in tmp:
        if 'model_epoch' in t and '.h5' in t:
            weightfiles.append(t)

    idx=0
    for weightfile in weightfiles:
        model = keras.models.load_model('output/'+tag+'/'+weightfile)
        weights_thismodel = []
        updates_thismodel = []
        activations_thismodel = []

        for i in range(len(model.layers)):
            try:
                W = model.layers[i].kernel.get_value(borrow=True)
            except AttributeError:
                continue
            W = model.layers[i].kernel.get_value(borrow=True)
            W = np.squeeze(W)
            W = W.ravel()
            weights_thismodel.append(W)
        weights.append(weights_thismodel)

        if idx > 0:
            updates_thismodel = [weights[idx][i] - weights[idx-1][i] for i in range(len(weights[idx]))]
            updates_thismodel = [updates_thismodel[i] / weights[idx-1][i] for i in range(len(weights[idx-1]))]
        rel_updates.append(updates_thismodel)
        idx += 1
    # print weights_norm
    # print rel_updates

    for i in range(len(weights)):
        allweights = []
        allupdates = []
        for j in range(len(weights[i])):
            # print i, j
            weights[i][j][weights[i][j]==inf] = 0.
            weights[i][j][weights[i][j]==-inf] = 0.
            allweights += weights[i][j].tolist()
            if i > 0:
                rel_updates[i][j][rel_updates[i][j]==inf] = 0.
                rel_updates[i][j][rel_updates[i][j]==-inf] = 0.
                allupdates += rel_updates[i][j].tolist()

        nbins = 50
        current_epoch = epochs/5 * i
        plt.clf()
        fig = plt.figure()
        plt.hist(allweights, bins=nbins, histtype='step', label='Weights after '+str(current_epoch)+' training epochs')
        plt.yscale('log')
        plt.xlabel('Weight')
        plt.ylabel('Number of nodes')
        fig.savefig(plotfolder+'/Weights_epoch'+str(current_epoch)+'.pdf')
        plt.close()

        if i>0:
            plt.clf()
            fig = plt.figure()
            plt.hist(allupdates, bins=nbins, histtype='step', label='Relative updates after '+str(current_epoch)+' training epochs')
            plt.yscale('log')
            plt.xlabel('Relative weight update')
            plt.ylabel('Number of nodes')
            fig.savefig(plotfolder+'/Updates_epoch'+str(current_epoch)+'.pdf')
            plt.close()


def plot_confusion_matrices(parameters, plotfolder, pred_train, labels_train, sample_weights_train, eventweights_train, pred_val, labels_val, sample_weights_val, eventweights_val, use_best_model=False):
    print 'Starting to plot confusion matrix'
    tag = dict_to_str(parameters)
    eqweight = parameters['eqweight']
    classtitles = get_classtitles(parameters)

    labels_1d = np.empty(labels_val.shape[0])
    pred_1d = np.empty(pred_val.shape[0])
    for i in range(len(labels_1d)):
        label = -1
        predclass = -1
        maxpred = -1
        for j in range(labels_val.shape[1]):
            if labels_val[i,j] == 1: label = j
            if maxpred < pred_val[i,j]:
                maxpred = pred_val[i,j]
                predclass = j
        if label == -1: raise ValueError('For this event, the labels of all classes are 0, so the event doesn\'t have a class?')
        labels_1d[i] = label
        pred_1d[i] = predclass
    if eqweight:
        conf_matrix_train = conf_matrix(labels_train, pred_train, sample_weights_train)
        conf_matrix_val = conf_matrix(labels_val, pred_val, sample_weights_val)
    else:
        conf_matrix_train = conf_matrix(labels_train, pred_train, eventweights_train)
        conf_matrix_val = conf_matrix(labels_val, pred_val, eventweights_val)


    plt.clf()
    fig = plt.figure()
    plot_confusion_matrix(conf_matrix_val, classes=classtitles, title='Confusion matrix for validation set, without normalization')
    title = 'Confusion_matrix_val'
    if use_best_model: title += '_best'
    title += '.pdf'
    fig.savefig(plotfolder+'/'+title)
    plt.close()

    plt.clf()
    fig = plt.figure()
    plot_confusion_matrix(conf_matrix_val, classes=classtitles, normalize=True, axis='x', title='Confusion matrix for validation set, rows normalized')
    title = 'Confusion_matrix_val_normx'
    if use_best_model: title += '_best'
    title += '.pdf'
    fig.savefig(plotfolder+'/'+title)
    plt.close()

    plt.clf()
    fig = plt.figure()
    plot_confusion_matrix(conf_matrix_val, classes=classtitles, normalize=True, axis='y', title='Confusion matrix for validation set, columns normalized')
    title = 'Confusion_matrix_val_normy'
    if use_best_model: title += '_best'
    title += '.pdf'
    fig.savefig(plotfolder+'/'+title)
    plt.close()

    plt.clf()
    fig = plt.figure()
    plot_confusion_matrix(conf_matrix_train, classes=classtitles, title='Confusion matrix for training set, without normalization')
    title = 'Confusion_matrix_train'
    if use_best_model: title += '_best'
    title += '.pdf'
    fig.savefig(plotfolder+'/'+title)
    plt.close()

    plt.clf()
    fig = plt.figure()
    plot_confusion_matrix(conf_matrix_train, classes=classtitles, normalize=True, axis='x', title='Confusion matrix for training set, rows normalized')
    title = 'Confusion_matrix_train_normx'
    if use_best_model: title += '_best'
    title += '.pdf'
    fig.savefig(plotfolder+'/'+title)
    plt.close()

    plt.clf()
    fig = plt.figure()
    plot_confusion_matrix(conf_matrix_train, classes=classtitles, normalize=True, axis='y', title='Confusion matrix for training set, columns normalized')
    title = 'Confusion_matrix_train_normy'
    if use_best_model: title += '_best'
    title += '.pdf'
    fig.savefig(plotfolder+'/'+title)
    plt.close()


def plot_outputs_2d(parameters, plotfolder, pred_vals, lumiweights_vals, use_best_model=False):
    print 'Starting to plot 2d plots of output variables'
    tag = dict_to_str(parameters)
    eqweight = parameters['eqweight']
    classes = parameters['classes']
    classtitles = get_classtitles(parameters)
    gErrorIgnoreLevel = kWarning

    gROOT.SetBatch(True);
    for i in classes.keys():
        for j in classes.keys():
            if j <= i: continue
            idx = 0
            hists = []
            for trueclass in classes.keys():
                c1 = TCanvas('c1', 'c1', 600, 600)
                hist = TH2F( classtitles[trueclass], classtitles[trueclass], 100, 0, 1.0001, 100, 0, 1.0001)
                for k in range(pred_vals[trueclass][i].shape[0]):
                    hist.Fill(pred_vals[trueclass][i][k], pred_vals[trueclass][j][k], lumiweights_vals[trueclass][i][k])
                hist.SetMarkerColor(rootcolors[colorstr[trueclass]])
                hist.GetXaxis().SetTitle('Classifier output, node ' + classtitles[i])
                hist.GetYaxis().SetTitle('Classifier output, node ' + classtitles[j])
                hist.Scale(1./hist.Integral())
                hist.GetZaxis().SetRangeUser(1.E-7, 1)
                hist.Draw("COLZ")
                idx += 1
                gStyle.SetOptStat(0)
                c1.SetLogz(True)
                title = 'Outputs_trueclass'+classtitles[trueclass]+'_node'+str(i)+'_node'+str(j)
                if use_best_model: title += '_best'
                title += '.pdf'
                c1.SaveAs(plotfolder+'/'+title)
                del c1


def plot_outputs_1d_nodes(parameters, plotfolder, pred_trains, labels_train, weights_trains, lumiweights_trains, normweights_trains, pred_vals, labels_val, weights_vals, lumiweights_vals, normweights_vals, pred_signals=None, eventweight_signals=None, normweight_signals=None, usesignals=[0], use_best_model=False):

    print 'Starting to plot the classifier output distribution'
    tag = dict_to_str(parameters)
    classtitles = get_classtitles(parameters)
#    do_sig = (pred_signals is not None) and (eventweight_signals is not None) and (normweight_signals is not None)
    do_sig = (pred_signals is None) and (eventweight_signals is None) and (normweight_signals is None)

    for cl in range(labels_train.shape[1]):
        # 'cl' is the output node number
        nbins = 100
        binwidth = 1./float(nbins)
        y_trains = {}
        y_vals = {}
        y_trains_norm = {}
        y_vals_norm = {}
        bin_edges_trains = {}
        bin_edges_vals = {}
        bin_centers = {}
        yerrs = {}
        yerrs_norm = {}
#        y_signals = {}
#        yerrs_signals = {}



        for i in range(labels_train.shape[1]):
            # 'i' is the true class (always the first index)
            y_trains[i], dummy = np.histogram(pred_trains[i][cl], bins=nbins, weights=lumiweights_trains[i][cl])
            y_trains_norm[i], bin_edges_trains[i] = np.histogram(pred_trains[i][cl], bins=nbins, weights=weights_trains[i][cl])
            y_vals[i], dummy = np.histogram(pred_vals[i][cl], bins=nbins, weights=lumiweights_vals[i][cl])
            y_vals_norm[i], bin_edges_vals[i] = np.histogram(pred_vals[i][cl], bins=nbins, weights=weights_vals[i][cl])
            bin_centers[i] = 0.5*(bin_edges_trains[i][1:] + bin_edges_trains[i][:-1])
            #print "y_vals_norm[i]",y_vals_norm[i]
            yerrs_norm[i] = y_vals_norm[i]**0.5
            yerrs[i] = y_vals[i]**0.5
            y_vals_norm[i] = y_vals_norm[i] * normweights_vals[i][cl][0]
            yerrs_norm[i] = yerrs_norm[i] * normweights_vals[i][cl][0]
#            print("yerrs_norm[i].shape",yerrs[i].shape)
#            print "yerrs_norm[i]",yerrs_norm[i]

        if do_sig:
            for key in pred_signals.keys():
                y_signals[key], dummy = np.histogram(pred_signals[key][:,cl], bins=nbins, weights=eventweight_signals[key])
                yerrs_signals[key] = y_signals[key]**0.5

        plt.clf()
        fig = plt.figure()
        classtitle_to_use = ''
        for i in range(labels_train.shape[1]):
            plt.hist(pred_trains[i][cl], weights=weights_trains[i][cl]*normweights_trains[i][cl], bins=nbins, histtype='step', label='Training sample, ' + classtitles[i], color=colorstr[i])
            plt.errorbar(bin_centers[i], y_vals_norm[i], yerr=yerrs_norm[i], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Validation sample, ' + classtitles[i], color=colorstr[i])
            if i == cl:
                classtitle_to_use = classtitles[i]
        if do_sig:
            for sigidx in range(len(usesignals)):
                plt.hist(pred_signals[usesignals[sigidx]][:,cl], weights=eventweight_signals[usesignals[sigidx]]*normweight_signals[usesignals[sigidx]], bins=nbins, histtype='step', label='Signal (%s)' % signalmasses[usesignals[sigidx]], color='k', linestyle=signal_linestyles[sigidx])
        plt.legend(loc='best', prop={'size': 8})
        plt.yscale('log')
        plt.xlim([-0.05, 1.05])
        plt.xlabel('Classifier output for node '+classtitle_to_use)
        plt.ylabel('Normalized number of events / bin')
        title = 'Distribution_node'+str(cl)+'_norm'
        if use_best_model: title += '_best'
        title += '.pdf'
        fig.savefig(plotfolder+'/'+title)

        plt.clf()
        fig = plt.figure()
        classtitle_to_use = ''
        for i in range(labels_train.shape[1]):
            plt.errorbar(bin_centers[i], y_vals[i], yerr=yerrs[i], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Validation sample, ' + classtitles[i], color=colorstr[i])
            # print i, y_vals[i]
            if i == cl:
                classtitle_to_use = classtitles[i]
        if do_sig:
            for sigidx in range(len(usesignals)):
                plt.hist(pred_signals[usesignals[sigidx]][:,cl], weights=eventweight_signals[usesignals[sigidx]], bins=nbins, histtype='step', label='Signal (%s)' % signalmasses[usesignals[sigidx]], color='k', linestyle=signal_linestyles[sigidx])
                # plt.errorbar(bin_centers[i], y_signals[usesignal], yerr=yerrs_signals[usesignal], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Signal sample', color='k')

        plt.legend(loc='best', prop={'size': 8})
        plt.yscale('log')
        plt.xlim([-0.05, 1.05])
        plt.xlabel('Classifier output for node '+classtitle_to_use)
        plt.ylabel('Number of events / bin (weighted by luminosity)')
        title = 'Distribution_node'+str(cl)
        if use_best_model: title += '_best'
        title += '.pdf'
        fig.savefig(plotfolder+'/'+title)
        plt.close()

def plot_outputs_1d_nodes_with_stderror(parameters, plotfolder, pred_trains,pred_trains_std, labels_train, weights_trains, lumiweights_trains, normweights_trains, pred_vals, pred_vals_std, labels_val, weights_vals, lumiweights_vals, normweights_vals, pred_signals=None, pred_signals_std=None, eventweight_signals=None, normweight_signals=None, usesignals=[0], use_best_model=False):

    print '****** Starting to plot the classifier output distribution with std error ******'
    tag = dict_to_str(parameters)
    classtitles = get_classtitles(parameters)
#    do_sig = (pred_signals is not None) and (eventweight_signals is not None) and (normweight_signals is not None)
    do_sig = (pred_signals is None) and (eventweight_signals is None) and (normweight_signals is None)

    for cl in range(labels_train.shape[1]):
        # 'cl' is the output node number
        nbins = 100
        binwidth = 1./float(nbins)
        y_trains = {}
        y_vals = {}
        y_trains_norm = {}
        y_vals_norm = {}
        bin_edges_trains = {}
        bin_edges_vals = {}
        bin_centers = {}
        yerrs = {}
        yerrs_norm = {}
#        y_signals = {}
#        yerrs_signals = {}

        for i in range(labels_train.shape[1]):
            # 'i' is the true class (always the first index)
            # print("pred_trains[i][0].shape",pred_trains[i][cl].shape)
            # print("pred_vals_std[i][0].shape",pred_vals_std[i][cl].shape)
            # print("pred_trains[i][0]:",pred_trains[i][0])
            # print("pred_vals_std[i][0]:",pred_vals_std[i][0])
            # print "lumiweights_trains[i][0]: ",lumiweights_trains[i][0]
            y_trains[i], dummy = np.histogram(pred_trains[i][cl], bins=nbins, weights=lumiweights_trains[i][cl])
            y_trains_norm[i], bin_edges_trains[i] = np.histogram(pred_trains[i][cl], bins=nbins, weights=weights_trains[i][cl])
            y_vals[i], dummy = np.histogram(pred_vals[i][cl], bins=nbins, weights=lumiweights_vals[i][cl])
            y_vals_norm[i], bin_edges_vals[i] = np.histogram(pred_vals[i][cl], bins=nbins, weights=weights_vals[i][cl])

            yerrs[i], dummy = np.histogram(pred_vals_std[i][cl], bins=nbins, weights=lumiweights_vals[i][cl])
            yerrs_norm[i], dummy = np.histogram(pred_vals_std[i][cl], bins=nbins, weights=weights_vals[i][cl])

            bin_centers[i] = 0.5*(bin_edges_trains[i][1:] + bin_edges_trains[i][:-1])
            y_vals_norm[i] = y_vals_norm[i] * normweights_vals[i][cl][0]
            yerrs_norm[i] = yerrs_norm[i] * normweights_vals[i][cl][0]


        if do_sig:
            for key in pred_signals.keys():
                y_signals[key], dummy = np.histogram(pred_signals[key][:,cl], bins=nbins, weights=eventweight_signals[key])
                #yerrs_signals[key] = y_signals[key]**0.5
                yerrs_signals[key], dummy =np.histogram(pred_signals_std[key][:,cl], bins=nbins, weights=eventweight_signals[key])

        plt.clf()
        fig = plt.figure()
        classtitle_to_use = ''
        for i in range(labels_train.shape[1]):
            #print("labels_train:",labels_train[i])
            plt.hist(pred_trains[i][cl], weights=weights_trains[i][cl]*normweights_trains[i][cl], bins=nbins, histtype='step', label='Training sample, ' + classtitles[i], color=colorstr[i])
            plt.errorbar(bin_centers[i], y_vals_norm[i], yerr=yerrs_norm[i], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Validation sample, ' + classtitles[i], color=colorstr[i])
            if i == cl:
                classtitle_to_use = classtitles[i]
        if do_sig:
            for sigidx in range(len(usesignals)):
                plt.hist(pred_signals[usesignals[sigidx]][:,cl], weights=eventweight_signals[usesignals[sigidx]]*normweight_signals[usesignals[sigidx]], bins=nbins, histtype='step', label='Signal (%s)' % signalmasses[usesignals[sigidx]], color='k', linestyle=signal_linestyles[sigidx])


        plt.legend(loc='best', prop={'size': 8})
        plt.yscale('log')
        plt.xlim([-0.05, 1.05])
        plt.xlabel('Classifier output for node '+classtitle_to_use)
        plt.ylabel('Normalized number of events / bin')
        title = 'Distribution_node'+str(cl)+'_norm'
        if use_best_model: title += '_best'
        title += '.pdf'
        fig.savefig(plotfolder+'/'+title)

        plt.clf()
        fig = plt.figure()
        classtitle_to_use = ''
        for i in range(labels_train.shape[1]):
            plt.errorbar(bin_centers[i], y_vals[i], yerr=yerrs[i], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Validation sample, ' + classtitles[i], color=colorstr[i])
            # print i, y_vals[i]
            if i == cl:
                classtitle_to_use = classtitles[i]
        if do_sig:
            for sigidx in range(len(usesignals)):
                plt.hist(pred_signals[usesignals[sigidx]][:,cl], weights=eventweight_signals[usesignals[sigidx]], bins=nbins, histtype='step', label='Signal (%s)' % signalmasses[usesignals[sigidx]], color='k', linestyle=signal_linestyles[sigidx])
                # plt.errorbar(bin_centers[i], y_signals[usesignal], yerr=yerrs_signals[usesignal], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Signal sample', color='k')

        plt.legend(loc='best', prop={'size': 8})
        plt.yscale('log')
        plt.xlim([-0.05, 1.05])
        plt.xlabel('Classifier output for node '+classtitle_to_use)
        plt.ylabel('Number of events / bin (weighted by luminosity)')
        title = 'Distribution_node'+str(cl)
        if use_best_model: title += '_best'
        title += '_with_stderror.pdf'
        fig.savefig(plotfolder+'/'+title)
        print("Store file: ",plotfolder+'/'+title)
        plt.close()

def plot_outputs_1d_nodes_std(parameters, plotfolder, pred_trains,pred_trains_std, labels_train, weights_trains, lumiweights_trains, normweights_trains, pred_vals, pred_vals_std, labels_val, weights_vals, lumiweights_vals, normweights_vals, pred_signals=None, pred_signals_std=None, eventweight_signals=None, normweight_signals=None, usesignals=[0], use_best_model=False):

    print '****** Starting to plot the std of classifier output distribution ******'
    tag = dict_to_str(parameters)
    classtitles = get_classtitles(parameters)
#    do_sig = (pred_signals is not None) and (eventweight_signals is not None) and (normweight_signals is not None)
    do_sig = (pred_signals is None) and (eventweight_signals is None) and (normweight_signals is None)

    for cl in range(labels_train.shape[1]):
        # 'cl' is the output node number
        nbins = 100
        binwidth = 1./float(nbins)
        y_trains = {}
        y_vals = {}
        y_trains_norm = {}
        y_vals_norm = {}
        bin_edges_trains = {}
        bin_edges_vals = {}
        bin_centers = {}
        yerrs = {}
        yerrs_norm = {}
#        y_signals = {}
#        yerrs_signals = {}

        for i in range(labels_train.shape[1]):
            # 'i' is the true class (always the first index)
#            print("pred_trains[i][0].shape",pred_trains[i][cl].shape)
 #           print("pred_vals_std[i][0].shape",pred_vals_std[i][cl].shape)
 #           print("pred_trains[i][0]:",pred_trains[i][0])
 #           print("pred_vals_std[i][0]:",pred_vals_std[i][0])
 #           print "lumiweights_trains[i][0]: ",lumiweights_trains[i][0]
#            print "pred_trains[i][cl].shape = ",pred_trains[i][cl].shape
#            print "pred_trains_std[i][cl].shape = ",pred_trains_std[i][cl].shape
#            print "lumiweights_trains[i][cl].shape = ",lumiweights_trains[i][cl].shape
            y_trains[i], dummy = np.histogram(pred_trains_std[i][cl], bins=nbins, weights=lumiweights_trains[i][cl])
            y_trains_norm[i], bin_edges_trains[i] = np.histogram(pred_trains_std[i][cl], bins=nbins, weights=weights_trains[i][cl])
            y_vals[i], dummy = np.histogram(pred_vals_std[i][cl], bins=nbins, weights=lumiweights_vals[i][cl])
            y_vals_norm[i], bin_edges_vals[i] = np.histogram(pred_vals_std[i][cl], bins=nbins, weights=weights_vals[i][cl])

            yerrs_norm[i] = y_vals_norm[i]**0.5
            yerrs[i] = y_vals[i]**0.5

            bin_centers[i] = 0.5*(bin_edges_trains[i][1:] + bin_edges_trains[i][:-1])
            y_vals_norm[i] = y_vals_norm[i] * normweights_vals[i][cl][0]
            yerrs_norm[i] = yerrs_norm[i] * normweights_vals[i][cl][0]


        if do_sig:
            for key in pred_signals.keys():
                y_signals[key], dummy = np.histogram(pred_signals_std[key][:,cl], bins=nbins, weights=eventweight_signals[key])
                yerrs_signals[key] = y_signals[key]**0.5

        plt.clf()
        fig = plt.figure()
        classtitle_to_use = ''
        for i in range(labels_train.shape[1]):
            #print("labels_train:",labels_train[i])
            plt.hist(pred_trains_std[i][cl], weights=weights_trains[i][cl]*normweights_trains[i][cl], bins=nbins, histtype='step', label='Training sample, ' + classtitles[i], color=colorstr[i])
            plt.errorbar(bin_centers[i], y_vals_norm[i], yerr=yerrs_norm[i], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Validation sample, ' + classtitles[i], color=colorstr[i])
            if i == cl:
                classtitle_to_use = classtitles[i]
        if do_sig:
            for sigidx in range(len(usesignals)):
                plt.hist(pred_signals_std[usesignals[sigidx]][:,cl], weights=eventweight_signals[usesignals[sigidx]]*normweight_signals[usesignals[sigidx]], bins=nbins, histtype='step', label='Signal (%s)' % signalmasses[usesignals[sigidx]], color='k', linestyle=signal_linestyles[sigidx])


        plt.legend(loc='best', prop={'size': 8})
        plt.yscale('log')
        plt.xlim([-0.05, 1.05])
        plt.xlabel('STD of classifier output for node '+classtitle_to_use)
        plt.ylabel('Normalized number of events / bin')
        title = 'Distribution_node'+str(cl)+'_norm'
        if use_best_model: title += '_best'
        title += '_std.pdf'
        fig.savefig(plotfolder+'/'+title)

        plt.clf()
        fig = plt.figure()
        classtitle_to_use = ''
        for i in range(labels_train.shape[1]):
            plt.errorbar(bin_centers[i], y_vals[i], yerr=yerrs[i], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Validation sample, ' + classtitles[i], color=colorstr[i])
            # print i, y_vals[i]
            if i == cl:
                classtitle_to_use = classtitles[i]
        if do_sig:
            for sigidx in range(len(usesignals)):
                plt.hist(pred_signals_std[usesignals[sigidx]][:,cl], weights=eventweight_signals[usesignals[sigidx]], bins=nbins, histtype='step', label='Signal (%s)' % signalmasses[usesignals[sigidx]], color='k', linestyle=signal_linestyles[sigidx])
                # plt.errorbar(bin_centers[i], y_signals[usesignal], yerr=yerrs_signals[usesignal], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Signal sample', color='k')

        plt.legend(loc='best', prop={'size': 8})
        plt.yscale('log')
        plt.xlim([-0.05, 1.05])
        plt.xlabel('STD of classifier output for node '+classtitle_to_use)
        plt.ylabel('Number of events / bin (weighted by luminosity)')
        title = 'Distribution_node'+str(cl)
        if use_best_model: title += '_best'
        title += '_std.pdf'
        fig.savefig(plotfolder+'/'+title)
        print("Store file: ",plotfolder+'/'+title)
        plt.close()


def plot_outputs_1d_classes(parameters, plotfolder, pred_trains, labels_train, weights_trains, lumiweights_trains, normweights_trains, pred_vals, labels_val, weights_vals, lumiweights_vals, normweights_vals, use_best_model=False):

    print 'Starting to plot the classifier output distribution'
    tag = dict_to_str(parameters)
    classtitles = get_classtitles(parameters)

    for cl in range(labels_train.shape[1]):
        # 'cl' is the output node number
        nbins = 50
        binwidth = 1./float(nbins)
        y_trains = {}
        y_vals = {}
        y_trains_norm = {}
        y_vals_norm = {}
        bin_edges_trains = {}
        bin_edges_vals = {}
        bin_centers = {}
        yerrs = {}
        yerrs_norm = {}
        #
        # print 'class: ', cl
        # print 'integral lumiweighted: ', lumiweights_vals[cl][0].sum()
        # print 'integral lumi+classwe: ', weights_vals[cl][0].sum()
        # print 'integral normed lumi+classwe: ', (weights_vals[cl][0]*normweights_vals[cl][0]).sum()


        for i in range(labels_train.shape[1]):
            # 'i' is the true class (always the first index)
            y_trains[i], dummy = np.histogram(pred_trains[cl][i], bins=nbins, weights=lumiweights_trains[cl][i])
            y_trains_norm[i], bin_edges_trains[i] = np.histogram(pred_trains[cl][i], bins=nbins, weights=weights_trains[cl][i])
            y_vals[i], dummy = np.histogram(pred_vals[cl][i], bins=nbins, weights=lumiweights_vals[cl][i])
            y_vals_norm[i], bin_edges_vals[i] = np.histogram(pred_vals[cl][i], bins=nbins, weights=weights_vals[cl][i])
            bin_centers[i] = 0.5*(bin_edges_trains[i][1:] + bin_edges_trains[i][:-1])
            yerrs_norm[i] = y_vals_norm[i]**0.5
            yerrs[i] = y_vals[i]**0.5
            y_vals_norm[i] = y_vals_norm[i] * normweights_vals[cl][i][0]
            yerrs_norm[i] = yerrs_norm[i] * normweights_vals[cl][i][0]

        plt.clf()
        fig = plt.figure()
        classtitle_to_use = ''
        for i in range(labels_train.shape[1]):
            plt.hist(pred_trains[cl][i], weights=weights_trains[cl][i]*normweights_trains[cl][i], bins=nbins, histtype='step', label='Training sample, ' + classtitles[i], color=colorstr[i])
            plt.errorbar(bin_centers[i], y_vals_norm[i], yerr=yerrs_norm[i], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Validation sample, ' + classtitles[i], color=colorstr[i])
            if i == cl:
                classtitle_to_use = classtitles[i]

        plt.legend(loc='best', prop={'size': 8})
        plt.yscale('log')
        plt.xlim([-0.05, 1.05])
        plt.xlabel('Classifier output for true class '+classtitle_to_use)
        plt.ylabel('Normalized number of events / bin')
        title = 'Distribution_class'+str(cl)+'_norm'
        if use_best_model: title += '_best'
        title += '.pdf'
        fig.savefig(plotfolder+'/'+title)

        plt.clf()
        fig = plt.figure()
        classtitle_to_use = ''
        for i in range(labels_train.shape[1]):
            plt.errorbar(bin_centers[i], y_vals[i], yerr=yerrs[i], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Validation sample, ' + classtitles[i], color=colorstr[i])
            if i == cl:
                classtitle_to_use = classtitles[i]

        plt.legend(loc='best', prop={'size': 8})
        plt.yscale('log')
        plt.xlim([-0.05, 1.05])
        plt.xlabel('Classifier output for true class '+classtitle_to_use)
        plt.ylabel('Number of events / bin (weighted by luminosity)')
        title = 'Distribution_class'+str(cl)+''
        if use_best_model: title += '_best'
        title += '.pdf'
        fig.savefig(plotfolder+'/'+title)
        plt.close()


#def cut_iteratively(parameters, outputfolder, pred_val, labels_val, eventweights_val, pred_signals=None, eventweight_signals=None, usesignals=[0]):
#
#    print 'Starting to cut iteratively'
#    tag = dict_to_str(parameters)
#    classtitles = get_classtitles(parameters)
#    classes = parameters['classes']
#    best_cuts = []
#
#    FalsePositiveRates, TruePositiveRates, Thresholds, aucs, SignalPuritys = get_fpr_tpr_thr_auc(parameters=parameters, pred_val=pred_val, labels_val=labels_val, weights_val=eventweights_val)
#    sign_max = {}
#    thr_cuts = {}
#    n_signals_total = {}
#    for key in pred_signals.keys():
#        n_signals_total[key] = eventweight_signals[key].sum()
#    for cl in range(len(classes)):
#        #find all weights belonging or not belonging to this class
#        nsig = eventweights_val[labels_val[:,cl] == 1].sum()
#        nbkg = eventweights_val[labels_val[:,cl] == 0].sum()
#
#
#        #find the cut maximizing the significance of this class
#        sign_max_this = 0.
#        best_idx = -1
#        eff_signals = {}
#        for key in pred_signals.keys():
#            eff_signals[key] = 0.
#        for idx in range(len(Thresholds[cl])):
#            nsig_sel = TruePositiveRates[cl][idx] * nsig
#            nbkg_sel = FalsePositiveRates[cl][idx] * nbkg
#            if idx%(len(Thresholds[cl])/10000) == 0:
#                for key in pred_signals.keys():
#                    eff_signals[key] = eventweight_signals[key][pred_signals[key][:,cl] > Thresholds[cl][idx]].sum() / float(n_signals_total[key])
#                # consider cut only if it leaves 90% of the signals monitored
#                should_i_stop = False
#                for sigidx in range(len(usesignals)):
#                    if eff_signals[usesignals[sigidx]] > 0.1:
#                        should_i_stop = True
#                if should_i_stop:
#                    break
#            significance = 0.
#            if nsig_sel + nbkg_sel > 0:
#                significance = nsig_sel / (math.sqrt(nsig_sel + nbkg_sel))
#            if significance > sign_max_this:
#                sign_max_this = significance
#                best_idx = idx
#        sign_max[cl] = sign_max_this
#        thr_cuts[cl] = Thresholds[cl][best_idx]
#        print 'best cut for class %s, output > %f, would have a significance of %f, keeping %f out of %f signal events (%f%%), while letting in %f out of %f background events (%f%%). Purity: %f' % (classtitles[cl], Thresholds[cl][best_idx], sign_max_this, TruePositiveRates[cl][best_idx] * nsig, nsig, TruePositiveRates[cl][best_idx]*100., FalsePositiveRates[cl][best_idx] * nbkg, nbkg, FalsePositiveRates[cl][best_idx]*100., TruePositiveRates[cl][best_idx] * nsig / (TruePositiveRates[cl][best_idx] * nsig + FalsePositiveRates[cl][best_idx] * nbkg))
#
#
#    #cut on distribution with highest significance
#    highest_sign = 0.
#    best_idx = -1
#    for cl in sign_max.keys():
#        if sign_max[cl] > highest_sign:
#            highest_sign = sign_max[cl]
#            best_idx = cl
#    # now 'best_idx' is the class we cut on
#    best_cuts.append({best_idx: thr_cuts[best_idx]})
#    mask = [pred_val[:,best_idx] > thr_cuts[best_idx]]
#    pred_val = pred_val[mask]
#    labels_val = labels_val[mask]
#    eventweights_val = eventweights_val[mask]
#    return best_cuts
#
#
#
#def plot_cuts(parameters, outputfolder, plotfolder, best_cuts, pred_vals, labels_val, lumiweights_vals, pred_signals=None, eventweight_signals=None, usesignals=[0], use_best_model=False):
#
#    print 'plotting the cuts'
#    do_sig = (pred_signals is not None) and (eventweight_signals is not None)
#    tag = dict_to_str(parameters)
#    classtitles = get_classtitles(parameters)
#    pred_vals_pass = deepcopy(pred_vals)
#    pred_vals_fail = deepcopy(pred_vals)
#    pred_signals_pass = deepcopy(pred_signals)
#    pred_signals_fail = deepcopy(pred_signals)
#    lumiweights_vals_pass = deepcopy(lumiweights_vals)
#    lumiweights_vals_fail = deepcopy(lumiweights_vals)
#    eventweight_signals_pass = deepcopy(eventweight_signals)
#    eventweight_signals_fail = deepcopy(eventweight_signals)
#
#    for k in range(len(best_cuts)):
#        # k is the number of consecutive cuts applied (k-1 have already been applied, only applying cut no. k in this turn)
#        if k > 0: continue
#
#        cut = best_cuts[k]
#        cutclass = cut.keys()[0]
#
#
#        n_total = {}
#        n_pass = {}
#        n_fail = {}
#        sum_pass = 0.
#        sum_fail = 0.
#        sum_total = 0.
#        for j in range(len(pred_vals)):
#            # j is the true class
#
#            # this means: we want to cut on the output of the true class 'j' predicted by node 'cutclass', because each cut always acts on exactly one node
#            # this cut is applied to events having failed earlier cuts -> events that are still in the game
#            mask_pass = pred_vals_fail[j][cutclass] > cut[cutclass]
#            mask_fail = pred_vals_fail[j][cutclass] <= cut[cutclass]
#            for m in range(len(pred_vals[j])):
#                # m is the output node we want to cut on
#                # irrespective of the node we are cutting on, a given true class 'j' must always have the same number of events in each node
#                # this cut is applied to events having failed earlier cuts -> events that are still in the game
#
#                pred_vals_pass[j][m] = deepcopy(pred_vals_fail[j][m][mask_pass])
#                lumiweights_vals_pass[j][m] = deepcopy(lumiweights_vals_fail[j][m][mask_pass])
#                pred_vals_fail[j][m] = deepcopy(pred_vals_fail[j][m][mask_fail])
#                lumiweights_vals_fail[j][m] = deepcopy(lumiweights_vals_fail[j][m][mask_fail])
#
#            n_pass[j] = lumiweights_vals_pass[j][0].sum()
#            n_fail[j] = lumiweights_vals_fail[j][0].sum()
#            n_total[j] = lumiweights_vals[j][0].sum()
#            sum_pass += lumiweights_vals_pass[j][0].sum()
#            sum_fail += lumiweights_vals_fail[j][0].sum()
#            sum_total += lumiweights_vals[j][0].sum()
#
#        eff_signals = {}
#        for key in pred_signals.keys():
#            eff_signals[key] = 0.
#        if do_sig:
#            for key in pred_signals.keys():
#                mask_signal_pass = pred_signals_fail[key][:,cutclass] > cut[cutclass]
#                mask_signal_fail = pred_signals_fail[key][:,cutclass] <= cut[cutclass]
#                pred_signals_pass[key] = deepcopy(pred_signals_fail[key][mask_signal_pass])
#                eventweight_signals_pass[key] = deepcopy(eventweight_signals_fail[key][mask_signal_pass])
#                pred_signals_fail[key] = deepcopy(pred_signals_fail[key][mask_signal_fail])
#                eventweight_signals_fail[key] = deepcopy(eventweight_signals_fail[key][mask_signal_fail])
#                eff_signals[key] = eventweight_signals_pass[key].sum() / float(eventweight_signals[key].sum())
#
#        # Log this cut
#        effs = {}
#        fractions_pass = {}
#        fractions_fail = {}
#        fractions_total = {}
#        for j in range(len(pred_vals)):
#            effs[j] = n_pass[j] / n_total[j]
#            fractions_pass[j] = n_pass[j] / sum_pass
#            fractions_fail[j] = n_fail[j] / sum_fail
#            fractions_total[j] = n_total[j] / sum_total
#
#        str_effs = '('
#        str_fracs_pass = '('
#        str_fracs_fail = '('
#        str_fracs_total = '('
#        for j in range(len(pred_vals)):
#            str_effs += '{0:3.3f}'.format(effs[j])
#            str_fracs_pass += '{0:3.3f}'.format(fractions_pass[j])
#            str_fracs_fail += '{0:3.3f}'.format(fractions_fail[j])
#            str_fracs_total += '{0:3.3f}'.format(fractions_total[j])
#            if j < len(pred_vals)-1:
#                str_effs += ','
#                str_fracs_pass += ','
#                str_fracs_fail += ','
#                str_fracs_total += ','
#        str_effs += ')'
#        str_fracs_pass += ')'
#        str_fracs_fail += ')'
#        str_fracs_total += ')'
#
#        str_eff_signals = '('
#        for j in range(len(usesignals)):
#            str_eff_signals += '{0:3.3f}'.format(eff_signals[usesignals[j]])
#            if j < len(usesignals)-1:
#                str_eff_signals += ','
#        str_eff_signals += ')'
#
#        title = 'CutPerformance'
#        if use_best_model: title += '_best'
#        title += '.txt'
#        with open(outputfolder+'/'+title, 'w') as f:
#            f.write('Tag: %s\n\n' % (tag))
#            f.write('cutclass, cutvalue (val >= cutvalue), (signal efficiency), (efficiencies), (fractions before cut), (fractions after cut), (fractions left)\n')
#            f.write('{0}, {1}, {2}, {3}, {4}, {5}, {6}\n'.format(cutclass, cut[cutclass], str_eff_signals, str_effs, str_fracs_total, str_fracs_pass, str_fracs_fail))
#
#
#        for cl in range(len(pred_vals)):
#            # 'cl' is the output node number
#            nbins = 100
#            binwidth = 1./float(nbins)
#            y_vals_pass = {}
#            y_vals_fail = {}
#            bin_edges_vals = {}
#            bin_edges_vals_pass = {}
#            bin_edges_vals_fail = {}
#            bin_centers_pass = {}
#            bin_centers_fail = {}
#            yerrs_pass = {}
#            yerrs_fail = {}
#
#            for i in range(len(pred_vals)):
#                # 'i' is the true class (always the first index)
#                y_vals_pass[i], bin_edges_vals_pass[i] = np.histogram(pred_vals_pass[i][cl], bins=nbins, weights=lumiweights_vals_pass[i][cl])
#                bin_centers_pass[i] = 0.5*(bin_edges_vals_pass[i][1:] + bin_edges_vals_pass[i][:-1])
#                yerrs_pass[i] = y_vals_pass[i]**0.5
#                y_vals_fail[i], bin_edges_vals_fail[i] = np.histogram(pred_vals_fail[i][cl], bins=nbins, weights=lumiweights_vals_fail[i][cl])
#                bin_centers_fail[i] = 0.5*(bin_edges_vals_fail[i][1:] + bin_edges_vals_fail[i][:-1])
#                yerrs_fail[i] = y_vals_fail[i]**0.5
#
#            y_signals_pass = {}
#            y_signals_fail = {}
#            yerrs_signals_pass = {}
#            yerrs_signals_fail = {}
#            if do_sig:
#                for key in pred_signals.keys():
#                    y_signals_pass[key], dummy = np.histogram(pred_signals_pass[key][:,cl], bins=nbins, weights=eventweight_signals_pass[key])
#                    yerrs_signals_pass[key] = y_signals_pass[key]**0.5
#                    y_signals_fail[key], dummy = np.histogram(pred_signals_fail[key][:,cl], bins=nbins, weights=eventweight_signals_fail[key])
#                    yerrs_signals_fail[key] = y_signals_fail[key]**0.5
#
#            # Passing events
#            plt.clf()
#            fig = plt.figure()
#            classtitle_to_use = ''
#            for i in range(len(pred_vals_pass)):
#                plt.errorbar(bin_centers_pass[i], y_vals_pass[i], yerr=yerrs_pass[i], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Validation sample, ' + classtitles[i], color=colorstr[i])
#                if i == cl:
#                    classtitle_to_use = classtitles[i]
#
#            if do_sig:
#                for sigidx in range(len(usesignals)):
#                    plt.hist(pred_signals_pass[usesignals[sigidx]][:,cl], weights=eventweight_signals_pass[usesignals[sigidx]], bins=nbins, histtype='step', label='Signal (%s)' % signalmasses[usesignals[sigidx]], color='k', linestyle=signal_linestyles[sigidx])
#                # plt.errorbar(bin_centers[i], y_signals[usesignal], yerr=yerrs_signals[usesignal], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Signal sample', color='k')
#            plt.legend(loc='best', prop={'size': 8})
#            plt.yscale('log')
#            plt.xlim([-0.05, 1.05])
#            plt.xlabel('Classifier output for node '+classtitle_to_use)
#            plt.ylabel('Number of events / bin (weighted by luminosity)')
#            title = 'Distribution_node'+str(cl)+'_enrichedclass' + str(cutclass)
#            if use_best_model: title += '_best'
#            title += '.pdf'
#            fig.savefig(plotfolder+'/'+title)
#            plt.close()
#
#            # Failing events
#            plt.clf()
#            fig = plt.figure()
#            classtitle_to_use = ''
#            for i in range(len(pred_vals_fail)):
#                plt.errorbar(bin_centers_fail[i], y_vals_fail[i], yerr=yerrs_fail[i], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Validation sample, ' + classtitles[i], color=colorstr[i])
#                if i == cl:
#                    classtitle_to_use = classtitles[i]
#
#            if do_sig:
#                for sigidx in range(len(usesignals)):
#                    plt.hist(pred_signals_fail[usesignals[sigidx]][:,cl], weights=eventweight_signals_fail[usesignals[sigidx]], bins=nbins, histtype='step', label='Signal (%s)' % signalmasses[usesignals[sigidx]], color='k', linestyle=signal_linestyles[sigidx])
#                # plt.errorbar(bin_centers[i], y_signals[usesignal], yerr=yerrs_signals[usesignal], fmt = '.', drawstyle = 'steps-mid', linestyle=' ', label='Signal sample', color='k')
#            plt.legend(loc='best', prop={'size': 8})
#            plt.yscale('log')
#            plt.xlim([-0.05, 1.05])
#            plt.xlabel('Classifier output for node '+classtitle_to_use)
#            plt.ylabel('Number of events / bin (weighted by luminosity)')
#            title = 'Distribution_node'+str(cl)+'_failingcut' + str(cutclass)
#            if use_best_model: title += '_best'
#            title += '.pdf'
#            fig.savefig(plotfolder+'/'+title)
#            plt.close()
#
#def apply_cuts(parameters, outputfolder, best_cuts, input_train, input_val, input_test, labels_train, labels_val, labels_test, sample_weights_train, sample_weights_val, sample_weights_test, eventweights_train, eventweights_val, eventweights_test, pred_train, pred_val, pred_test, signals=None, eventweight_signals=None, pred_signals=None, signal_identifiers=None, use_best_model=False):
#
#    print 'Now applying the cuts'
#    do_sig = (pred_signals is not None) and (eventweight_signals is not None) and (signals is not None) and (signal_identifiers is not None)
#    tag = dict_to_str(parameters)
#    classtitles = get_classtitles(parameters)
#    fraction = get_fraction(parameters)
#
#    cut = best_cuts[0]
#    cutclass = cut.keys()[0]
#    cutvalue = cut[cutclass]
#
#    # Define masks for passing and failing the cut
#    mask_train_pass = pred_train[:,cutclass] > cutvalue
#    mask_train_fail = pred_train[:,cutclass] <= cutvalue
#    mask_val_pass = pred_val[:,cutclass] > cutvalue
#    mask_val_fail = pred_val[:,cutclass] <= cutvalue
#    mask_test_pass = pred_test[:,cutclass] > cutvalue
#    mask_test_fail = pred_test[:,cutclass] <= cutvalue
#    mask_signals_pass = {}
#    mask_signals_fail = {}
#    if do_sig:
#        for key in pred_signals.keys():
#            mask_signals_pass[key] = pred_signals[key][:,cutclass] > cutvalue
#            mask_signals_fail[key] = pred_signals[key][:,cutclass] <= cutvalue
#
#    # Make deepcopies of pass and fail arrays
#    input_train_pass = deepcopy(input_train)
#    input_val_pass = deepcopy(input_val)
#    input_test_pass = deepcopy(input_test)
#    labels_train_pass = deepcopy(labels_train)
#    labels_val_pass = deepcopy(labels_val)
#    labels_test_pass = deepcopy(labels_test)
#    sample_weights_train_pass = deepcopy(sample_weights_train)
#    sample_weights_val_pass = deepcopy(sample_weights_val)
#    sample_weights_test_pass = deepcopy(sample_weights_test)
#    eventweights_train_pass = deepcopy(eventweights_train)
#    eventweights_val_pass = deepcopy(eventweights_val)
#    eventweights_test_pass = deepcopy(eventweights_test)
#    signals_pass = {}
#    eventweight_signals_pass = {}
#    pred_signals_pass = {}
#    if do_sig:
#        for key in pred_signals.keys():
#            signals_pass[key] =   deepcopy(signals[key])
#            eventweight_signals_pass[key] = deepcopy(eventweight_signals[key])
#            pred_signals_pass[key] = deepcopy(pred_signals[key])
#
#    input_train_fail = deepcopy(input_train)
#    input_val_fail = deepcopy(input_val)
#    input_test_fail = deepcopy(input_test)
#    labels_train_fail = deepcopy(labels_train)
#    labels_val_fail = deepcopy(labels_val)
#    labels_test_fail = deepcopy(labels_test)
#    sample_weights_train_fail = deepcopy(sample_weights_train)
#    sample_weights_val_fail = deepcopy(sample_weights_val)
#    sample_weights_test_fail = deepcopy(sample_weights_test)
#    eventweights_train_fail = deepcopy(eventweights_train)
#    eventweights_val_fail = deepcopy(eventweights_val)
#    eventweights_test_fail = deepcopy(eventweights_test)
#    signals_fail = {}
#    eventweight_signals_fail = {}
#    pred_signals_fail = {}
#    if do_sig:
#        for key in pred_signals.keys():
#            signals_fail[key] = deepcopy(signals[key])
#            eventweight_signals_fail[key] = deepcopy(eventweight_signals[key])
#            pred_signals_fail[key] = deepcopy(pred_signals[key])
#
#    # Apply masks
#    input_train_pass = input_train_pass[mask_train_pass]
#    input_val_pass = input_val_pass[mask_val_pass]
#    input_test_pass = input_test_pass[mask_test_pass]
#    labels_train_pass = labels_train_pass[mask_train_pass]
#    labels_val_pass = labels_val_pass[mask_val_pass]
#    labels_test_pass = labels_test_pass[mask_test_pass]
#    sample_weights_train_pass = sample_weights_train_pass[mask_train_pass]
#    sample_weights_val_pass = sample_weights_val_pass[mask_val_pass]
#    sample_weights_test_pass = sample_weights_test_pass[mask_test_pass]
#    eventweights_train_pass = eventweights_train_pass[mask_train_pass]
#    eventweights_val_pass = eventweights_val_pass[mask_val_pass]
#    eventweights_test_pass = eventweights_test_pass[mask_test_pass]
#    if do_sig:
#        for key in pred_signals.keys():
#            signals_pass[key] = signals_pass[key][mask_signals_pass[key]]
#            eventweight_signals_pass[key] = eventweight_signals_pass[key][mask_signals_pass[key]]
#            pred_signals_pass[key] = pred_signals_pass[key][mask_signals_pass[key]]
#
#    input_train_fail = input_train_fail[mask_train_fail]
#    input_val_fail = input_val_fail[mask_val_fail]
#    input_test_fail = input_test_fail[mask_test_fail]
#    labels_train_fail = labels_train_fail[mask_train_fail]
#    labels_val_fail = labels_val_fail[mask_val_fail]
#    labels_test_fail = labels_test_fail[mask_test_fail]
#    sample_weights_train_fail = sample_weights_train_fail[mask_train_fail]
#    sample_weights_val_fail = sample_weights_val_fail[mask_val_fail]
#    sample_weights_test_fail = sample_weights_test_fail[mask_test_fail]
#    eventweights_train_fail = eventweights_train_fail[mask_train_fail]
#    eventweights_val_fail = eventweights_val_fail[mask_val_fail]
#    eventweights_test_fail = eventweights_test_fail[mask_test_fail]
#    if do_sig:
#        for key in pred_signals.keys():
#            signals_fail[key] = signals_fail[key][mask_signals_fail[key]]
#            eventweight_signals_fail[key] = eventweight_signals_fail[key][mask_signals_fail[key]]
#            pred_signals_fail[key] = pred_signals_fail[key][mask_signals_fail[key]]
#
#    # Save outputs according to model used (last or best)
#    ending = ''
#    if use_best_model: ending += '_best'
#    ending += '.npy'
#
#    if not os.path.isdir(outputfolder+'/cut'): os.makedirs(outputfolder+'/cut')
#    np.save(outputfolder+'/cut/input_'+fraction+'_train_pass'+ending           , input_train_pass)
#    np.save(outputfolder+'/cut/input_'+fraction+'_test_pass'+ending            , input_test_pass)
#    np.save(outputfolder+'/cut/input_'+fraction+'_val_pass'+ending             , input_val_pass)
#    np.save(outputfolder+'/cut/labels_'+fraction+'_train_pass'+ending          , labels_train_pass)
#    np.save(outputfolder+'/cut/labels_'+fraction+'_test_pass'+ending           , labels_test_pass)
#    np.save(outputfolder+'/cut/labels_'+fraction+'_val_pass'+ending            , labels_val_pass)
#    np.save(outputfolder+'/cut/sample_weights_'+fraction+'_train_pass'+ending  , sample_weights_train_pass)
#    np.save(outputfolder+'/cut/eventweights_'+fraction+'_train_pass'+ending    , eventweights_train_pass)
#    np.save(outputfolder+'/cut/sample_weights_'+fraction+'_test_pass'+ending   , sample_weights_test_pass)
#    np.save(outputfolder+'/cut/eventweights_'+fraction+'_test_pass'+ending     , eventweights_test_pass)
#    np.save(outputfolder+'/cut/sample_weights_'+fraction+'_val_pass'+ending    , sample_weights_val_pass)
#    np.save(outputfolder+'/cut/eventweights_'+fraction+'_val_pass'+ending      , eventweights_val_pass)
#    if do_sig:
#        for key in pred_signals.keys():
#            np.save(outputfolder+'/cut/'+signal_identifiers[key]+'_pass'+ending                , signals_pass[key])
#            np.save(outputfolder+'/cut/'+signal_identifiers[key]+'_eventweight_pass'+ending    , eventweight_signals_pass[key])
#
#    np.save(outputfolder+'/cut/input_'+fraction+'_train_fail'+ending           , input_train_fail)
#    np.save(outputfolder+'/cut/input_'+fraction+'_test_fail'+ending            , input_test_fail)
#    np.save(outputfolder+'/cut/input_'+fraction+'_val_fail'+ending             , input_val_fail)
#    np.save(outputfolder+'/cut/labels_'+fraction+'_train_fail'+ending          , labels_train_fail)
#    np.save(outputfolder+'/cut/labels_'+fraction+'_test_fail'+ending           , labels_test_fail)
#    np.save(outputfolder+'/cut/labels_'+fraction+'_val_fail'+ending            , labels_val_fail)
#    np.save(outputfolder+'/cut/sample_weights_'+fraction+'_train_fail'+ending  , sample_weights_train_fail)
#    np.save(outputfolder+'/cut/eventweights_'+fraction+'_train_fail'+ending    , eventweights_train_fail)
#    np.save(outputfolder+'/cut/sample_weights_'+fraction+'_test_fail'+ending   , sample_weights_test_fail)
#    np.save(outputfolder+'/cut/eventweights_'+fraction+'_test_fail'+ending     , eventweights_test_fail)
#    np.save(outputfolder+'/cut/sample_weights_'+fraction+'_val_fail'+ending    , sample_weights_val_fail)
#    np.save(outputfolder+'/cut/eventweights_'+fraction+'_val_fail'+ending      , eventweights_val_fail)
#    if do_sig:
#        for key in pred_signals.keys():
#            np.save(outputfolder+'/cut/'+signal_identifiers[key]+'_fail'+ending                , signals_fail[key])
#            np.save(outputfolder+'/cut/'+signal_identifiers[key]+'_eventweight_fail'+ending    , eventweight_signals_fail[key])
#

class lr_reduce_on_plateau:
    """Class for reducing lr when given validation loss is not improving
    any longer.

    Args:
        lr:       inital learning rate (float)
        patience: number of times loss doesn't improve
            until reduce on plateau is triggered
        fraction: determines lr update after triggering reduce on plateau
            lr = lr*fraction (should be float between 0 and 1)
        delte_improv: loss value has to be smaller than best loss value
            minus delta_improv to count as an improvment

    Example for usage:
        lr = lr_reduce_on_plateau(
            lr=0.1, patience=3, fraction=0.1)
        for i in n_epochs:
            ...
            sess.run(train_op, feed_dict={lr: lr.lr})
            lr.update_on_plateau(val_loss)
    """

    def __init__(self, lr, patience, delta_improv, fraction):
        self.patience = patience
        self.fraction = fraction
        self.delta_improv = delta_improv
        self.lr = lr # initial lr
        self.best_loss = None
        self.counter = 0

    def _update(self, x):
        if self.best_loss == None:
            self.best_loss = x
        elif (self.best_loss - x) > self.delta_improv:
            self.counter = 0.
        else:
            self.counter +=1
        if x < self.best_loss:
            self.best_loss = x

    def _lr_reduce(self):
        self.lr = self.lr*self.fraction

    def update_on_plateau(self, x):
        self._update(x)
        if self.counter >= self.patience:
            self._lr_reduce()
            self.counter = 0. # reset counter
            print("Reduce on Plateau triggered! (patience: {:}, fraction: {:}, ".format(
                self.patience, self.fraction) +\
                "delta_improv: {:}, new lr: {:})".format(self.delta_improv, self.lr))



class check_early_stopping:
    """Class for Early Stopping. Early Stopping stopps training
    after validation loss hasen't improved for patience-time.
    Look into example for usgae (check mehtod just gives Boolen output).

    Args:
        patience: number of times loss doesn't improve
            until Early Stopping is triggered
        delte_improv: loss value has to be smaller than best loss value
            minus delta_improv to count as an improvment

    Example for usage:
        check = check_early_stopping(
            patience=3, fraction=0.1)
        for i in n_epochs:
            ...
            sess.run(train_op)
            if check.update_and_check(val_loss):
                break
    """

    def __init__(self, patience, delta_improv=0.):
        self.patience = patience
        self.delta_improv = delta_improv
        self.best_loss = None
        self.counter = 0

    def _update(self, x):
        if self.best_loss == None:
            self.best_loss = x
        elif (self.best_loss - x) > self.delta_improv:
            self.counter = 0.
        else:
            self.counter +=1
        if x < self.best_loss:
            self.best_loss = x

    def update_and_check(self, x):
        self._update(x)
        if self.counter >= self.patience:
            print("Early Stopping triggered! (patience: {:}, delta_improv: {:})".format(
                self.patience, self.delta_improv))
            return True
        else:
            return False


def plot_prediction_samples(parameters, plotfolder, pred_train_all, labels_train, eventID):
    if not os.path.isdir(plotfolder): os.makedirs(plotfolder)

    print 'Starting to plot prediction in all samples for 1 event and 1 class'
    tag = dict_to_str(parameters)
    classtitles = get_classtitles(parameters)
#    y_trains[i], dummy = np.histogram(pred_trains_all[:,eventID,class_label], bins=nbins)

    nbins = 25
    binwidth = 1./float(nbins)

    plt.clf()
    fig = plt.figure()
    true_lable = 'true lable: '
    for i in range(labels_train.shape[1]):
        plt.hist(pred_train_all[:,eventID,i], bins=nbins, histtype='step', label='Training sample, prediction for ' + classtitles[i]+' node', color=colorstr[i])
        if(labels_train[eventID][i]>0):
            true_lable = true_lable + classtitles[i]
    plt.legend(loc='best', prop={'size': 8})
    plt.yscale('log')
    plt.xlim([-0.05, 1.05])
#    plt.xlabel('Classifier output for class #'+str(class_label)+', event '+str(eventID))
    plt.xlabel('Classifier output for event '+str(eventID)+', '+true_lable)
    plt.ylabel('Number of events / bin')

    #title = 'Distribution_event'+str(eventID)+'_class'+str(class_label)
    title = 'Distribution_train_event'+str(eventID)
    title += '.pdf'
    fig.savefig(plotfolder+'/'+title)
    plt.close()
