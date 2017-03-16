#!/usr/bin/python

import os
import sys
import numpy as np
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB

###############################################################################

# Global Dictionaries & TransTable
dictionary = {'love':0, 'wonderful':1, 'best':2, 'great':3, 'superb':4, 'still':5, 'beautiful':6,
             'bad':7, 'worst':8, 'stupid':9, 'waste':10, 'boring':11, '?':12, '!':13, '<UNK>':14}

vocabulary = ['love', 'wonderful', 'best', 'great', 'superb', 'still', 'beautiful',
              'bad', 'worst', 'stupid', 'waste', 'boring', '?', '!', '<UNK>']

tran_table = {ord(c): None for c in string.punctuation.replace('?', '').replace('!', '')}

# Transfer func
def transfer(fileDj):
    raw = open(fileDj, 'r').read()
    tokens = raw.replace('loving', 'love').replace('loves','love').replace('loved', 'love').split()
    BOWDj = [0] * 15
    for token in tokens:
        if token in vocabulary:
            BOWDj[dictionary[token]] += 1
        else:
            BOWDj[14] += 1
    return BOWDj

# Load data
def loadData(Path):

    # Process Training data
    pos_path = Path+'training_set/pos/'
    neg_path = Path+'training_set/neg/'
    xtrain_pos = os.listdir(pos_path)
    xtrain_neg = os.listdir(neg_path)

    xtrain = []
    for file_name in xtrain_pos: xtrain.append(transfer(pos_path+file_name))
    for file_name in xtrain_neg: xtrain.append(transfer(neg_path+file_name))

    ytrain = np.asarray([1] * len(xtrain_pos) + [0] * len(xtrain_neg))
    Xtrain = np.asarray(xtrain)

    # Process testing data
    pos_path = Path+'test_set/pos/'
    neg_path = Path+'test_set/neg/'
    xtest_pos = os.listdir(pos_path)
    xtest_neg = os.listdir(neg_path)

    xtest = []
    for file_name in xtest_pos: xtest.append(transfer(pos_path+file_name))
    for file_name in xtest_neg: xtest.append(transfer(neg_path+file_name))

    ytest = np.asarray([1] * len(xtest_pos) + [0] * len(xtest_neg))
    Xtest = np.asarray(xtest)

    return Xtrain, Xtest, ytrain, ytest

# Naive multinomial Bayes, training step
def naiveBayesMulFeature_train(Xtrain, ytrain):
    thetaPos = []
    thetaNeg = []

    # separate data with pos and neg labels
    pos_range = int(len(ytrain.flatten())/2)
    pos_X = (Xtrain[0:pos_range+1]).transpose()
    neg_X = (Xtrain[pos_range+1:len(ytrain)+1]).transpose()
    sum_pos = np.sum([row.sum() for row in pos_X])
    sum_neg = np.sum([row.sum() for row in neg_X])

    # Calculate probabilities
    for i in np.arange(0, 15):
        pos_prob = (np.sum(pos_X[i])+1)/(sum_pos+15)
        neg_prob = (np.sum(neg_X[i])+1)/(sum_neg+15)
        thetaPos.append(pos_prob)
        thetaNeg.append(neg_prob)

    return thetaPos, thetaNeg

# Naive multinomial Bayes, testing step
def naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg):
    yPredict = []
    log_thetaPos = [np.log(theta) for theta in thetaPos]
    log_thetaNeg = [np.log(theta) for theta in thetaNeg]
    for doc in Xtest:
        prob_pos = np.dot(doc, log_thetaPos)
        prob_neg = np.dot(doc, log_thetaNeg)
        yPredict.append(prob_pos >= prob_neg)
    comp = [int(yPredict[i] == ytest[i]) for i in np.arange(0, len(ytest))]
    Accuracy = len([i for i in comp if i is 1])/len(comp)

    return yPredict, Accuracy

# Sklearn's implementation of MNBC
def naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest):
    clf = MultinomialNB(alpha=1)
    clf.fit(Xtrain, ytrain)
    return clf.score(Xtest, ytest)

# For naiveBayesMulFeature_testDirect, compute log likelihood on the fly and yield YPredict
def naiveBayesMulFeature_testDirectOne(path,thetaPos, thetaNeg, vocabulary):
    yPredict = []
    log_thetaPos = [np.log(theta) for theta in thetaPos]
    log_thetaNeg = [np.log(theta) for theta in thetaNeg]

    pos_path = path+'pos/'
    neg_path = path+'neg/'
    pos_list = os.listdir(pos_path)
    neg_list = os.listdir(neg_path)

    for pos_doc in pos_list:
        raw = open(pos_path+pos_doc, 'r').read()
        tokens = raw.replace('loving', 'love').replace('loves', 'love').replace('loved', 'love').split()
        pos_likeli = 0
        neg_likeli = 0
        for token in tokens:
            if token in vocabulary:
                pos_likeli += log_thetaPos[dictionary[token]]
                neg_likeli += log_thetaNeg[dictionary[token]]
            else:
                pos_likeli += log_thetaPos[dictionary['<UNK>']]
                neg_likeli += log_thetaNeg[dictionary['<UNK>']]
        yPredict.append(int(pos_likeli > neg_likeli))

    for neg_doc in neg_list:
        raw = open(neg_path+neg_doc, 'r').read()
        tokens = raw.replace('loving', 'love').replace('loves', 'love').replace('loved', 'love').split()
        pos_likeli = 0
        neg_likeli = 0
        for token in tokens:
            if token in vocabulary:
                pos_likeli += log_thetaPos[dictionary[token]]
                neg_likeli += log_thetaNeg[dictionary[token]]
            else:
                pos_likeli += log_thetaPos[dictionary['<UNK>']]
                neg_likeli += log_thetaNeg[dictionary['<UNK>']]
        yPredict.append(int(pos_likeli > neg_likeli))

    return yPredict

# Classifiy test documents without explicit BOWs
def naiveBayesMulFeature_testDirect(path,thetaPos, thetaNeg, vocabulary):
    yPredict = naiveBayesMulFeature_testDirectOne(path,thetaPos, thetaNeg, vocabulary)

    pos_path = path+'/pos/'
    neg_path = path+'/neg/'
    pos_list = os.listdir(pos_path)
    neg_list = os.listdir(neg_path)
    ytest = [1] * len(pos_list) + [0] * len(neg_list)
    #print(yPredict)
    #print(ytest)
    comp = [int(yPredict[i] == ytest[i]) for i in np.arange(0, len(ytest))]
    Accuracy = len([i for i in comp if i > 0])/len(comp)
    return yPredict, Accuracy

# Multivariate model for bayesian classifier, training phase
def naiveBayesBernFeature_train(Xtrain, ytrain):
    thetaPosTrue = []
    thetaNegTrue = []

    # divide data in pos and neg entries
    pos_range = int(len(ytrain.flatten())/2)
    pos_X = (Xtrain[0:pos_range]).transpose()
    neg_X = (Xtrain[pos_range:len(ytrain)+1]).transpose()
    pos_count = len(pos_X[0])
    neg_count = len(neg_X[0])

    # Calculate probabilities
    for i in np.arange(0, 15):
        pos_prob = (len([entry1 for entry1 in pos_X[i] if entry1 > 0])+1)/(pos_count+2)
        neg_prob = (len([entry2 for entry2 in neg_X[i] if entry2 > 0])+1)/(neg_count+2)
        thetaPosTrue.append(pos_prob)
        thetaNegTrue.append(neg_prob)

    return thetaPosTrue, thetaNegTrue

# Multivariate model for bayesian classifier, testing phase
def naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []
    # Calculate log likelihood of pos-true, pos-false, neg-true and neg-false
    log_thetaPosTrue = [np.log(theta) for theta in thetaPosTrue]
    log_thetaPosFalse = [np.log(1 - theta) for theta in thetaPosTrue]
    log_thetaNegTrue = [np.log(theta) for theta in thetaNegTrue]
    log_thetaNegFalse = [np.log(1 - theta) for theta in thetaNegTrue]
    ran = np.arange(0, 14)

    # Give prediction and score
    for doc in Xtest:
        true_index = [i for i in ran if doc[i] > 0]
        false_index = [i for i in ran if doc[i] == 0]
        prob_pos = np.sum([log_thetaPosTrue[i] for i in true_index] + [log_thetaPosFalse[i] for i in false_index])
        prob_neg = np.sum([log_thetaNegTrue[i] for i in true_index] + [log_thetaNegFalse[i] for i in false_index])
        yPredict.append(int(prob_pos > prob_neg))
    comp = [int(yPredict[i] == ytest[i]) for i in np.arange(0, len(ytest))]
    Accuracy = len([i for i in comp if i == 1])/len(comp)
    return yPredict, Accuracy

# Load data with stemmer
def loadData_with_stemmer(Path):
    stemmer = PorterStemmer()
    exclude = string.punctuation.replace('?', '').replace('!', '')
    tran_table = {ord(c): None for c in exclude}
    set_stop = set(stopwords.words('english'))

    train_pos_path = Path+'/training_set/pos/'
    train_neg_path = Path+'/training_set/neg/'
    test_pos_path = Path+'/test_set/pos/'
    test_neg_path = Path+'/test_set/neg/'
    xtrain_pos = os.listdir(train_pos_path)
    xtrain_neg = os.listdir(train_neg_path)
    xtest_pos = os.listdir(test_pos_path)
    xtest_neg = os.listdir(test_neg_path)

    all_files = [train_pos_path+filename for filename in xtrain_pos] \
                + [train_neg_path+filename for filename in xtrain_neg]\
                + [test_pos_path+filename for filename in xtest_pos]\
                + [test_neg_path+filename for filename in xtest_neg]
    #for file in all_files:


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python naiveBayes.py dataSetPath testSetPath")
        sys.exit()

    print("--------------------")
    textDataSetsDirectoryFullPath = sys.argv[1]
    testFileDirectoryFullPath = sys.argv[2]

    #Xtrain, Xtest, ytrain, ytest = loadData('./data_sets')
    Xtrain, Xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)


    thetaPos, thetaNeg = naiveBayesMulFeature_train(Xtrain, ytrain)
    print("thetaPos =", thetaPos)
    print("thetaNeg =", thetaNeg)
    print("--------------------")
    yPredict, Accuracy = naiveBayesMulFeature_test(Xtest, ytest, thetaPos, thetaNeg)
    print("MNBC classification accuracy =", Accuracy)


    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(Xtrain, ytrain, Xtest, ytest)
    print("Sklearn MultinomialNB accuracy =", Accuracy_sk)

    yPredict, Accuracy = naiveBayesMulFeature_testDirect(testFileDirectoryFullPath, thetaPos, thetaNeg, vocabulary)
    #yPredict, Accuracy = naiveBayesMulFeature_testDirect('./data_sets/test_set', thetaPos, thetaNeg, vocabulary)
    print("Directly MNBC testing accuracy =", Accuracy)
    print("--------------------")

    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(Xtrain, ytrain)
    print("thetaPosTrue =", thetaPosTrue)
    print("thetaNegTrue =", thetaNegTrue)
    print("--------------------")

    yPredict, Accuracy = naiveBayesBernFeature_test(Xtest, ytest, thetaPosTrue, thetaNegTrue)
    print("BNBC classification accuracy =", Accuracy)
    print("--------------------")
