#!/usr/bin/env python

import sys
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import RandomizedPCA
dtree_para = {'max_depth':200, 'max_leaf_nodes':300, 'max_features':200}
knn_para = [2, 3, 4, 5, 10]
num_of_component = [1, 10, 100]

def load_data(train, test):
    Xtrain = []
    ytrain = []
    Xtest = []
    ytest = []
    file_train = open(train, 'r')
    file_test = open(test, 'r')
    while True:
        raw = file_train.readline()
        if raw is None or raw is '':break
        data = raw.split()
        Xtrain.append([float(pixel) for pixel in data[1:]])
        ytrain.append(float(data[0]))
    while True:
        raw = file_test.readline()
        if raw is None or raw is '':break
        data = raw.split()
        Xtest.append([float(pixel) for pixel in data[1:]])
        ytest.append(float(data[0]))
    return np.asarray(Xtrain), np.asarray(ytrain), np.asarray(Xtest), np.asarray(ytest)

# Finished
def decision_tree(train, test):
    y = []
    Xtrain, ytrain, Xtest, ytest = load_data(train, test)
    num_features = len(Xtrain[0])
    if num_features < dtree_para['max_features']: dtree_para['max_features'] = num_features
    dtree = DecisionTreeClassifier(
            max_depth=dtree_para['max_depth'],
            max_leaf_nodes=dtree_para['max_leaf_nodes'],
            max_features=dtree_para['max_features'],
            criterion='entropy'
    )
    dtree.fit(X=Xtrain, y=ytrain)
    y = dtree.predict(X=Xtest)
    #print(1 - dtree.score(X=Xtest, y=ytest))
    return y

# Finished
def knn(train, test):
    y = []
    Xtrain, ytrain, Xtest, ytest = load_data(train, test)
    clf = KNeighborsClassifier(
        n_neighbors=knn_para[2],
        weights='distance'
    )
    clf.fit(X=Xtrain, y=ytrain)
    y = clf.predict(X=Xtest)
    return y

def svm(train, test):
    y = []
    Xtrain, ytrain, Xtest, ytest = load_data(train, test)
    clf = SVC(
        kernel='rbf',
        C=1,
        gamma=0.02
    )
    clf.fit(X=Xtrain, y=ytrain);
    y = clf.predict(X=Xtest)
    #print(1 - clf.score(X=Xtest,y=ytest));
    return y

def pca_knn(train, test):
    y = []
    Xtrain, ytrain, Xtest, ytest = load_data(train, test)
    dim_red = RandomizedPCA(n_components=43)
    dim_red.fit(Xtrain)
    Rtrain = dim_red.transform(Xtrain)
    Rtest = dim_red.transform(Xtest)
    clf = KNeighborsClassifier(
        n_neighbors=knn_para[2],
        weights='distance'
    )
    clf.fit(X=Rtrain, y=ytrain)
    y = clf.predict(X=Rtest)
    #print(1 - clf.score(X=Rtest, y=ytest))
    return y

def pca_svm(train, test):
    y = []
    Xtrain, ytrain, Xtest, ytest = load_data(train, test)
    dim_red = RandomizedPCA(n_components=50)
    dim_red.fit(Xtrain)
    Rtrain = dim_red.transform(Xtrain)
    Rtest = dim_red.transform(Xtest)
    clf = SVC(
        kernel='poly',
        C=1,
        gamma=0.02
    )
    clf.fit(X=Rtrain, y=ytrain);
    y = clf.predict(X=Rtest)
    #print(1 - clf.score(X=Rtest, y=ytest));
    return y

if __name__ == '__main__':
    model = sys.argv[1]
    train = sys.argv[2]
    test = sys.argv[3]

    if model == "dtree":
        print(decision_tree(train, test))
    elif model == "knn":
        print(knn(train, test))
    elif model == "svm":
        print(svm(train, test))
    elif model == "pcaknn":
        print(pca_knn(train, test))
    elif model == "pcasvm":
        print(pca_svm(train, test))
    else:
        print("Invalid method selected!")
