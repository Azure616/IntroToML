#!/usr/bin/python

import sys

import itertools
import numpy as np
import random
import matplotlib.pyplot as plt
#Your code here


# load data
def load_data(fileDj):
    data = []
    file = open(fileDj, "r")
    while True:
        raw = file.readline()
        if raw == '': break;
        entry = raw.split()
        #data.append([float(entry[0]), float(entry[1]), float(entry[2])])
        data.append([float(entry[i]) for i in range(len(entry))])
    return scale_data(data)

# scale data
def scale_data(array):
    nparray = np.asarray(array)
    X = nparray.transpose()
    for i in range(len(X)-1):
        X[i] -= np.min(X[i])
        X[i] /= np.max(X[i])
    X = X.transpose()
    return X

## K-means functions 

# Get randomized clusters
def getInitialCentroids(k):
    random.seed(10)
    init_centroids = []
    for i in range(k):
        init_centroids.append([random.uniform(0, 1), random.uniform(0, 1)])
    return np.asarray(init_centroids)

# Get distance between two points
def getDistance(pt1,pt2):
    return np.sqrt(np.power(pt1[0]-pt2[0], 2)+np.power(pt1[1]-pt2[1], 2))

# Divide points (X matrix) into clusters
def allocatePoints(X, centroids):
    k = len(centroids)
    clusters = x = [[] for i in range(k)]
    for point in X:
        min_dist = np.inf
        min_index = 0
        for index, centroid in enumerate(centroids):
            distance = getDistance(point, centroid)
            if distance <= min_dist:
                min_dist = distance
                min_index = index
        clusters[min_index].append(point)
    return np.asarray(clusters)

# Update centroids according to clusters
def updateCentroids(clusters):
    new_centroids = []
    for cluster in clusters:
        aver_x = np.average([point[0] for point in cluster])
        aver_y = np.average([point[1] for point in cluster])
        new_centroids.append([aver_x, aver_y])
    return new_centroids

# Visualize the result
def visualizeClusters(clusters):
    colors = itertools.cycle(["c", "r", "b", "g", "y", "k"])
    for cluster in clusters:
        plt.scatter([point[0] for point in cluster], [point[1] for point in cluster], c=next(colors))
    plt.show()

# k-means clustering
def kmeans(X, k, maxIter=1000):
    centroids = getInitialCentroids(k)
    iteration = 0
    new_centroids = []
    while iteration < maxIter:
        clusters = allocatePoints(X,centroids)
        #print("Cluster 0 has ", clusters[0], " | Cluster 1 has " ,clusters[1])
        new_centroids = updateCentroids(clusters)
        #print("Iteration {}, ".format(iteration), "new cetroids are: {} ".format(new_centroids))
        iteration += 1
    return new_centroids, clusters

# Plot k v.s. obj_func
def kneeFinding(X,kList):
    obj_func_vals = []
    for k in kList:
        centroids, clusters = kmeans(X, k, 500)
        obj_func = 0
        for index, centroid in enumerate(centroids):
            for point in clusters[index]:
                obj_func += np.power(centroid[0] - point[0], 2)+np.power(centroid[1] - point[1], 2)
        obj_func_vals.append(obj_func)
    plt.plot(kList, obj_func_vals)
    plt.xlabel("number of clusters")
    plt.ylabel("objective function")
    plt.show()
    return

# Calculate purity
def purity(dataset, clusters):
    k = len(clusters)
    purities = []
    for cluster in clusters:
        label_counts = [0, 0]
        for point in cluster:
            label_counts[int(point[2])-1] += 1
        count_sum = sum(label_counts)
        purities.append(max(label_counts[0], label_counts[1])/count_sum)
    return purities

## GMM functions 

#calculate the initial covariance matrix
#covType: diag, full

def getInitialsGMM(X, k, covType):
    if covType == 'full':
        dataArray = np.transpose(np.array([pt[0:-1] for pt in X]))
        #dataArray = np.asarray([pt[0:-1] for pt in X]).transpose()
        covMat = np.cov(dataArray)
    else:
        covMatList = []
        for i in range(len(X[0])-1):
            data = [pt[i] for pt in X]
            cov = np.asscalar(np.cov(data))
            covMatList.append(cov)
        covMat = np.diag(covMatList)

    # Get centroids, assign initial proportions
    #random.seed(10)
    init_centroids = []
    dimen = len(X[0])
    for i in range(k):
        centroid = []
        for j in range(dimen-1):
                centroid.append(random.uniform(0, 1))
        init_centroids.append(centroid)
    # Here cluster has a different representation, a cluster is a circle range instead of a set of points
    # So cluster = {mean, radius(var)}
    init_clusters = [init_centroids, [1/k]*k]
    #init_clusters = [[[0.3, 0.3], [0.6, 0.6]], [1/k]*k]
    #print(init_clusters)
    return init_clusters, covMat


def visualizeClustersGMM(X, clusters, labels):
    colors = itertools.cycle(["c", "r", "b", "g", "y", "k"])
    k = len(clusters)
    clusters_to_plot = [[] for i in range(k)]
    for index, label in enumerate(labels):
        clusters_to_plot[label].append([X[index][0], X[index][1]])
    for cluster_to_plot in clusters_to_plot:
        plt.scatter([point[0] for point in cluster_to_plot], [point[1] for point in cluster_to_plot], c=next(colors))
    plt.show()
    #Your code here


def calculate(i, j, means, cov, data):
    x = data[j]
    mean = means[i]
    xm = x[:-1] - mean
    a = np.exp(-.5 * np.dot(np.dot(xm, np.linalg.inv(cov)), xm))
    return 1 / (2 * np.pi * np.linalg.det(cov) ** 0.5) * a


def updateEStep(z, data, means, cov):  # CHANGED
    for i in range(0, 2):  # Dim
        for j in range(data.shape[0]):  # dado
            z[j, i] = calculate(i, j, means, cov, data)
    z = (z.T / z.sum(axis=1)).T
    return z


def updateMStep(means, z, data):
    mean_prev = means
    mean_prev = mean_prev * 100
    newmi = np.zeros_like(means)
    for i in range(2):
        for j in range(data.shape[0]):
            newmi[i] += z[j, i] * data[j][:-1]
        means[i] = newmi[i] / z[:, i].sum()
    mean_new = means
    mean_new = mean_new * 100
    diff = 0
    for i in range(data.shape[1]-1):
        diff += (mean_prev[0][i] - mean_new[0][i]) ** 2

    diff = diff * 1000
    return diff

def gmmCluster(X, k, covType, maxIter=1000):
    #initial clusters
    clusters, covMat = getInitialsGMM(X,k,covType)
    #print(clusters[0])
    labels = []
    n = len(X)
    k = len(clusters[0])
    EMatrix = np.zeros((n, k))
    i = 0

    while i < 100:
        EMatrix = updateEStep(z=EMatrix, data=np.asarray(X), means=clusters[0],cov=np.asarray(covMat))
        updateMStep(means=clusters[0], z=EMatrix, data=np.asarray(X))
        """
        # E-step: assign new proportion according to centroid positions
        EMatrix = updateEStep(X, covMat, clusters, k)
        # M-step: update centroid positions and porbabilities
        clusters = updateMStep(X, clusters, EMatrix)
        """
        i += 1
    # Then get the clustering of each data point using Ematrix
    k = len(clusters)
    labels = []
    for index1, proportions in enumerate(EMatrix):
        max_prop = -np.inf
        max_index = 0
        for index2, proportion in enumerate(proportions):
            if max_prop < proportion:
                max_prop = proportion
                max_index = index2
        labels.append(max_index)
    visualizeClustersGMM(X, clusters, labels)
    return labels, clusters


def purityGMM(X, clusters, labels):

    k = len(clusters)
    n = len(X)

    cluster1_true = [X[i][-1] for i in range(n) if labels[i] == 0]
    cluster2_true = [X[i][-1] for i in range(n) if labels[i] == 1]
    count_1 = [0, 0]
    for true_val in cluster1_true:
        if true_val == 1:
            count_1[0] += 1
        else:
            count_1[1] += 1
    purity_1 = max(count_1[0]/sum(count_1), count_1[1]/sum(count_1))
    count_2 = [0, 0]
    for true_val in cluster2_true:
        if true_val == 1:
            count_2[0] += 1
        else:
            count_2[1] += 1
    purity_2 = max(count_2[0] / sum(count_2), count_2[1] / sum(count_2))

    return [purity_1, purity_2]

def main():
    #######dataset path
    datadir = sys.argv[1]
    #datadir = "./data_sets_clustering"
    pathDataset1 = datadir+'/humanData.txt'
    pathDataset2 = datadir+'/audioData.txt'
    dataset1 = load_data(pathDataset1)
    dataset2 = load_data(pathDataset2)

    #Q4
    kneeFinding(dataset1,range(1,7))

    #Q5
    centroids, clusters = kmeans(dataset1, 2, maxIter=100)
    visualizeClusters(clusters)
    pur_metric = purity(dataset1,clusters)
    print(pur_metric)




    #Q7
    labels11, clustersGMM11 = gmmCluster(dataset1, 2, 'diag')

    #labels12, clustersGMM12 = gmmCluster(dataset1, 2, 'full')

    #Q8

    #labels21,clustersGMM21 = gmmCluster(dataset2, 2, 'diag')

    #Q9
    purities11 = purityGMM(dataset1, clustersGMM11, labels11)

    #purities12 = purityGMM(dataset1, clustersGMM12, labels12)

    #purities21 = purityGMM(dataset2, clustersGMM21, labels21)

if __name__ == "__main__":
    main()