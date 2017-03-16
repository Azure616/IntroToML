import numpy as np
from sklearn.preprocessing import MinMaxScaler

#E-step
def updateEStep(X, covMat, clusters, k):
    # k = number of possible j
    # len(X) = number of possible i
    n = len(X)
    #EMatrix = np.zeroes((k, n))
    EMatrix = []
    centroids = np.asarray(clusters[0])
    probs = np.asarray(clusters[1])
    covMatrix = np.asarray(covMat)
    #print(covMat)
    for point in X:
        ijs = []
        for index, centroid in enumerate(centroids):
            num_of_features = len(centroid)
            vec = np.asarray(point[:num_of_features])
            diff = vec - centroid
            inverse = np.linalg.inv(covMatrix)
            #pow = float((-1/2)*np.dot(np.dot(diff.transpose(), inverse), diff))
            pow = (-1/2)*diff.transpose().dot(inverse).dot(diff)

            ##
            const = 1/np.sqrt(np.power(2*np.pi,num_of_features)*np.linalg.det(covMatrix))

            ##
            ijs.append(const*np.exp(pow)*probs[index])
        sum_ijs = np.sum(ijs)
        #print(sum_ijs)
        EMatrix.append([ij/sum_ijs for ij in ijs])
    #print(EMatrix)
    return EMatrix

#M-step
def updateMStep(X,clusters,EMatrix):
    # Get new cluster centers and new P (mu = mu_j)
    k = len(clusters) # number of clusters
    n = len(X)        # number of points
    dimen = len(clusters[0][0]) # number of dimensions
    new_centroids = []
    new_probs = []
    EMatrix_trans = np.asarray(EMatrix).transpose()

    # For each point, their proportions to different clusters
    for proportions in EMatrix_trans:
        sum_zij = np.sum(proportions)
        new_probs.append(sum_zij/n)
        new_centroid = np.asarray([0.0] * dimen)
        for index, proportion in enumerate(proportions):
            vec = np.asarray(X[index][:dimen])
            #print((proportion*vec/sum_zij).shape)
            new_centroid += proportion*vec/sum_zij
        new_centroids.append(new_centroid)
    clusters = [new_centroids, new_probs]
    print(new_probs)
    return clusters