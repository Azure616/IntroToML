import numpy as np
import matplotlib.pyplot as plt

# Dataset loading
def loadDataSet(data):
    f = open(data)
    x_val = []
    y_val = []
    i = 0
    while True:
        entry = f.readline()
        if entry == '': break
        x_0,x_1,y = entry.split("\t")
        x_val.append((float(x_0),float(x_1)))
        y_val.append(float(y))
        i += 1
    x = np.array(x_val)
    y = np.array(y_val).transpose()
    # Plot component disabled to speed up performance of the code
    # x_1 = [row[1] for row in x_val]
    # plt.scatter(x_1, y_val, alpha=0.5)
    # plt.show()
    return x, y

# Standard, simple linear regression
def standRegres(xVal, yVal):
    x = xVal
    x_t = x.transpose()
    y = yVal
    mul = np.dot(x_t, x)
    inv = np.linalg.inv(mul)
    theta = np.dot(np.dot(inv,x_t), y)
    # Plot component disabled to speed up performance of the code
    # x_1 = [row[1] for row in x]
    # x_range = np.arange(0,5, 0.005)
    # x_plot = [[1, i] for i in x_range]
    # f_x = np.dot(x_plot, theta)
    # plt.plot(x_1, y, ".")
    # plt.plot(x_plot, f_x, "-")
    # plt.show()
    return theta

# Polynomial regression in Stochastic Gradient Descent
def polyRegres(xVal, yVal):
    theta = np.asarray([[-1],[-1],[-1]])
    alpha = 0.005
    x = xVal
    y = yVal.reshape((200,1))
    x_sq = [[pow(x_1, 2)] for (_, x_1) in x]
    x = np.concatenate((x, x_sq), axis=1)
    xTran = x.transpose()
    for i in range(0, 1000):
        loss = np.dot(x, theta) - y
        gradient = np.dot(xTran, loss)
        theta = theta - alpha * gradient
    # Plot component disabled to speed up performance of the code
    # x_range = np.arange(0,5, 0.005)
    # x_plot = [[1, i, pow(i,2)] for i in x_range]
    # f_x = np.dot(x_plot, theta)
    # x_1 = [a[1] for a in x]
    # plt.plot(x_1, y, ".")
    # plt.plot(x_range, f_x,"-")
    # plt.show()
    return theta

# Main() area, used for testing
x, y = loadDataSet("./Q2data.txt")
#standRegres(x, y)
print(polyRegres(x, y))