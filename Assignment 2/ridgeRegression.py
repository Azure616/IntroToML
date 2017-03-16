import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def loadDataSet(data):
    f = open(data)
    x_val = []
    y_val = []
    while True:
        entry = f.readline()
        if entry == '': break
        x_0, x_1, x_2, y = entry.split()
        x_val.append((float(x_0), float(x_1), float(x_2)))
        y_val.append(float(y))
    x = np.array(x_val)
    y = np.array(y_val).reshape(200, 1)
    # x_1 = np.asarray([row[1] for row in x]).reshape(200,1)
    # x_2 = np.asarray([row[2] for row in x]).reshape(200,1)
    # plotting
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.scatter(x_1, x_2, y)
    # plt.show()
    return x, y

def ridgeRegress(xVal, yVal, lmbda):
    theta = np.asarray([[1], [1], [1]])
    x = xVal
    y = yVal.reshape((yVal.size, 1))
    xTran = x.transpose()
    reg = [[lmbda, 0, 0],
           [0, lmbda, 0],
           [0, 0, lmbda]]
    temp = np.dot(xTran, x) + reg
    inverse = np.linalg.inv(temp)
    beta = np.dot(np.dot(inverse, xTran), y)
    return beta

def ridgeRegressWithGD(xVal, yVal, lmbda):
    beta = np.asarray([[-40], [-40], [-40]])
    alpha = 0.0005
    x = xVal
    y = yVal.reshape((yVal.size, 1))
    xTran = x.transpose()
    for i in range(0, 10000):
        loss = np.dot(x, beta) - y
        gradient = np.dot(xTran, loss)
        lmbda_reg = [[0, 0, 0],
                     [0, lmbda, 0],
                     [0, 0, lmbda]]  # Do not regulate for bias
        regul_term = np.dot(lmbda_reg, beta)
        beta = beta - alpha * (gradient + regul_term)
    diffs = np.dot(x, beta) - y
    sum = 0
    return beta

def cv(xVal, yVal):
    x = xVal
    y = yVal
    all_lmbda = np.arange(0.02, 1.02, 0.02)
    rand = list(range(200))
    mses = []
    np.random.seed(7)
    np.random.shuffle(rand)
    for lmbda in all_lmbda:
        errors = []
        for j in range(0, 9):
            train_index = [rand[i] for i in rand[:j * 20] + rand[(j + 1) * 20:]]
            test_index = [rand[i] for i in range(j * 20, (j + 1) * 20)]
            x_train = np.asarray([x[i] for i in train_index])
            y_train = np.asarray([y[i] for i in train_index])
            x_test = np.asarray([x[i] for i in test_index])
            y_test = np.asarray([y[i] for i in test_index])
            beta = ridgeRegress(x_train, y_train, lmbda)
            diffs = y_test - np.dot(x_test, beta)
            diffs2 = diffs.transpose()
            error = np.dot(diffs2, diffs)
            errors.append(error / 20)
        mse = np.average(errors)
        # print(mse)
        mses.append((lmbda, mse))
    temp = list(mses)
    temp.sort(key=lambda a: a[1])
    best_lmbda = temp[0][0]
    print(best_lmbda)
    plt.plot([ele[0] for ele in mses], [ele[1] for ele in mses], "-")
    plt.xlabel("lambda")
    plt.ylabel("MSE")
    plt.show()
    return best_lmbda

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

# Main area, for plotting and testing
x, y = loadDataSet("./RRdata.txt")
# Q 2.5: Compare the performance of ridge regression with:
# lambda = 0
# lambda = best lambda
# beta = [[3], [1], [1]]
best_lmbda = cv(x, y)
beta_1 = ridgeRegress(x, y, 0)
beta_2 = ridgeRegress(x, y, best_lmbda)
beta_3 = np.asarray([[3], [1], [1]])
print(beta_1)
print(beta_2)
print(beta_3)
x_1_ran, x_2_ran = np.meshgrid(np.arange(-5, 5, 0.05), np.arange(-5, 5, 0.05))
x_1 = np.asarray([[1, row[1]] for row in x]).reshape((200,2))
x_2 = np.asarray([row[2] for row in x]).reshape((200,1));
theta = standRegres(x_1, x_2)
print(theta)
x_range = np.arange(-5,5, 0.05)
f_x = theta[0]*1 + theta[1]*x_range

plt.plot([row[1] for row in x], [row[2] for row in x], ".")
plt.plot(x_range, f_x, "-")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

#pred_1 = 1*beta_1[0] + x_1_ran*beta_1[1] + x_2_ran*beta_1[2]
#pred_2 = 1*beta_2[0] + x_1_ran*beta_2[1] + x_2_ran*beta_2[2]
#pred_3 = 1*beta_3[0] + x_1_ran*beta_3[1] + x_2_ran*beta_3[2]
#fig = plt.figure()
#ax = fig.gca(projection='3d')
#ax.scatter(x_1, x_2, y)
#ax.plot_surface(x_1_ran, x_2_ran, pred_1, rstride=8, cstride=8, alpha=0.3, color = "b", linewidth = 0)
#ax.plot_surface(x_1_ran, x_2_ran, pred_2, rstride=8, cstride=8, alpha=0.3, color = "r", linewidth = 0)
#ax.plot_surface(x_1_ran, x_2_ran, pred_3, rstride=8, cstride=8, alpha=0.3, color = "g", linewidth = 0)
#ax.set_xlabel("x1")
#ax.set_ylabel("x2")
#ax.set_zlabel("y")
#plt.xlabel("x1")
#plt.ylabel("x2")


