import numpy as np
import sklearn.datasets
import sklearn.neighbors
import matplotlib.pyplot as plt


def make_data(noise=0.2, outlier=1):
    prng = np.random.RandomState(0)
    n = 500

    x0 = np.array([0, 0])[None, :] + noise * prng.randn(n, 2)
    y0 = np.ones(n)
    x1 = np.array([1, 1])[None, :] + noise * prng.randn(n, 2)
    y1 = -1 * np.ones(n)

    x = np.concatenate([x0, x1])
    y = np.concatenate([y0, y1]).astype(np.int32)

    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(
        x, y, test_size=0.1, shuffle=True, random_state=0)
    xplot, yplot = xtrain, ytrain

    outlier = outlier * np.array([1, 1.75])[None, :]
    youtlier = np.array([-1])
    xtrain = np.concatenate([xtrain, outlier])
    ytrain = np.concatenate([ytrain, youtlier])
    return xtrain, xtest, ytrain, ytest, xplot, yplot


class LinearLeastSquares(object):
    def __init__(self):
        self.w = []

    def fit(self, x, y):
        # TODO find minimizer of least squares objective
        x_bias = np.c_[x, np.ones(len(x))]
        self.w = (np.linalg.inv(x_bias.T @ x_bias) @ x_bias.T @ y)

    def predict(self, xquery):
        # TODO implement prediction using linear score function
        y_predict = np.sign(np.c_[xquery, np.ones(len(xquery))] @ self.w)
        return y_predict


def color(y):
    if y == 1:
        return "red"
    else:
        return "blue"


def task1():
    # get data
    for outlier in [1, 2, 4, 8, 16]:
        # get data. xplot, yplot is same as xtrain, ytrain but without outlier
        xtrain, xtest, ytrain, ytest, xplot, yplot = make_data(outlier=outlier)

        # TODO visualize xtrain via scatterplot
        xytrain = np.c_[xtrain, ytrain]
        plt.scatter(xytrain[xytrain[:, 2] == 1][:, 0], xytrain[xytrain[:, 2] == 1][:, 1], color="red")
        plt.scatter(xytrain[xytrain[:, 2] == -1][:, 0], xytrain[xytrain[:, 2] == -1][:, 1], color="blue")
        pltTitle = "Training Dataset with outlier of " + str(outlier)
        plt.title(pltTitle)
        plt.show()
        lls = LinearLeastSquares()
        lls.fit(xtrain, ytrain)

        # TODO evaluate accuracy and decision boundary of LLS
        y_predict = lls.predict(xtest)
        accuracy = np.sum((y_predict * ytest) == 1) / len(ytest)
        print("The accuracy rate of", outlier, "outlier is: ", accuracy)
        x = np.linspace(-1.5, 2.5, 100)
        y = np.linspace(-1.0, 1.5, 100)
        xx, yy = np.meshgrid(x, y)
        z = lls.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=plt.cm.Paired)
        xytest = np.c_[xtest, y_predict]
        # plt.scatter(xtest[:, 0], xtest[:, 1])
        plt.scatter(xytest[xytest[:, 2] == 1][:, 0], xytest[xytest[:, 2] == 1][:, 1], color="red")
        plt.scatter(xytest[xytest[:, 2] == -1][:, 0], xytest[xytest[:, 2] == -1][:, 1], color="blue")
        plt.xlim(-1.5, 2.5)
        plt.ylim(-1.0, 1.5)
        pltTitle = "decision boundary and test data with outlier of " + str(outlier)
        plt.title(pltTitle)
        plt.show()


if __name__ == "__main__":
    task1()
