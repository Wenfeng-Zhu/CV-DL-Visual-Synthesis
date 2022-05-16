#!/usr/bin/env python
# coding: utf-8
import random
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm  # Not needed but very cool!


def load_data(train=True):
    mnist = datasets.MNIST('../data',
                           train=train,
                           download=True)
    return mnist


def plot_examples(data):
    #########################
    #### Your Code here  ####
    #########################
    # print(random.randint(0, 5996))
    train_numpy = data.data.numpy()
    # print(np.argmax(train_numpy, axis=0))
    print("The max of train dataset is:\n", train_numpy.max(axis=0))
    print("\nThe min of train dataset is:\n", train_numpy.min(axis=0))
    print("\nThe mean of train dataset is:\n", train_numpy.mean(axis=0))
    print("\nThe shape of train dataset is:\n", data.data.shape)
    print("\nThe dtype of train dataset is:\n", data.data.type())
    # Plot some examples and put their corresponding label on top as title.
    start = random.randint(0, 59989)
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(data.data[start + i].numpy(), cmap="gray")
        plt.title(data.targets[i + start])
    plt.show()


def convert_mnist_to_vectors(data):
    '''Converts the ``[28, 28]`` MNIST images to vectors of size ``[28*28]``.
       It outputs mnist_vectors as a array with the shape of [N, 784], where
       N is the number of images in data.
    '''

    mnist_vectors = []
    labels = []

    #########################
    #### Your Code here  ####
    #########################
    # image--PIL.Image.Image; label--int
    for image, label in tqdm(data):
        mnist_vectors.append(np.asarray(image).ravel())
        labels.append(label)

    # return as numpy arrays
    mnist_vectors = np.asarray(mnist_vectors)
    # print(mnist_vectors.mean(axis=1).shape)
    # np.subtract()
    # mnist_vectors = (mnist_vectors - mnist_vectors.mean(axis=0)[:, None]) / (mnist_vectors.max(
    #     axis=1) - mnist_vectors.min(axis=1))[:, None] * 2
    # mnist_vectors_max = mnist_vectors.max(axis=1)[:, None]
    # mnist_vectors_min = mnist_vectors.min(axis=1)[:, None]
    # mnist_vectors_mean = mnist_vectors.mean(axis=1)[:, None]
    # mnist_vectors_sub = mnist_vectors_max - mnist_vectors_min
    mnist_vectors_center = (mnist_vectors - mnist_vectors.mean(axis=1)[:, None]) / (
            mnist_vectors.max(axis=1)[:, None] - mnist_vectors.min(axis=1)[:, None])
    labels = np.asarray(labels)

    return mnist_vectors_center, labels


def do_pca(data):
    '''Returns matrix [784x784] whose columns are the sorted eigenvectors.
       Eigenvectors (prinicipal components) are sorted according to their
       eigenvalues in decreasing order.
    '''

    mnist_vectors, labels = convert_mnist_to_vectors(data)
    #     prepare_data(mnist_vectors)

    # compute covariance matrix of data with shape [784x784]
    cov = np.cov(mnist_vectors.T)
    # print("covariance matrix",cov[390])

    # compute eigenvalues and vectors
    eigVals, eigVec = np.linalg.eig(cov)

    # sort eigenVectors by eigenValues
    sorted_index = eigVals.argsort()[::-1]
    eigVals = eigVals[sorted_index]
    sorted_eigenVectors = eigVec[:, sorted_index]
    print(type(sorted_eigenVectors), sorted_eigenVectors.shape)
    sorted_eigenVectors_real = sorted_eigenVectors.real.astype(float).T
    return sorted_eigenVectors.real.astype(float).T


def plot_pcs(sorted_eigenVectors, num=10):
    '''Plots the first ``num`` eigenVectors as images.'''

    #########################
    #### Your Code here  ####
    #########################
    pc_10 = np.empty((10, 28, 28))
    for i in range(num):
        pc_10[i] = sorted_eigenVectors[i].reshape((28, 28))
        plt.subplot(2, 5, i + 1)
        plt.imshow(sorted_eigenVectors[i].reshape((28, 28)), cmap="gray")
        plt.title("PC-" + str(i + 1))
    plt.show()


def plot_projection(sorted_eigenVectors, data):
    '''Projects ``data`` onto the first two ``sorted_eigenVectors`` and makes
    a scatterplot of the resulting points'''

    #########################
    #### Your Code here  ####
    #########################
    mnist_vectors, labels = convert_mnist_to_vectors(data)
    pc_1 = mnist_vectors @ sorted_eigenVectors[0]
    pc_2 = mnist_vectors @ sorted_eigenVectors[1]
    for i in range(10):
        indices = np.argwhere(labels == i).ravel()
        # print(indices.shape, pc_1.shape)
        plt.scatter(pc_1, pc_2)
        plt.scatter(pc_1[indices], pc_2[indices], color="red")
        plt.xlabel("PC-1")
        plt.ylabel("PC-2")
        plt.title("Projection from MNIST to 1&2-PC feature space(Red with label-" + str(i)+")")
        plt.show()


if __name__ == '__main__':
    # You can run this part of the code from the terminal using python ex1.py
    # dataloading
    data = load_data()

    # subtask 1
    # plot_examples(data)

    # # subtask 2
    # mnist_vectors, labels = convert_mnist_to_vectors(data)
    # #Comment in once the above function is implemented, to check the shape of your dataset
    # print('Data shape', mnist_vectors)
    #
    #
    # # subtask 3
    pcs = do_pca(data)
    #
    # # subtask 3
    # plot_pcs(pcs)
    #
    # # subtask 4
    plot_projection(pcs, data)
