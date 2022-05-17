#!/usr/bin/env python
# coding: utf-8
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm.auto import tqdm  # Not needed but very cool!
import torch
import torch.nn as nn


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
        plt.title("Projection from MNIST to 1&2-PC feature space(Red with label-" + str(i) + ")")
        plt.show()


# =================================================================Task-3================================================================


class MultilayerPerceptron(nn.Module):

    def __init__(self, size_hidden=100, size_out=10):
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, size_hidden)
        self.fc2 = nn.Linear(size_hidden, size_hidden)
        self.fc3 = nn.Linear(size_hidden, size_hidden)
        self.fc4 = nn.Linear(size_hidden, size_hidden)
        self.out_layer = nn.Linear(size_hidden, size_out)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)

        # Your Code here: The rest of the layers

        out = self.fc2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.relu(out)

        out = self.fc4(out)
        out = self.relu(out)

        out = self.out_layer(out)

        return out


# Pytorch modules keep track of all model parameters internally. Those will be
# e.g. the matrix and bias of the `Linear` operation we just implemented.

# To be able to feed the mnist vectors to out MultilayerPerceptron we first have to
# convert them to `torch.Tensor`s. To not have to do this everytime we want to
# do an operation on those vectors you can find a `torch.Dataset` version of
# the mnist vectors below. All it does is a simple casting operation.


class MnistVectors(torch.utils.data.Dataset):
    '''A Pytorch Dataset, which does the same data preparation as was done in
    the PCA exercise.'''

    def __init__(self, split='train'):
        super().__init__()

        mnist = datasets.MNIST('../data',
                               train=split == 'train',
                               download=True)

        ########################
        #### Your Code here ####
        # self.mnist_vectors, self.labels = your conversion function from task 2
        ########################
        self.mnist_vectors, self.labels = convert_mnist_to_vectors(mnist)

    def __getitem__(self, idx):
        '''Implements the ``[idx]`` method. Here we convert the numpy data to
        torch tensors.
        '''

        mvec = torch.tensor(self.mnist_vectors[idx]).float()
        label = torch.tensor(self.labels[idx]).long()

        return mvec, label

    def __len__(self):
        return len(self.labels)


# The following two functions are needed to track the progress of the training.
# One transforms the output of the MultilayerPerceptron into a scalar class label, the
# other uses that label to calculate the batch accuracy.

def batch_accuracy(prediction, label):
    acc = torch.count_nonzero(prediction == label) / prediction.shape[0]
    return torch.count_nonzero(prediction == label) / prediction.shape[0] * 100


def class_label(prediction):
    # prediction_labels = torch.argmax(prediction, dim=1)
    return torch.argmax(prediction, dim=1)


def train(use_gpu=True):  # if torch.cuda.is_available(), use gpu to speed up training

    # Here we instantiate our model. The weights of the model are automatically
    # initialized by pytorch
    P = MultilayerPerceptron()
    # print(P.parameters())

    TrainData = MnistVectors()
    TestData = MnistVectors('test')
    # Dataloaders allow us to load the data in batches. This allows us a better
    # estimate of the parameter updates when doing backprop.
    # We need two Dataloaders so that we can train on the train data split
    # and evaluate on the test datasplit.
    Dl = DataLoader(TrainData, batch_size=16, shuffle=True)
    testDl = DataLoader(TestData, batch_size=16, shuffle=False)

    # Use the Adam optimizer with learning rate 1e-4 and otherwise default value
    # Use the torch.optim.Adam to define a optimizer, use the parameters() method of MLP to pass in iterable parameters and define the leaning rate with 1e-4
    optimizer = torch.optim.Adam(P.parameters(), lr=1e-4)

    # Use the Cross Entropy loss from pytorch. Make sure your MultilayerPerceptron does not use any activation function on the output layer! (Do you know why?)
    criterion = nn.CrossEntropyLoss()  # define a loss function with Cross Entropy loss
    print(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    if use_gpu:
        P.cuda()
        criterion.cuda()

    for epoch in tqdm(range(5), desc='Epoch'):
        for step, [example, label] in enumerate(tqdm(Dl, desc='Batch')):
            if use_gpu:
                example = example.cuda()
                label = label.cuda()

            # The optimizer knows about all model parameters. These in turn
            # store their own gradients. When calling loss.backward() the newly
            # computed gradients are added on top of the existing ones. Thus
            # at before calculating new gradients we need to clear the old ones
            # using ther zero_grad() method.
            optimizer.zero_grad()

            prediction = P(example)

            loss = criterion(prediction, label)

            # Here pytorch applies backpropagation for us completely automatically!!! That is quite awesome!
            loss.backward()

            # The step method now adds the gradients onto the model parameters as specified by the optimizer and the learning rate.
            optimizer.step()

            # To keep track of what is happening print some outputs from time to time.
            if (step % 375) == 0:
                # Your code here
                print("In step-", step)
                acc = batch_accuracy(class_label(prediction), label)
                tqdm.write('Batch Accuracy: {}%, Loss: {}'.format(acc, loss))

        # Now validate on the whole test set
        accuracies = []
        for idx, [test_ex, test_l] in enumerate(tqdm(testDl, desc='Test')):
            if use_gpu:
                test_ex = test_ex.cuda()
                test_l = test_l.cuda()

            #########################
            #### Your Code here  ####
            #########################
            accuracies.append(batch_accuracy(class_label(P(test_ex)), test_l).cpu().numpy())

            # Using your batch_accuracy function, also print the mean accuracy
            # over the whole test split of the data.

        print('Validation Accuracy: {}%'.format(np.mean(accuracies)))

        # Now let's write out a checkpoint of the model, so that we can
        # reuse it:
        torch.save(P.state_dict(), 'perceptron_{}.ckpt'.format(step))

        # If you need to load the checkpoint instanciate your model and the
        # load the state dict from a checkpoint:
        # P = MultilayerPerceptron()
        # P.load_state_dict(torch.load(perceptron_3750.ckpt))
        # Make sure to use the latest checkpoint by entering the right number.

        ######################################
        ###### Add code for task 4 here ######
        ######################################


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
    # plot_projection(pcs, data)
    train()
