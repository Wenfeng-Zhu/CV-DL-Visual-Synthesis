import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os


# ---------------Task 1.1 Define and train your network---------------#
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # convolution block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5), padding=0,
                               stride=1)  # 32x32  ->  28x28
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28  ->  14x14
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), padding=0, stride=1)  # 14x14  ->  10x10
        # self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1, stride=1)  # 8x8 -> 8x8
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 10x10 -> 5x5
        # fully connected block
        self.fc1 = nn.Linear(in_features=16 * 25, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        # activation function
        self.rule = nn.ReLU()

    def forward(self, x):
        # conv1 layer
        out = self.rule(self.conv1(x))
        # pool layer
        out = self.pool(out)
        # conv2 layer
        out = self.rule(self.conv2(out))
        # pool2 layer
        out = self.pool2(out)
        # reshape for the first fully connected layer
        out = out.view(16, -1)
        # fc1 layer
        out = self.rule(self.fc1(out))
        # fc2 layer
        out = self.rule(self.fc2(out))
        # fc3 layer
        out = self.fc3(out)
        out = out.view(16, 10)
        return out


def task_1_1():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def class_label(prediction):
        _, predicted_class = torch.max(prediction, 1)
        return predicted_class

    # load data ~200MB for CIFAR10 dataset
    trainset_base = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,
                                                 transform=transforms.ToTensor())
    trainloader_base = torch.utils.data.DataLoader(trainset_base, batch_size=16, shuffle=True)

    testset_base = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,
                                                transform=transforms.ToTensor())
    testloader_base = torch.utils.data.DataLoader(testset_base, batch_size=16, shuffle=False)

    # choose network
    net = Net()
    net.to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    criterion.to(device)

    # train model
    n_epochs = 10
    best_acc = 0
    model_name = 'base'
    # save the accuracies for each epoch in this list
    acc_per_epoch = []
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        net.train()
        for step, [example, label] in enumerate(tqdm(trainloader_base, desc='Batch')):
            example, label = example.to(device), label.to(device)
            # reset gradient
            optimizer.zero_grad()
            prediction = net(example)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()

        # evaluate
        running_loss = 0.0
        if epoch % 1 == 0:
            correct = 0
            total = 0
            print('\n *** Summary: Epoch [{} / {}] ***'.format(epoch, n_epochs))
            running_loss += loss.item()
            print('loss: {}'.format(running_loss / len(trainloader_base.dataset)))

            for idx, [test_example, test_label] in enumerate(testloader_base):
                test_example, test_label = test_example.to(device), test_label.to(device)
                net.eval()  # eval mode (matters for batchnorm layer, dropout, ...)
                with torch.no_grad():
                    test_prediction = net(test_example)
                    predicted_label = class_label(test_prediction)
                    correct += (predicted_label == test_label).sum()
                    total += test_label.size(0)

            accuracy = correct.item() / total
            acc_per_epoch.append(accuracy * 100)
            print("\n *** Summary: Epoch [{} / {}]  Test Accuracy: {}% ***".format(epoch + 1, n_epochs, accuracy * 100))

    path_save = './logging/{}'.format(model_name)
    np.save(path_save + '.npy', acc_per_epoch)
    epochs = np.array(range(1, 11))
    plt.title("Accuracies for each epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracies(%)")
    plt.plot(epochs, acc_per_epoch)
    plt.scatter(epochs, acc_per_epoch)
    plt.show()


def task_1_2():
    transform_train = transforms.Compose([
        # extra aumentations
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor()
        # normalization
        # transforms.Normalize()
    ])
    transform_val = transforms.Compose([
        # extra aumentations
        transforms.GaussianBlur(kernel_size=3, sigma=0.2),
        transforms.ToTensor(),
        # normalization
        ############ your code here ############
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

    def against_img(index):
        img_original = trainset.data[index]
        img_transform = trainset.__getitem__(index)[0].permute(1, 2, 0)
        fig = plt.figure()
        f1 = fig.add_subplot(121)
        f2 = fig.add_subplot(122)
        f1.imshow(img_original)
        f1.title.set_text("Original Image")
        f2.imshow(img_transform)
        f2.title.set_text("Transformed Image")
        plt.show()

    against_img(100)
    against_img(1000)
    against_img(500)
    against_img(400)
    pass


if __name__ == "__main__":
    # task_1_1()
    task_1_2()
    # task_3()
    # task_4()
