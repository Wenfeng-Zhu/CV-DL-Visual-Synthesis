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
from PIL import Image
from sklearn.decomposition import PCA

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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


def load_data(transform_train, transform_test):
    trainset_base = torchvision.datasets.CIFAR10(root='../data', train=True, download=True,
                                                 transform=transform_train)
    trainloader_base = torch.utils.data.DataLoader(trainset_base, batch_size=16, shuffle=True)

    testset_base = torchvision.datasets.CIFAR10(root='../data', train=False, download=True,
                                                transform=transform_test)
    testloader_base = torch.utils.data.DataLoader(testset_base, batch_size=16, shuffle=False)

    return trainloader_base, testloader_base


def class_label(prediction):
    _, predicted_class = torch.max(prediction, 1)
    return predicted_class


def train_test(trainloader_base, testloader_base, model_name):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = Net()
    net.to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    criterion.to(device)

    # train model
    n_epochs = 10
    best_acc = 0
    # model_name = 'base'
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
    return acc_per_epoch


def train_base():
    trainloader_base, testloader_base = load_data(transforms.ToTensor(), transforms.ToTensor())
    acc_per_epoch = train_test(trainloader_base, testloader_base, 'base')
    epochs = np.array(range(1, 11))
    plt.title("Accuracies for each epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracies(%)")
    plt.plot(epochs, acc_per_epoch)
    plt.scatter(epochs, acc_per_epoch)
    plt.show()


def show_transformed():
    transform_train = transforms.Compose([
        # extra augmentations
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        # normalization
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_val = transforms.Compose([
        # extra aumentations
        transforms.GaussianBlur(kernel_size=3, sigma=0.2),
        transforms.ToTensor(),
        # normalization
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform_val)
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

    print("Show some comparison images:")
    against_img(100)
    against_img(1000)
    against_img(500)
    against_img(400)
    pass


def train_normalize_transformed():
    transform_train = transforms.Compose([
        # extra augmentations
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32, padding=4),
        transforms.ToTensor(),
        # normalization
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transform_val = transforms.Compose([
        # extra aumentations
        transforms.GaussianBlur(kernel_size=3, sigma=0.2),
        transforms.ToTensor(),
        # normalization
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    trainloader, testloader = load_data(transform_train, transform_val)
    train_test(trainloader, testloader, 'transformed')


def compare_acc():
    acc_per_epoch_base = np.load('./logging/base.npy')
    acc_per_epoch_transformed = np.load('./logging/transformed.npy')
    epochs = np.array(range(1, 11))
    plt.title("Accuracies for each epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracies(%)")
    plt.plot(epochs, acc_per_epoch_base, color='red', label='base')
    plt.plot(epochs, acc_per_epoch_transformed, color='blue', label='transformed')
    plt.scatter(epochs, acc_per_epoch_base, color='red')
    plt.scatter(epochs, acc_per_epoch_transformed, color='blue')
    plt.legend()
    plt.show()


def show_tensor(tensor, title):
    if tensor.ndim == 4:
        temp = tensor.squeeze()
    else:
        temp = tensor
    plt.imshow(temp.cpu().permute(1, 2, 0))
    plt.title(title)
    plt.show()


def task_2():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    resNet = torchvision.models.resnet18(pretrained=True).eval().to(device)
    print(resNet)
    image = Image.open('pug.jpg')
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    X = transform(image).unsqueeze(dim=0).to(device)

    show_tensor(X.squeeze(), "The transformed image")
    activation_maps = []

    # define a hook function
    def hook(module, input, output):
        activation_maps.append(output)

    # register the forward hook in the layer1 conv1 and conv2
    resNet.layer1[0].conv1.register_forward_hook(hook)
    resNet.layer1[0].conv2.register_forward_hook(hook)
    output = resNet(X)

    # print(resNet.layer1)
    def activation_map_pca(map, title):
        map1 = map.permute(0, 2, 3, 1).view(-1, 64).cpu().detach().numpy()
        pca = PCA(n_components=3)
        map1_reduce = torch.tensor(pca.fit_transform(map1)).view(1, 56, 56, 3).permute(0, 3, 1, 2)
        show_tensor(map1_reduce, title)

    activation_map_pca(activation_maps[0], "The activation map of conv1 in Layer1")
    activation_map_pca(activation_maps[1], "The activation map of conv2 in Layer1")
    pass


def preprocess(img, size=224):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[None]),  # equivalent to unsqueeze()
    ])
    return transform(img)


def saliency_maps(X, y, model):
    """
    Compute a class saliency map using the model for images X and labels y.

    Input:
    - X: Input images; Tensor of shape (N, 3, H, W)
    - y: Labels for X; LongTensor of shape (N,)
    - model: A pretrained CNN that will be used to compute the saliency map.

    Returns:
    - saliency: A Tensor of shape (N, H, W) giving the saliency maps for the input
    images.
    """
    # Make sure the model is in "test" mode
    model.eval()

    # make a copy of the tensor and set X_var to require gradient for later visualization
    X_var = X.clone().detach()
    X_var.requires_grad = True
    y_var = y.clone().detach()
    saliency = None
    ############ your code here ############

    ##############################################################################
    # Implement this function. Perform a forward and backward pass through       #
    # the model to compute the gradient of the correct class score with respect  #
    # to each input image. You first want to compute the loss over the correct   #
    # scores, and then compute the gradients with a backward pass.               #
    ##############################################################################
    output = model(X_var)
    score, _ = torch.max(output, dim=1)
    criterion = nn.CrossEntropyLoss()
    loss = criterion(output, y_var)
    loss.backward()
    saliency, _ = torch.max(torch.abs(X_var.grad), dim=1)
    ############ end of your code############
    return saliency


def show_saliency_maps():
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    names = ['catdog_243.png', 'catdog_243.png', 'snake_56.png', 'spider_72.png']

    X = [np.array(Image.open(name).convert('RGB')) for name in names]
    y = [int(s.rsplit('_')[1].rsplit('.')[0]) for s in names]
    # intentionally change the label to a wrong one
    y[1] = 285
    model = torchvision.models.resnet18(pretrained=True).eval().to(device)
    # Convert X and y from numpy arrays to Torch Tensors
    X_tensor = torch.cat([preprocess(Image.fromarray(x)) for x in X], dim=0).to(device)
    y_tensor = torch.LongTensor(y).to(device)
    # Compute saliency maps for images in X
    saliency = saliency_maps(X_tensor, y_tensor, model)

    # Convert the saliency map from Torch Tensor to numpy array and show images
    # and saliency maps together.
    saliency = saliency.cpu().numpy()
    N = len(X)
    for i in range(N):
        plt.subplot(2, N, i + 1)
        plt.imshow(X[i])
        plt.axis('off')
        plt.title('ground truth label set to {}'.format(y[i]))
        plt.subplot(2, N, N + i + 1)
        plt.imshow(saliency[i], cmap=plt.cm.hot)
        plt.axis('off')
        plt.gcf().set_size_inches(20, 10)
    plt.show()
    pass


def gather_example():
    N, C = 4, 5
    s = torch.randn(N, C)
    y = torch.LongTensor([1, 2, 1, 3])
    print(s)
    print(y)
    print(s.gather(1, y.view(-1, 1)).squeeze())


if __name__ == "__main__":
    # # task-1.1
    # train_base()
    # # task_1.2
    # show_transformed()
    # # task_1.3
    # train_normalize_transformed()
    # compare_acc()
    # task_2()
    # gather_example()
    show_saliency_maps()
    pass
