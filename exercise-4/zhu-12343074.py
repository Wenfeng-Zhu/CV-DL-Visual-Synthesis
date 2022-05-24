import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def task_1():
    input_img = torch.tensor([[[1., 0., 0., 1., 1.],
                               [0., 1., 0., 0., 0.],
                               [0., 0., 0., 1., 0.],
                               [1., 0., 0., 0., 1.],
                               [1., 0., 0., 0., 0.]]])
    w = torch.tensor([[[[1., 0.], [0., 1.]]],
                      [[[1., 0.], [1., 0.]]],
                      [[[1., 1.], [0., 0.]]]])
    bias = torch.tensor([-1., -1., -1.])
    conv = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(2, 2), bias=True, )
    with torch.no_grad():
        conv.weight = nn.Parameter(w)
        conv.bias = nn.Parameter(bias)
        print("weight:", conv.weight.size(), "bias:", conv.bias.size())

    output = F.relu(conv(input_img))
    output_img = output.permute(1, 2, 0).detach().numpy()
    plt.imshow(output_img)
    plt.title("output image of stride=1 and padding=0")
    plt.show()

    conv.stride = 2
    output = F.relu(conv(input_img))
    output_img = output.permute(1, 2, 0).detach().numpy()
    plt.title("output image of stride=2 and padding=0")
    plt.imshow(output_img)
    plt.show()


class ConvolutionalNetwork(nn.Module):
    def __init__(self, normalization=False):
        super().__init__()  # spatial dimension
        self.normalization = normalization  # input   output
        # convolution block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=0,
                               stride=1)  # 28x28  ->  24x24
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 24x24  ->  12x12
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), padding=0, stride=1)  # 12x12  ->  8x8
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1, stride=1)  # 8x8 -> 8x8
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8x8 -> 4x4
        # fully connected block

        self.fc1 = nn.Linear(in_features=32 * 16, out_features=120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.rule = nn.ReLU()

    def forward(self, x):
        # Implement the forward pass and use the layers in the order they were defined in the init method.
        # Use Relu as the activation function
        # Hint: you can use tensor.view() to reshape a tensor in pytorch
        #########################
        #### Your Code here  ####
        #########################
        # conv1 layer
        out = self.rule(self.conv1(x))
        # pool layer
        out = self.pool(out)
        # conv2 layer
        out = self.rule(self.conv2(out))
        # conv3 layer
        out = self.rule(self.conv3(out))
        # pool2 layer
        out = self.pool2(out)

        # reshape for the first fully connected layer
        out = out.view(16, 1, -1)
        # fc1 layer
        out = self.rule(self.fc1(out))
        # fc2 layer
        out = self.rule(self.fc2(out))
        # fc3 layer
        out = self.fc3(out)
        out = out.view(16, 10)

        return out


def get_data():
    TrainData = datasets.MNIST(root="../data", train=True, download=True, transform=transforms.ToTensor())
    TestData = datasets.MNIST(root="../data", train=False, download=True, transform=transforms.ToTensor())
    train_dataloader = DataLoader(TrainData, batch_size=16, shuffle=True)
    test_dataloader = DataLoader(TestData, batch_size=16, shuffle=False)
    return train_dataloader, test_dataloader


def class_label(prediction):
    _, predicted_class = torch.max(prediction, 1)
    return predicted_class


def train_and_eval(n_epochs, normalization=False,
                   use_gpu=False):  # if torch.cuda.is_available(), use gpu to speed up training
    ConvNet = ConvolutionalNetwork(normalization=normalization)
    Dl, testDl = get_data()

    # Using Adam optimizer with learning rate 1e-4 and otherwise default
    optimizer = optim.Adam(ConvNet.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    if use_gpu:
        ConvNet.cuda()
        criterion.cuda()

    for epoch in range(n_epochs):
        ConvNet.train()  # train mode (matters for batchnorm layer, dropout, ...)
        for step, [example, label] in enumerate(tqdm(Dl, desc='Batch')):
            if use_gpu:
                example, label = example.cuda(), label.cuda()

            # reset gradient
            optimizer.zero_grad()
            prediction = ConvNet(example)  # todo change name
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()

        # Now validate on the whole test set
        correct = 0
        total = 0
        for idx, [test_example, test_label] in enumerate(testDl):
            if use_gpu:
                test_example, test_label = test_example.cuda(), test_label.cuda()

            ConvNet.eval()  # eval mode (matters for batchnorm layer, dropout, ...)
            with torch.no_grad():

                test_prediction = ConvNet(test_example)
                predicted_label = class_label(test_prediction)  # todo change name
                correct += (predicted_label == test_label).sum()
                total += test_label.size(0)

        accuracy = correct / total
        print("\n *** Summary: Epoch [{} / {}]  Test Accuracy: {}***".format(epoch + 1, n_epochs, accuracy))
        torch.save(ConvNet.state_dict(), 'ConvNet{}.ckpt'.format(epoch + 1))


def task_3():
    train_and_eval(n_epochs=5, normalization=False, use_gpu=True)
    print("\nThe result after normalization:\n")
    # train_and_eval(n_epochs=5, normalization=True, use_gpu=True)
    pass


def min_max_normalize(tensor):
    #########################
    #### Your Code here  ####
    #########################
    d = tensor.shape[0]
    c = tensor.shape[1]
    h = tensor.shape[2]
    w = tensor.shape[3]
    tensor = tensor.view(d, -1)
    max = torch.max(tensor, 1).values.view(1, -1).t()
    min = torch.min(tensor, 1).values.view(1, -1).t()
    tensor = (tensor - min) / (max - min)
    tensor = tensor.view(d, c, h, w)
    return tensor


def plot_list_to_grid(list_of_images, nr, nc):
    fig = plt.figure(figsize=(nr, nc))
    grid = ImageGrid(fig, 111, nrows_ncols=(nr, nc), axes_pad=0.1)
    for ax, im in zip(grid, list_of_images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im, cmap='gray')
    plt.show()
    plt.close()


def task_4():
    model = torch.load("ConvNet4.ckpt")
    conv_filter1 = model['conv1.weight']
    # conv_filter1 = conv_filter1.permute(0, 2, 3, 1).cpu().numpy()
    our_filters = conv_filter1[:4]
    # our_filters = our_filters.view(4, 1, -1)
    our_filters = min_max_normalize(our_filters).view(4, 1, 5, 5).permute(0, 2, 3, 1).cpu().numpy()
    print("The convolution filters of the trained ConvNet's first layer")
    plot_list_to_grid(our_filters, 2, 2)

    alexnet = torchvision.models.alexnet(pretrained=True)
    alexnet_filters = alexnet.features[0].weight.data
    alexnet_filters = min_max_normalize(alexnet_filters).permute(0, 2, 3, 1).cpu().numpy()
    print("The convolution filters of the trained AlexNet's first layer")
    plot_list_to_grid(alexnet_filters, 8, 8)
    pass


if __name__ == "__main__":
    # task_1()
    # task_3()
    task_4()
