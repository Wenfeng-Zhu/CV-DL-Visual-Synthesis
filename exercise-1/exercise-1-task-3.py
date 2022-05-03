import torch
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt


def numpy_to_torch():
    img = image.imread('dog.jpg')
    plt.imshow(img)
    plt.title("Original image")
    plt.show()
    img_torch = torch.from_numpy(img)
    img_chw = img_torch.permute(2, 0, 1)
    img_swap_back = img_chw.permute(1, 2, 0)
    img_np = img_swap_back.numpy()
    plt.imshow(img_np)
    plt.title("Double Converted image")
    plt.show()


x = np.random.rand(5, 5, 1)
w = np.random.rand(2, 2, 1)
conv = torch.nn.Conv2d(1, 1, 2)
# print(w)
# print(conv.weight)
x_torch = torch.from_numpy(x).permute(2, 0, 1)
w_torch = torch.from_numpy(w).permute(2, 0, 1)
x_torch_add = x_torch[None, :]
w_torch_add = w_torch[None, :]
# Set the weight of conv with w
conv.weight = torch.nn.Parameter(w_torch_add)
output = conv(x_torch_add)
print(output.shape)
# print(conv.weight)
