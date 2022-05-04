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


def convolution_operation(imgGray, conv_filter):
    height, width = imgGray.shape[0], imgGray.shape[1]
    size = len(conv_filter)
    # create a new numpy array with padding to make sure the filtered image has the same size as the input image
    # It must be noted, however, that filters of odd size will facilitate the calculation
    # img_padding = np.zeros((height + size - 1, width + size - 1))
    # img_padding[((size - 1) // 2):((size - 1) // 2 + height), ((size - 1) // 2):((size - 1) // 2 + width)] = imgGray
    img_conv = np.zeros((height - size + 1, width - size + 1, 1))
    for i in range(height - size + 1):
        for j in range(width - size + 1):
            input = imgGray[i:i + size, j:j + size]
            img_conv[i][j] = np.sum(np.multiply(input, conv_filter))
    return img_conv


def gaussian_kernel(kernel_size, mean, variance):
    kernel = np.zeros((kernel_size, kernel_size, 1))
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - mean[0], j - mean[1]
            # gaussian kernel function
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * variance)) / (2 * np.pi * variance)
    # kernel = normalized_grayscale(kernel)
    print('Gaussian Kernel of size', kernel_size, "x", kernel_size, ", mean", mean, ", variance", variance)
    return kernel


def grayscale(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    plt.imshow(imgGray, cmap='gray')
    pltTitle = 'Grayscale images of Dog image'
    plt.title(pltTitle)
    plt.show()
    # print('grayscale image: ', imgGray)
    return imgGray


x = np.random.rand(5, 5, 1)
w = np.random.rand(2, 2, 1)
conv = torch.nn.Conv2d(1, 1, (2, 2), bias=False)
x_torch = torch.from_numpy(x).permute(2, 0, 1)
w_torch = torch.from_numpy(w).permute(2, 0, 1)
x_torch_add = x_torch[None, :]
w_torch_add = w_torch[None, :]
# Set the weight of conv with w
conv.weight = torch.nn.Parameter(w_torch_add)
output = conv(x_torch_add)
print("Output of conv2d:\n", output)
output_own = convolution_operation(x, w)
print("Output of my own operation:\n", output_own)
# print(conv.weight)
img = image.imread('dog.jpg')
imgGray = grayscale(img)
imgGray_torch = torch.from_numpy(imgGray)[:, :, None]
imgGray_torch = imgGray_torch.permute(2, 0, 1)
imgGray_torch_add = imgGray_torch[None, :]
print(imgGray.shape)
gaussian_filter = gaussian_kernel(7, [4, 3], 30)
gaussian_filter_torch = torch.from_numpy(gaussian_filter).permute(2, 0, 1)
gaussian_filter_torch_add = gaussian_filter_torch[None, :]
conv.weight = torch.nn.Parameter(gaussian_filter_torch_add)
output_conv = conv(imgGray_torch_add).detach().numpy()[0][0]
plt.imshow(output_conv, cmap='gray')
plt.title('Blurred image use conv2d')
plt.show()
