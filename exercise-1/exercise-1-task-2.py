import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt


def grayscale(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    imgGray = (0.2989 * R + 0.5870 * G + 0.1140 * B)/255
    plt.imshow(imgGray, cmap='gray')
    pltTitle = 'Grayscale images of Dog image'
    plt.title(pltTitle)
    plt.show()
    print('grayscale image: ', imgGray)
    return imgGray


def convolution_operation(imgGray, conv_filter):
    print(conv_filter)
    height, width = imgGray.shape
    img_conv = []
    for i in range(height - len(conv_filter) - 1):
        row = []
        for j in range(width - len(conv_filter) - 1):
            input = imgGray[i:i + len(conv_filter), j:j + len(conv_filter)]
            row.append(np.sum(np.multiply(input, conv_filter)))
        img_conv.append(row)
    return np.array(img_conv)


def gaussian_kernel(kernel_size, mean, variance):
    kernel = np.zeros((kernel_size, kernel_size))
    # center = kernel_size // 2
    sum_value = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - mean[0], j - mean[1]
            # gaussian kernel function
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * variance)) / (2 * np.pi * variance)
            sum_value += kernel[i, j]
    kernel = kernel / sum_value
    print('normalized convolution filter: ', kernel)
    # plt.imshow(kernel, cmap='gray')
    # plt.show()
    return kernel


img = image.imread('dog.jpg')
imgGray = grayscale(img)
filter_1 = np.asarray([[1, 2],
                       [2, 1]])

# print(imgGray, height, width)
# img_conv = convolution_operation(imgGray, filter_1)
# plt.imshow(img_conv)
# print(img_conv, img_conv.shape)
# plt.show()
# plt.clf()
# img_conv_gaussian = convolution_operation(imgGray, gaussian_kernel(200, 2))

gaussian_filter = gaussian_kernel(200, [60, 100], 30)
plt.imshow(gaussian_filter, cmap='gray')
plt.title('Gaussian Filter of 200x200 size')
plt.show()
img_conv_gaussian = convolution_operation(imgGray, gaussian_kernel(18, [10, 8], 30))
print(img_conv_gaussian)
plt.imshow(img_conv_gaussian, cmap='gray')
plt.title('blurred image of Dog image')
plt.show()
