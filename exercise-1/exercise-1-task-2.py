import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt


# import cv2


def convolution_fix():
    np.set_printoptions(linewidth=9)
    img = np.array([[[0], [2], [1]], [[3], [4], [2]], [[1], [0], [3]]])
    print("The input array is:\n", img)
    kernel = np.array([[[1], [2]], [[2], [1]]])
    print("The kernel array is:\n", kernel)
    img_conv = np.zeros((2, 2, 1))
    for i in range(2):
        for j in range(2):
            input = img[i:i + len(kernel), j:j + len(kernel)]
            img_conv[i][j] = np.sum(np.multiply(input, kernel))
    print("The output array is:\n", img_conv)


def grayscale(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    plt.imshow(imgGray, cmap='gray')
    pltTitle = 'Grayscale images of Dog image'
    plt.title(pltTitle)
    plt.show()
    # print('grayscale image: ', imgGray)
    return imgGray


def normalized_grayscale(array):
    norm = (array - np.min(array)) / (np.max(array) - np.min(array))
    # array = array * norm * 255
    return array * norm * 255


def convolution_operation(imgGray, conv_filter):
    height, width = imgGray.shape
    size = len(conv_filter)
    # create a new numpy array with padding to make sure the filtered image has the same size as the input image
    # It must be noted, however, that filters of odd size will facilitate the calculation
    # img_padding = np.zeros((height + size - 1, width + size - 1))
    # img_padding[((size - 1) // 2):((size - 1) // 2 + height), ((size - 1) // 2):((size - 1) // 2 + width)] = imgGray
    img_conv = []
    for i in range(height - size + 1):
        row = []
        for j in range(width - size + 1):
            input = imgGray[i:i + len(conv_filter), j:j + len(conv_filter)]
            row.append(np.sum(np.multiply(input, conv_filter)))
        img_conv.append(row)
    return np.array(img_conv)


def gaussian_kernel(kernel_size, mean, variance):
    kernel = np.zeros((kernel_size, kernel_size))
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - mean[0], j - mean[1]
            # gaussian kernel function
            kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * variance)) / (2 * np.pi * variance)
    # kernel = normalized_grayscale(kernel)
    print('Gaussian Kernel of size', kernel_size, "x", kernel_size, ", mean", mean, ", variance", variance)
    return kernel


def laplacian_of_gaussian(x, y, sigma):
    # Formatted this way for readability
    nom = (y ** 2) + (x ** 2) - 2 * (sigma ** 2)
    denom = 2 * np.pi * (sigma ** 6)
    expo = np.exp(-((x ** 2) + (y ** 2)) / (2 * (sigma ** 2)))
    return (nom * expo / denom)


def create_log(sigma, kernel_size):
    w = (np.ceil(float(kernel_size) * float(sigma)))
    if w % 2 == 0:
        w = w + 1
    l_o_g_mask = []
    w_range = int(np.floor(w / 2))
    for i in range(-w_range, w_range + 1):
        for j in range(-w_range, w_range + 1):
            l_o_g_mask.append(laplacian_of_gaussian(i, j, sigma))
    l_o_g_mask = np.array(l_o_g_mask)
    l_o_g_mask = l_o_g_mask.reshape(int(w), int(w))
    return l_o_g_mask


def laplacian_filter(imgGray, kernel_size, sigma):
    log_mask = create_log(sigma, kernel_size)
    img_log = convolution_operation(imgGray, log_mask)
    norm = (img_log - np.min(img_log)) / (np.max(img_log) - np.min(img_log))
    img_log = img_log * norm * 255
    return img_log


convolution_fix()

img = image.imread('dog.jpg')

imgGray = grayscale(img)

gaussian_filter = gaussian_kernel(200, [80, 100], 50)
plt.imshow(gaussian_filter, cmap='gray')
plt.title('Gaussian Kernel')
plt.show()

img_conv_gaussian = convolution_operation(imgGray, gaussian_kernel(7, [4, 3], 30))
plt.imshow(img_conv_gaussian, cmap="gray")
plt.title('Blurred image')
plt.show()

img_laplacian = laplacian_filter(imgGray, 7, 1)
plt.imshow(img_laplacian, cmap="gray")
plt.title('Laplace operation')
plt.show()
