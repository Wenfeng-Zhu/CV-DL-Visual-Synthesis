import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
import cv2


def grayscale(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    plt.imshow(imgGray, cmap='gray')
    pltTitle = 'Grayscale images of Dog image'
    plt.title(pltTitle)
    plt.show()
    print('grayscale image: ', imgGray)
    return imgGray


def convolution_operation(imgGray, conv_filter):
    height, width = imgGray.shape
    size = len(conv_filter)
    # create a new numpy array with padding to make sure the filtered image has the same size as the input image
    # It must be noted, however, that filters of odd size will facilitate the calculation
    img_padding = np.zeros((height + size - 1, width + size - 1))
    img_padding[((size - 1) // 2):((size - 1) // 2 + height), ((size - 1) // 2):((size - 1) // 2 + width)] = imgGray
    img_conv = []
    for i in range(height):
        row = []
        for j in range(width):
            input = img_padding[i:i + len(conv_filter), j:j + len(conv_filter)]
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
    print("Going from " + str(-w_range) + " to " + str(w_range))
    for i in range(-w_range, w_range + 1):
        for j in range(-w_range, w_range + 1):
            l_o_g_mask.append(laplacian_of_gaussian(i, j, sigma))
    print(len(l_o_g_mask))
    l_o_g_mask = np.array(l_o_g_mask)
    print(l_o_g_mask)
    l_o_g_mask = l_o_g_mask.reshape(int(w), int(w))
    return l_o_g_mask


def laplacian_filter(imgGray, kernel_size, sigma):
    log_mask = create_log(sigma, kernel_size)
    img_log = convolution_operation(imgGray, log_mask)
    norm = (img_log - np.min(img_log)) / (np.max(img_log) - np.min(img_log))
    img_log = img_log * norm * 255
    plt.imshow(img_log, cmap="gray")
    plt.show()
    plt.clf()
    return img_log


img = image.imread('dog.jpg')
imgGray = grayscale(img)
filter_1 = np.asarray([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
gaussian_filter = gaussian_kernel(200, [60, 100], 30)
# plt.imshow(gaussian_filter, cmap='gray')
# plt.title('Gaussian Filter of 200x200 size')
# plt.show()
img_conv_gaussian = convolution_operation(imgGray, gaussian_kernel(7, [4, 3], 30))
# plt.imshow(img_conv_gaussian, cmap='gray')
# plt.title('blurred image of Dog image')
# plt.show()
# laplacian_kernel = laplacian_filter(3, 2)
img_laplacian = laplacian_filter(imgGray, 7, 1)
# plt.imshow(img_laplacian, cmap='gray')
# plt.title('Sharpening image of Dog image')
# print(img_laplacian)
# plt.show()
source = cv2.imread('dog.jpg', cv2.IMREAD_COLOR)
source = cv2.GaussianBlur(source, (3, 3), 0)
source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
dest = cv2.Laplacian(source_gray, cv2.CV_16S, ksize=3)
abs_dest = cv2.convertScaleAbs(dest)
plt.imshow(abs_dest, cmap="gray")
plt.show()
