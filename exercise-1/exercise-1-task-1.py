import matplotlib.pyplot as plt
import matplotlib.image as image
import random
import numpy as np
from PIL import Image


# Plot the original image and print the Height, Width and number of Channels
def load_image(img):
    plt.imshow(img)
    height, width = img.shape[0], img.shape[1]
    channels = img.shape[2]
    print("height =", height, "width =", width, "channels =", channels)
    pltTitle = 'Dog image of ' + str(height) + ' X ' + str(width) + ' X ' + str(channels)
    plt.title(pltTitle)
    plt.show()
    plt.clf()


# According to the start point coordinate to slice the crop we need
def random_crop(img, x, y):
    img_crop = img[y:(y + 256), x:(x + 256)]
    plt.imshow(img_crop)
    pltTitle = 'Random Crop of 256 X 256 X 3 starts from (' + str(x) + ' , ' + str(y) + ')'
    plt.title(pltTitle)
    plt.show()
    plt.clf()
    return img_crop


# Covert RGB original image to grayscale
def grayscale(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    # RGB to grayscale formula
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    return imgGray


# Covert the grayscale slice to RGB channel and insert it back to the original image
def insert_grayscale(img, img_crop_grayscale, x, y):
    img[y:(y + 256), x:(x + 256)] = np.stack((img_crop_grayscale,) * 3, axis=-1)
    plt.imshow(img)
    pltTitle = 'Insert the grayscale patch back into the original image'
    plt.title(pltTitle)
    plt.show()
    plt.clf()
    return img


# use the image and resize factor as the input
def resize(img, factor):
    new_size = int(img.shape[1] * factor), int(img.shape[0] * factor)
    # use PIL library to transform numpy array to PIL array
    img_pil = Image.fromarray(img)
    # use the resize() function of PIL library
    img_resize = img_pil.resize(new_size)
    plt.imshow(img_resize)
    pltTitle = 'Resize image with 1/2 of grayscale-inserted imag'
    plt.title(pltTitle)
    plt.show()


# use the imread() function to load the dog image, and it will return a NumPy array
img = image.imread('dog.jpg')
# use the length of the array to present the Height, the Width, and the number of Channels
height, width, channels = len(img), len(img[0]), len(img[0][0])
# randomly generate the start point of the slice of 256*256 size
start_x, start_y = random.randint(0, (width - 257)), random.randint(0, (height - 257))
# Load dog image with Load() function
load_image(img)
# Random Crop
img_crop = random_crop(img, start_x, start_y)
# Turn the random crop to grayscale
img_crop_grayscale = grayscale(img_crop)
plt.imshow(img_crop_grayscale, cmap='gray')
pltTitle = 'Grayscale images of Random Crop'
plt.title(pltTitle)
plt.show()
# Insert the grayscale random crop to original image
img_insert_grayscale = insert_grayscale(img, img_crop_grayscale, start_x, start_y)
# resize the image
resize(img_insert_grayscale, 0.5)
