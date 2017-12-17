import numpy as np
import scipy
import scipy.signal
from skimage.color import rgb2gray
from scipy.misc import imread as imread
from scipy import linalg as linalg
import matplotlib.pyplot as plt
import os


def read_image(filename, representation):
    """
    A method for reading an image from a path and loading in as gray or in color
    :param filename: The path for the picture to be loaded
    :param representation: The type of color the image will be load in. 1 for gray,
    2 for color
    :return: The loaded image
    """
    im = imread(filename)
    if representation == 1:
        # converting to gray
        im = rgb2gray(im) / 255
    else:
        if representation == 2:
            im = im.astype(np.float64)
            # setting the image's matrix to be between 0 and 1
            im = im / 255
    return im


def create_gaussian_line(size):
    """
    A helper method for creating a gaussian kernel 'line' with the input size
    :param size: The size of the output dimension of the gaussian kernel
    :return: A discrete gaussian kernel
    """
    bin_arr = np.array([1, 1])
    org_arr = np.array([1, 1])
    if (size == 1):
        # special case, returning a [1] matrix
        return np.array([1])
    for i in range(size-2):
        # iterating to create the initial row of the kernel
        bin_arr = scipy.signal.convolve(bin_arr, org_arr)
    bin_arr = np.divide(bin_arr, bin_arr.sum())
    bin_arr = np.reshape(bin_arr, (1,size))
    return bin_arr


def create_gaussian_kernel(size):
    """
    A helper method for creating a gaussian kernel with the input size
    :param size: The size of the output dimension of the gaussian kernel
    :return: A discrete gaussian kernel
    """
    bin_arr = np.array([1, 1])
    org_arr = np.array([1, 1])
    sum = 0
    gaussian_matrix = np.zeros(shape=(size, size))
    # TODO: what if size==1 - should return kernel with [1] ?
    if (size == 1):
        # special case, returning a [1] matrix
        return np.array([1])
    for i in range(size-2):
        # iterating to create the initial row of the kernel
        bin_arr = scipy.signal.convolve(bin_arr, org_arr)
    # calculating values on each entry in matrix
    for x in range(size):
        for y in range(size):
            gaussian_matrix[x][y] = bin_arr[x] * bin_arr[y]
            sum += gaussian_matrix[x][y]
    # TODO: search for element-wise multiplication for vector*vector=matrix
    # TODO: maybe create a matrix from repeated row vector
    # normalizing matrix to 1
    for x in range(size):
        for y in range(size):
            gaussian_matrix[x][y] /= sum
    return gaussian_matrix


def blur_spatial(im, kernel_size):
    """
    A method for calculating a blurred version of input picture
    :param im: The image to blur
    :param kernel_size: The size of the gaussian matrix, indicating the intensity of the blur
    :return: The blurred picture
    """
    # assuming kernel_size is odd
    gaussian_kernel = create_gaussian_kernel(kernel_size)
    conv_im = scipy.signal.convolve2d(im, gaussian_kernel, mode='same')
    return conv_im


def expand(im, filter_vec):
    """
    a helper method for expanding the image by double from it's input size
    :param im: the input picture to expand
    :param filter_vec: a custom filter in case we'd like to convolve with different one
    :return: the expanded picture after convolution
    """
    new_expand = np.zeros(shape=(int(im.shape[0]*2), int(im.shape[1]*2)))
    new_expand[::2,::2] = im
    new_expand = scipy.signal.convolve2d(new_expand, 2*filter_vec, mode='same')
    new_expand = scipy.signal.convolve2d(new_expand, np.transpose(2*filter_vec), mode='same')

    return new_expand


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    a method for building a gaussian pyramid
    :param im: the input image to construct the pyramid from
    :param max_levels: maximum levels in the pyramid
    :param filter_size: the size of the gaussian filter we're using
    :return: an array representing the pyramid
    """
    filter_vec = create_gaussian_line(filter_size)
    # creating duplicate for confy use
    temp_im = im
    pyr = [im]

    for i in range(max_levels - 1):
        # blurring the cur layer
        temp_im = scipy.signal.convolve2d(temp_im, filter_vec, mode='same')
        temp_im = scipy.signal.convolve2d(temp_im, np.transpose(filter_vec), mode='same')
        # sampling only every 2nd row and column
        temp_im = temp_im[::2, ::2]
        pyr.append(temp_im)

    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    a method for building a laplacian pyramid
    :param im: the input image to construct the pyramid from
    :param max_levels: maximum levels in the pyramid
    :param filter_size: the size of the laplacian filter we're using
    :return: an array representing the pyramid
    """
    pyr = []
    org_reduce, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    for i in range(max_levels - 1):
        temp_expand = expand(org_reduce[i + 1], filter_vec)
        org_layer = org_reduce[i]
        temp = org_layer - temp_expand
        pyr.append(temp)
    # plt.imshow(org_reduce[-1], cmap='gray')
    # plt.show()
    pyr.append(org_reduce[-1])
    return pyr, filter_vec


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    a method for blending 2 pictures using a binary mask
    :param im1: the first picture to blend
    :param im2: the second picture to blend
    :param mask: the binary mask
    :param max_levels: number of max levels to be used while constructing the pyramids
    :param filter_size_im: size of the filter for the images
    :param filter_size_mask: size of the filter for the mask
    :return:
    """
    mask = mask.astype(np.float64)
    lap_pyr1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lap_pyr2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)[0]
    gauss_pyr = build_gaussian_pyramid(mask, max_levels, filter_size_mask)[0]
    # TODO: find more elegant way instead of loop
    for i in range(len(gauss_pyr)):
        gauss_pyr[i] = np.array(gauss_pyr[i], dtype=np.float64)
    new_lap_pyr = []
    coeff = [1] * max_levels
    for i in range(max_levels):
        cur_lap_layer = np.multiply(gauss_pyr[i], lap_pyr1[i]) + np.multiply(1 - gauss_pyr[i], lap_pyr2[i])
        new_lap_pyr.append(cur_lap_layer)
    final_image = laplacian_to_image(new_lap_pyr, filter_vec, coeff)
    return np.clip(final_image, 0, 1)
