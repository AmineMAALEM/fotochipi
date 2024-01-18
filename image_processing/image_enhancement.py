import numpy as np
import cv2
import matplotlib.pyplot as plt


def manual_grey_scale(image):
    r,g,b = cv2.split(image)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def image_quatization(image, pixel_depth=8):

    L=256
    factor = L / pixel_depth
    quantized_img = (image / factor).astype(np.uint8) * factor

    return quantized_img


def windowing(img, lower_bound=0,upper_bound=255):

    img = img.copy()
    img[img <= lower_bound] = 0
    img[img >= upper_bound] = 255

    return img


def transf_log_intensity(img):

    max_img = np.max(img)
    img = max_img * np.log(1 + img) / np.log(1 + max_img)

    return img


def gamma(img, gamma):

    img = np.power(img, gamma)
    #img = img / 255.0

    return img


def inverse(image):
    return (255-image)


def inverse_all(image):

    r, g, b = cv2.split(image)
    inverse_r = inverse(r)
    inverse_g = inverse(g)
    inverse_b = inverse(b)

    return inverse_r, inverse_g, inverse_b

def hist_equalizer_mk1(image):

    r2, g2, b2 = cv2.split(image)
    equalized_r2 = cv2.equalizeHist(r2)
    equalized_g2 = cv2.equalizeHist(g2)
    equalized_b2 = cv2.equalizeHist(b2)
    equalized_img_mk1 = cv2.merge([equalized_r2,equalized_g2,equalized_b2])

    return equalized_img_mk1

def hist_equalizer_mk2(image):

    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2RGB)

    return img_eq


def hist_stretch(image):

    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    img_min = np.min(img_yuv[:,:,0])
    img_max = np.max(img_yuv[:,:,0])
    img_yuv[:,:,0] = (img_yuv[:,:,0] - img_min) * (255) / (img_max - img_min)
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

    return img_eq