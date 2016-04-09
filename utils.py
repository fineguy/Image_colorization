from consts import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit

import os
import glob
import pickle
from time import time
from functools import partial
from multiprocessing import Pool, cpu_count

from skimage.segmentation import slic
from skimage.transform import rescale
from skimage.io import imread, imshow, imsave
from skimage.color import rgb2lab, lab2rgb

from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        print("Finished %s in %d seconds" % (func.__name__, time() - start_time))
        return result
    return wrapper


# segment a grey image into superpixels
def get_segments(img_grey, n_segments=N_SEGMENTS):
    return slic(img_grey, n_segments=n_segments, compactness=0.1, max_iter=100, multichannel=False)


def get_centroids(segments):
    n_segments = segments.max() + 1
    pixel_count = np.zeros(n_segments, dtype='uint')
    centroids = np.zeros((n_segments, 2), dtype='uint')

    # accumulate values
    for (i,j), value in np.ndenumerate(segments):
        pixel_count[value] += 1
        centroids[value][0] += i
        centroids[value][1] += j

    # average values
    centroids = (centroids.T // pixel_count).T

    return centroids


# return the corresponding patch for centroid
def get_patch(centroid, img_grey):
    patch = np.zeros((PATCH_HEIGHT, PATCH_WIDTH))
    top = max(centroid[0] - PATCH_HEIGHT//2, 0)
    left = max(centroid[1] - PATCH_WIDTH//2, 0)
    bottom = min(centroid[0] + (PATCH_HEIGHT - 1)//2, img_grey.shape[0]-1) + 1
    right = min(centroid[1] + (PATCH_WIDTH - 1)//2, img_grey.shape[1]-1) + 1
    patch[:(bottom-top), :(right-left)] = img_grey[top:bottom, left:right]
    return patch


def extract_features(img_grey, segments):
    n_segments = segments.max() + 1

    if VERBOSE:
        print("Calculating centroids")

    centroids = get_centroids(segments)

    if VERBOSE:
        print("Calculating features")

    f = lambda x : np.fft.fft2(get_patch(x, img_grey)).reshape(PATCH_HEIGHT*PATCH_WIDTH)
    features = np.array([f(centroid) for centroid in centroids])

    return features


# returns features and their corresponding U and V values
def get_data(img_path, rewrite=REWRITE):
    data_path = img_path + DATA_EXT

    if not os.path.exists(data_path) or rewrite:
        img = imread(img_path)
        scale_factor = min(IMG_HEIGHT/img.shape[0], IMG_WIDTH/img.shape[1])
        img_lab = rgb2lab(rescale(img, scale_factor))
        img_grey = img_lab[:,:,0]*255//100
        segments = get_segments(img_grey)
        n_segments = segments.max() + 1

        if VERBOSE:
            print("--- Extracting features for %s ---" % img_path)
        features = extract_features(img_grey, segments)

        pixel_count = np.zeros(n_segments, dtype='uint')
        second_ch = np.zeros(n_segments)
        third_ch = np.zeros(n_segments)

        # accumulate values
        for (i,j), value in np.ndenumerate(segments):
            pixel_count[value] += 1
            second_ch[value] += img_lab[i][j][1]
            third_ch[value] += img_lab[i][j][2]

        # average values
        second_ch[pixel_count != 0] //= pixel_count[pixel_count != 0]
        third_ch[pixel_count != 0] //= pixel_count[pixel_count != 0]

        data_dict = {}
        data_dict['features'], data_dict['second'], data_dict['third'] = features, second_ch, third_ch
        np.save(data_path, data_dict)
    else:
        data_dict = np.load(data_path).item()
        features, second_ch, third_ch = data_dict['features'], data_dict['second'], data_dict['third']

    return features, second_ch, third_ch


# train separate models for predicting U and V values
def get_predictors(features, u_values, v_values, retrain=RETRAIN):
    if not os.path.exists(MODEL_U) or retrain:
        if VERBOSE:
            print("Fitting second channel values")
        svr_u = SVR(C=C, epsilon=EPS).fit(np.absolute(features), u_values)
        with open(MODEL_U, 'wb') as f:
            pickle.dump(svr_u, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(MODEL_U, 'rb') as f:
            svr_u = pickle.load(f)

    if not os.path.exists(MODEL_V) or retrain:
        if VERBOSE:
            print("Fitting third channel values")
        svr_v = SVR(C=C, epsilon=EPS).fit(np.absolute(features), v_values)
        with open(MODEL_V, 'wb') as f:
            pickle.dump(svr_v, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(MODEL_V, 'rb') as f:
            svr_v = pickle.load(f)

    return svr_u, svr_v


def img_features(img_files):
    with Pool(N_PROCESSES) as pool:
        data = pool.map(get_data, img_files, chunksize=5)
    features = np.concatenate([x[0] for x in data])
    u_values = np.concatenate([x[1] for x in data])
    v_values = np.concatenate([x[2] for x in data])
    return features, u_values, v_values


def colorize_image(img_grey, grey_path, clf_second, clf_third, save, show):
    segments = get_segments(img_grey)
    features = extract_features(img_grey, segments)
    second_pred, third_pred = clf_second.predict(features), clf_third.predict(features)

    # Reconstruct image
    img_ = np.zeros((img_grey.shape[0], img_grey.shape[1], 3))
    img_[:,:,0] = img_grey
    for (i,j), value in np.ndenumerate(segments):
        img_[i][j][1] = second_pred[value]
        img_[i][j][2] = third_pred[value]
    img_ = lab2rgb(img_)

    if save:
        imsave(grey_path, img_)

    if show:
        plt.figure(figsize=FIGSIZE)
        plt.imshow(img_grey, plt.cm.binary)

        plt.figure(figsize=FIGSIZE)
        plt.imshow(img_)


def show_work(img_files, clf_second, clf_third):
    cont = True
    for img_path in img_files:
        grey_path = img_path + GREY_EXT
        img = imread(img_path)
        scale_factor = min(IMG_HEIGHT/img.shape[0], IMG_WIDTH/img.shape[1])
        img = rgb2lab(rescale(img, scale_factor))[:,:,0]*255//100
        plt.figure(figsize=FIGSIZE)
        plt.imshow(img)
        colorize_image(img_grey, grey_path, clf_second, clf_third, save=False, show=True)
        if input() == 'c':
            break

@timeit
def main():
    img_files = glob.glob(os.path.join(IMG_PATH, "*.jpg"))
    train_img_files, test_img_files = train_test_split(img_files, test_size=0.7)

    train_features, train_u_values, train_v_values = img_features(train_img_files)
    test_features, test_u_values, test_v_values = img_features(test_img_files)

    clf_second, clf_third = get_predictors(train_features, train_u_values, train_v_values)
    show_work(test_img_files, clf_second, clf_third)


if __name__ == "__main__":
    main()