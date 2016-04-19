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

from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler


def timeit(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        print("Finished %s in %d seconds" % (func.__name__, time() - start_time))
        return result
    return wrapper


def clamp(values, low, high):
    return np.maximum(np.minimum(values, high), low)


def convert(img):
    return np.dot(img, YUV_FROM_RGB)


def retrieve(img):
    return clamp(np.dot(img, RGB_FROM_YUV), 0, 1)


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
        img_ = convert(rescale(img, scale_factor))
        img_grey = img_[:,:,0]
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
            second_ch[value] += img_[i][j][1]
            third_ch[value] += img_[i][j][2]

        # average values
        second_ch[pixel_count != 0] /= pixel_count[pixel_count != 0]
        third_ch[pixel_count != 0] /= pixel_count[pixel_count != 0]

        data_dict = {}
        data_dict['features'], data_dict['second'], data_dict['third'] = features, second_ch, third_ch
        np.save(data_path, data_dict)
    else:
        data_dict = np.load(data_path).item()
        features, second_ch, third_ch = data_dict['features'], data_dict['second'], data_dict['third']

    return features, second_ch, third_ch


# train separate models for predicting second and third channels
@timeit
def get_predictors(features, second_ch, third_ch, retrain=RETRAIN):
    if not os.path.exists(SCALER) or retrain:
        if VERBOSE:
            print("Scaling data")
        scl = StandardScaler().fit(np.absolute(features))
        with open(SCALER, 'wb') as f:
            pickle.dump(scl, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(SCALER, 'rb') as f:
            scl = pickle.load(f)

    if not os.path.exists(MODEL_SECOND) or retrain:
        if VERBOSE:
            print("Fitting second channel values")
        clf_second = SVR(C=C, epsilon=EPS).fit(features, second_ch)
        with open(MODEL_SECOND, 'wb') as f:
            pickle.dump(clf_second, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(MODEL_SECOND, 'rb') as f:
            clf_second = pickle.load(f)

    if not os.path.exists(MODEL_THIRD) or retrain:
        if VERBOSE:
            print("Fitting third channel values")
        clf_third = SVR(C=C, epsilon=EPS).fit(features, third_ch)
        with open(MODEL_THIRD, 'wb') as f:
            pickle.dump(clf_third, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(MODEL_THIRD, 'rb') as f:
            clf_third = pickle.load(f)

    return clf_second, clf_third


@timeit
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
    second_pred = clamp(clf_second.predict(features), -U_MAX, U_MAX)
    third_pred = clamp(clf_third.predict(features), -V_MAX, V_MAX)

    # Reconstruct image
    img_ = np.zeros((img_grey.shape[0], img_grey.shape[1], 3))
    img_[:,:,0] = img_grey
    for (i,j), value in np.ndenumerate(segments):
        img_[i][j][1] = second_pred[value]
        img_[i][j][2] = third_pred[value]
    img_ = retrieve(img_)

    if save:
        imsave(grey_path, img_)

    if show:
        plt.figure(figsize=FIGSIZE)
        plt.imshow(img_grey, plt.cm.binary)

        plt.figure(figsize=FIGSIZE)
        plt.imshow(img_)

'''
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
'''

# Generates the adjacency list for each of the segments in the image.
def get_neighbors(segments, features, threshold=THRESHOLD):
    n_segments = segments.max() + 1
    neighbors = [set() for i in range(n_segments)]

    for (i,j), val in np.ndenumerate(segments):
        # Check vertical adjacency
        if i < segments.shape[0] - 1:
            next_val = segments[i + 1][j]
            if val != next_val and np.linalg.norm(features[val] - features[next_val]) < threshold:
                neighbors[val].add(next_val)
                neighbors[next_val].add(val)

        # Check horizontal adjacency
        if j < segments.shape[1] - 1:
            next_val = segments[i][j + 1]
            if val != next_val and np.linalg.norm(features[val] - features[next_val]) < threshold:
                neighbors[val].add(next_val)
                neighbors[next_val].add(val)

    return neighbors


# Given the prior observed_u and observed_v, which are generated using the SVR,
# represent the system as a Markov Random Field and optimize over it using
# Iterated Conditional Modes. Return the prediction of the hidden U and V values
# of the segments.
# For now, we assume that the U and V channels behave independently.
def apply_mrf(U, V, segments, features):
    n_segments = segments.max() + 1
    hid_u = np.copy(U)
    hid_v = np.copy(V)
    u_arr = np.arange(-U_MAX, U_MAX, .001)
    v_arr = np.arange(-V_MAX, V_MAX, .001)
    u_mat = np.tile(np.vstack(u_arr), len(hid_u))
    v_mat = np.tile(np.vstack(v_arr), len(hid_v))

    neighbors = get_neighbors(segments, features)

    for i in range(ICM_ITERATIONS):
        new_u = np.zeros(n_segments)
        new_v = np.zeros(n_segments)

        for k in range(n_segments):
            # Compute conditional probability over all possibilities of U
            comp_u = np.square(u_arr-U[k])/(2*COVAR) + \
                WEIGHT_DIFF*np.sum(np.square(u_mat[:,list(neighbors[k])]-hid_u[list(neighbors[k])]), axis=1)
            new_u[k] = u_arr[np.argmin(comp_u)]

            # Compute conditional probability over all possibilities of V
            comp_v = np.square(v_arr-V[k])/(2*COVAR) + \
                WEIGHT_DIFF*np.sum(np.square(v_mat[:,list(neighbors[k])]-hid_v[list(neighbors[k])]), axis=1)
            new_v[k] = v_arr[np.argmin(comp_v)]

        u_diff = np.linalg.norm(hid_u - new_u)
        v_diff = np.linalg.norm(hid_v - new_v)
        hid_u = new_u
        hid_v = new_v

        if u_diff < ITER_EPSILON and v_diff < ITER_EPSILON:
            break

    return hid_u, hid_v


def work_on(img_path):
    img = imread(img_path)
    scale_factor = min(IMG_HEIGHT/img.shape[0], IMG_WIDTH/img.shape[1])

    plt.figure(figsize=FIGSIZE)
    plt.imshow(img)

    img_grey = convert(rescale(img, scale_factor))[:,:,0]
    segments = get_segments(img_grey)
    features = extract_features(img_grey, segments)
    second_pred = clamp(clf_second.predict(features), -U_MAX, U_MAX)
    third_pred = clamp(clf_third.predict(features), -V_MAX, V_MAX)

    plt.figure(figsize=FIGSIZE)
    plt.imshow(img_grey, plt.cm.binary)

    mrf_second_pred, mrf_third_pred = apply_mrf(second_pred*2, third_pred*2, segments, features)

    # Reconstruct image
    img_ = np.zeros((img_grey.shape[0], img_grey.shape[1], 3))
    img_[:,:,0] = img_grey
    for (i,j), value in np.ndenumerate(segments):
        img_[i][j][1] = mrf_second_pred[value]
        img_[i][j][2] = mrf_third_pred[value]

    plt.figure(figsize=FIGSIZE)
    plt.imshow(retrieve(img_))