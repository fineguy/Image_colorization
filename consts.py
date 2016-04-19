import numpy as np

# Behavior
IMG_PATH = "./Images"
VERBOSE = False
REWRITE = False
RETRAIN = False
N_PROCESSES = 4

N_SEGMENTS = 200

IMG_WIDTH = 500
IMG_HEIGHT = 500
PATCH_WIDTH = 10
PATCH_HEIGHT = 10

FIGSIZE = (10, 10)

RGB_FROM_YUV = np.array([[1, 0, 1.13983],
                         [1, -0.39465, -.58060],
                         [1, 2.03211, 0]]).T
YUV_FROM_RGB = np.array([[0.299, 0.587, 0.114],
                         [-0.14713, -0.28886, 0.436],
                         [0.615, -0.51499, -0.10001]]).T
U_MAX = 0.436
V_MAX = 0.615

EPS = 0.0625
C = 0.125

# Constants for running ICM on the MRF
ICM_ITERATIONS = 10
ITER_EPSILON = .01
COVAR = 0.25       # Covariance of predicted chrominance from SVR and actual covariance
WEIGHT_DIFF = 2    # Relative importance of neighboring superpixels
THRESHOLD = 25     # Threshold for comparing adjacent superpixels.
                   # Setting a higher threshold reduces error, but causes the image to appear more uniform.

# Paths
MODEL_SECOND = "second.model"
MODEL_THIRD = "third.model"
SCALER = "scaler.model"
DATA_EXT = ".data.npy"
GREY_EXT = ".grey"