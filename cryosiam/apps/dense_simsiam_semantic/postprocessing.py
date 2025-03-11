import numpy as np
from skimage.measure import label
from skimage.filters import gaussian


def get_largest_cc(segmentation, threshold=0.5):
    segmentation = (gaussian(segmentation, sigma=10, mode='constant') > threshold).astype(int)
    labels = label(segmentation)
    largest_cc = labels == np.argmax(np.bincount(labels.flat, weights=segmentation.flat))
    return largest_cc
