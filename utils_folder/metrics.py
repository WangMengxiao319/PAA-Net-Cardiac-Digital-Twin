import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastdtw import fastdtw
import pywt
from tqdm import tqdm

def DTW(y, y_pred):
    """
    Compute Dynamic Time Warping (DTW) for the signals.

    Parameters:
    y: Original signal.
    y_pred: Predicted signal.
    shape: (batch, signal_length)
    Returns:
    DTW value for each signal.

    Interpretation:
    A lower DTW value indicates greater similarity between the predicted and original signals.
    """

    dtw_scores = []
    for i in range(len(y)):
        distance, _ = fastdtw(y[i], y_pred[i])
        dtw_scores.append(distance)

    return np.array(dtw_scores)
