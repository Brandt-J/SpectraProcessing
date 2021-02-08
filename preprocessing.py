import numpy as np
from scipy.signal import savgol_filter


def snv(input_data: np.ndarray) -> np.ndarray:
    """
    Standard normal variate Correction.
    :param input_data: Shape (NxM) array of N samples with M features
    :return: corrected data in same shape
    """
    output_data: np.ndarray = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        output_data[i, :] = (input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])
    return output_data


def mean_center(input_data: np.ndarray) -> np.ndarray:
    """
    Mean Centering.
    :param input_data: Shape (NxM) array of N samples with M features
    :return: corrected data in same shape
    """
    output_data: np.ndarray = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        output_data[i, :] = input_data[i, :] - np.mean(input_data[i, :])
    return output_data


def detrend(input_data: np.ndarray) -> np.ndarray:
    """
    Removes a linear baseline.
    :param input_data: Shape (NxM) array of N samples with M features
    :return: corrected data in same shape
    """
    output_data: np.ndarray = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        baseline = np.linspace(input_data[i, 0], input_data[i, -1], len(input_data[i, :]))
        output_data[i, :] = input_data[i, :] - baseline
    return output_data


def smooth(input_data: np.ndarray, windowSize: int) -> np.ndarray:
    """
    Applies Savitzky Golay smoothing to all given data.
    :param input_data: Shape (NxM) array of N samples with M features
    :param windowSize: integer, the window size for smoothing
    :return: corrected data in same shape
    """
    output_data: np.ndarray = np.zeros_like(input_data)
    if windowSize % 2 == 0:
        windowSize += 1  # has to be an odd number!
    for i in range(input_data.shape[0]):
        output_data[i, :] = savgol_filter(input_data[i, :], windowSize, polyorder=3)
    return output_data
