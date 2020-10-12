import numpy as np
from scipy.signal import gaussian


def add_distortions(spectra: np.ndarray, level: float = 0.2, seed: float = 42) -> np.ndarray:
    """
    Adds random distortions with max height of "level" to the set of spectra.
    :param spectra: (N, M) array, M-1 spectra with N wavenumbers, wavenumbers in first column
    :param level: Max height of added distortion, relative to normalized intensity
    :param seed: Random seed
    :return: the altered spectra, shape (N, M) array
    """
    np.random.seed(seed)
    spectra: np.ndarray = spectra.copy()

    minDistortWidth, maxDistortWidth = round(spectra.shape[0]*0.6), round(spectra.shape[0]*0.9)
    minDistortStd, maxDistortStd = 20, 40

    for i in range(spectra.shape[1]-1):
        intensities: np.ndarray = spectra[:, i+1]

        # for each, first normalize, then add the distortion, then scale back up to orig dimensions
        minVal, maxVal = intensities.min(), intensities.max()
        intensities -= minVal
        intensities /= (maxVal - minVal)

        gaussSize: int = int(round(np.random.rand() * (maxDistortWidth-minDistortWidth) + minDistortWidth))
        gaussStd: float = np.random.rand() * (maxDistortStd - minDistortStd) + minDistortStd
        randGauss = gaussian(gaussSize, gaussStd) * np.random.rand() * level

        start = int(round(np.random.rand() * (len(intensities) - gaussSize)))

        intensities[start:start+gaussSize] += randGauss

        intensities *= (maxVal - minVal)
        intensities += minVal

        spectra[:, i+1] = intensities

    return spectra


def add_noise(spectra: np.ndarray, maxLevel: float = 0.1, seed: float = 42) -> np.ndarray:
    """
    Adds random noise to the spectra..
    :param spectra: (N, M) array, M-1 spectra with N wavenumbers, wavenumbers in first column
    :param maxLevel: max Level of noise
    :param seed: random seed
    :return: new Spectra (N, M) array
    """
    np.random.seed(seed)
    spectra = spectra.copy()
    spectra[:, 1:] *= (1-maxLevel/2) + np.random.rand(spectra.shape[0], spectra.shape[1]-1) * maxLevel
    return spectra
