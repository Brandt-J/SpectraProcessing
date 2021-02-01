import numpy as np
from scipy.signal import gaussian


def create_n_distorted_copies(spectra: np.ndarray, n: int, level: float = 0.3, seed: int = 42) -> np.ndarray:
    """
    Creates n copies with distortions of the given spectra set.
    :param spectra: (N, M) array, M-1 spectra with N wavenumbers, wavenumbers in first column
    :param n: int, number of variations to create
    :param level: Max height of added distortion, relative to normalized intensity
    :param seed: Random seed
    :return: the altered spectra, shape (N, (M-1)*(n+1) + 1) array
    """
    numSpectra: int = spectra.shape[1] - 1
    finalSpectra: np.ndarray = np.zeros((spectra.shape[0], numSpectra * (n + 1) + 1))
    finalSpectra[:, :spectra.shape[1]] = spectra
    maxIterations = 5
    print(f'requesting {n} Variations, producing them with max {maxIterations} iterations')
    iterationSeed = seed
    for i in range(n):
        newSpecs: np.ndarray = spectra.copy()
        iterationSeed += 1
        np.random.seed(iterationSeed)
        curRand = np.random.rand()
        for j in range(np.random.randint(maxIterations)):
            curLevel = np.clip((np.random.rand()*3 + 0.5) * level, 0.0, 1.0)
            iterationSeed += 1
            newSpecs = add_noise(newSpecs, level=curLevel, seed=iterationSeed)
            if curRand > 0.2:
                newSpecs = add_distortions(newSpecs, level=curLevel, seed=iterationSeed)
            if curRand > 0.4:
                newSpecs = add_ghost_peaks(newSpecs, level=curLevel, seed=iterationSeed)

        start, stop = (i+1) * numSpectra + 1, (i+2) * numSpectra + 1
        finalSpectra[:, start:stop] = newSpecs[:, 1:]

    return finalSpectra


def add_distortions(spectra: np.ndarray, level: float = 0.1, seed: int = 42) -> np.ndarray:
    """
    Adds random distortions with max height of "level" to the set of spectra.
    :param spectra: (N, M) array, M-1 spectra with N wavenumbers, wavenumbers in first column
    :param level: Max height of added distortion, relative to normalized intensity
    :param seed: Random seed
    :return: the altered spectra, shape (N, M) array
    """
    spectra: np.ndarray = spectra.copy()
    for i in range(spectra.shape[1]-1):
        seed += 1
        np.random.seed(seed)

        intensities: np.ndarray = spectra[:, i+1]
        # for each, first normalize, then add the distortion, then scale back up to orig dimensions
        minVal, maxVal = intensities.min(), intensities.max()
        intensities -= minVal
        intensities /= (maxVal - minVal)

        # Bend Baseline
        randInt = np.random.rand() * level
        randFreq = 5e-5 + np.random.rand() * 4e-3
        randOffset = np.random.rand() * 1000
        distortion = np.sin(spectra[:, 0] * randFreq + randOffset)
        for j in range(np.random.randint(1, 5)):
            power = np.random.randint(1, 5)
            distortion += 0.5 * np.random.rand() * np.sin(spectra[:, 0] * randFreq * (j+3) + (j+1) * randOffset) ** power

        distortion -= distortion.min()
        distortion /= distortion.max()
        intensities = (1 - randInt) * intensities + randInt * distortion

        intensities *= (maxVal - minVal)
        intensities += minVal

        spectra[:, i+1] = intensities

    return spectra


def add_ghost_peaks(spectra: np.ndarray, level: float = 0.1, seed: int = 42) -> np.ndarray:
    spectra: np.ndarray = spectra.copy()

    minDistortWidth, maxDistortWidth = round(spectra.shape[0] * 0.6), round(spectra.shape[0] * 0.9)
    minDistortStd, maxDistortStd = 20, 40

    for i in range(spectra.shape[1]-1):
        seed += 1
        np.random.seed(seed)
        intensities: np.ndarray = spectra[:, i+1]
        # for each, first normalize, then add the distortion, then scale back up to orig dimensions
        minVal, maxVal = intensities.min(), intensities.max()
        intensities -= minVal
        intensities /= (maxVal - minVal)

        # Add fake peaks
        gaussSize: int = int(round(np.random.rand() * (maxDistortWidth - minDistortWidth) + minDistortWidth))
        gaussStd: float = np.random.rand() * (maxDistortStd - minDistortStd) + minDistortStd
        randGauss = gaussian(gaussSize, gaussStd) * np.random.rand() * level

        start = int(round(np.random.rand() * (len(intensities) - gaussSize)))
        intensities[start:start + gaussSize] += randGauss

        intensities *= (maxVal - minVal)
        intensities += minVal

        spectra[:, i+1] = intensities

    return spectra


def add_noise(spectra: np.ndarray, level: float = 0.1, seed: int = 42) -> np.ndarray:
    """
    Adds random noise to the spectra..
    :param spectra: (N, M) array, M-1 spectra with N wavenumbers, wavenumbers in first column
    :param level: max Level of noise
    :param seed: random seed
    :return: new Spectra (N, M) array
    """
    np.random.seed(seed)
    spectra = spectra.copy()
    freq = np.random.rand() * 0.01
    offset = np.random.rand() * spectra[:, -1]
    noisePower: np.ndarray = np.sin(freq * np.arange(spectra.shape[0]) + offset)/2 + 0.5  # sin function between 0 and 1
    for i in range(spectra.shape[1]-1):
        seed += 1
        np.random.seed(seed)
        spectra[:, i+1] *= (1-level/2) + np.random.rand(spectra.shape[0]) * noisePower * level
    return spectra
