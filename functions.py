from typing import List, Tuple, Dict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance_matrix


def compareResultLists(trueList: List[str], estimatedList: List[str]) -> Tuple[float, Dict[str, float]]:
    """
    Compares two assignment lists.
    :param trueList: List of true assignments
    :param estimatedList: List of estimated assignments
    :returns: Tuple(agreement in percent, Dictionary of agreements in percent per type of assignment)
    """
    assert len(trueList) == len(estimatedList)

    replaceDict: dict = {'ldpe': 'pe',
                         'hdpe': 'pe'}

    for i in range(len(trueList)):
        trueList[i] = trueList[i].lower()
        estimatedList[i] = estimatedList[i].lower()

        if trueList[i] in replaceDict.keys():
            trueList[i] = replaceDict[trueList[i]]

        if estimatedList[i] in replaceDict.keys():
            estimatedList[i] = replaceDict[estimatedList[i]]

    resultDict: Dict[str, List[int]] = {}  # key: name, value: [numCorrect, numFalse]
    totalErrors: int = 0
    for expected, actual in zip(trueList, estimatedList):
        correct = expected == actual
        if not correct:
            totalErrors += 1

        if expected not in resultDict.keys():
            resultDict[expected] = [1, 0] if correct else [0, 1]
        else:
            if correct:
                resultDict[expected][0] += 1
            else:
                resultDict[expected][1] += 1

    for name, trueFalse in resultDict.items():
        resultDict[name] = round(trueFalse[0] / (trueFalse[0] + trueFalse[1]) * 100, 2)

    sortedList: List[Tuple[str, str]] = sorted(resultDict.items(), key=lambda x: x[1], reverse=True)
    resultDict = {entry[0]: entry[1] for entry in sortedList}
    totalQuality = 100 - (totalErrors / len(trueList)) * 100
    return totalQuality, resultDict


def getNMostDifferentSpectraIndices(spectra: np.ndarray, n: int) -> List[int]:
    """
    Takes a set of spectra and returns the indices of the n spectra that are furthest apart from each other,
    in terms of the first two PCA components.
    :param spectra: (NxM) array of spec of M-1 spectra with N wavenumbers (wavenumbers in first column).
    :param n: Desired number of spectra to keep
    :return:
    """
    intensities = spectra[:, 1:]
    indices: List[int] = []
    pca: PCA = PCA(n_components=2)
    princComps: np.ndarray = pca.fit_transform(intensities.transpose())
    centers = k_means(princComps, n)[0]

    for i in range(n):
        distances = np.linalg.norm(princComps-centers[i, :], axis=1)
        indices.append(int(np.argmin(distances)))

    return indices
