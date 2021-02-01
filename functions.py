"""
SPECTRA PROCESSING
Copyright (C) 2020 Josef Brandt, University of Gothenborg.
<josef.brandt@gu.se>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program, see COPYING.
If not, see <https://www.gnu.org/licenses/>.
"""


from typing import List, Tuple, Dict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
from sklearn.preprocessing import StandardScaler


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
    maxIndex = np.argmin(np.abs(spectra[:, 0] - 2000))  # only go up to 2000 cm-1, above it's so unspecific...
    intensities = spectra[:maxIndex, 1:]
    intensities = StandardScaler().fit_transform(intensities.transpose())
    indices: List[int] = []
    pca: PCA = PCA(n_components=2, random_state=42)
    princComps: np.ndarray = pca.fit_transform(intensities)
    centers = k_means(princComps, n, random_state=42)[0]

    for i in range(n):
        distances = np.linalg.norm(princComps-centers[i, :], axis=1)
        indices.append(int(np.argmin(distances)))

    # FOR DEBUG DISPLAY
    # import matplotlib.pyplot as plt
    # plt.scatter(princComps[:, 0], princComps[:, 1], color='lightgray')
    # plt.scatter(princComps[indices, 0], princComps[indices, 1], color='black')
    # plt.xlabel('PC 1')
    # plt.ylabel('PC 2')
    # plt.title(f'Chosing {n} out of {princComps.shape[0]} spectra')
    # plt.show(block=True)

    return indices
