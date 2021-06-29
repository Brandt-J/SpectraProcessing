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
from typing import List, Tuple, Dict, Set
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import k_means
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import importData as io


def compareResultLists(trueList: List[str], estimatedList: List[str]) -> Tuple[float, float, float, float]:
    """
    Compares two assignment lists.
    :param trueList: List of true assignments
    :param estimatedList: List of estimated assignments
    :returns: Tuple(agreement in percent, Dictionary of agreements in percent per type of assignment)
    """
    assert len(trueList) == len(estimatedList)
    # Do some replacements to facilitate evaluation
    replaceDict: dict = {'ldpe': 'pe',
                         'hdpe': 'pe'}
    plastError = getPlasticError(trueList, estimatedList)
    for i in range(len(trueList)):
        trueList[i] = trueList[i].lower()
        estimatedList[i] = estimatedList[i].lower()
        if trueList[i] in replaceDict.keys():
            trueList[i] = replaceDict[trueList[i]]
        if estimatedList[i] in replaceDict.keys():
            estimatedList[i] = replaceDict[estimatedList[i]]

    # remove unknowns (we are not interested in statistics about them)
    cleanTrueList, cleanEstimateList = [], []
    for i, entry in enumerate(trueList):
        if entry != "unknown":
            cleanTrueList.append(entry)
            cleanEstimateList.append(estimatedList[i])
    trueList = cleanTrueList
    estimatedList = cleanEstimateList

    uniques: List[str] = list(np.unique(trueList))
    report: Dict[str, Dict] = classification_report(trueList, estimatedList, output_dict=True, zero_division=0)
    precisionDict: Dict[str, float] = {}
    recallDict: Dict[str, float] = {}
    for cls, results in report.items():
        if cls in uniques:
            precisionDict[cls] = round(results["precision"], 2)
            recallDict[cls] = round(results["recall"], 2)

    avgPrecision = np.mean(list(precisionDict.values())) * 100
    avgRecall = np.mean(list(recallDict.values())) * 100
    if (avgRecall + avgPrecision) == 0:
        avgF1 = 0
    else:
        avgF1 = 2*(avgPrecision*avgRecall) / (avgPrecision + avgRecall)
    return avgPrecision, avgRecall, avgF1, plastError


def getPlasticError(trueList: List[str], estimatedList: List[str]) -> float:
    nonPlastNames: Set[str] = set(io.getNonPlasticNames() + ["unknown"])
    from collections import Counter
    # print("True unknowns:", Counter(trueList).get("unknown"), "Estimated unknowns:", Counter(estimatedList).get("unknown"))
    plastCountTrue, plastCountEstimated = 0, 0
    for true, estim in zip(trueList, estimatedList):
        if true not in nonPlastNames:
            plastCountTrue += 1
        if estim not in nonPlastNames:
            plastCountEstimated += 1
    error: float = round((plastCountEstimated - plastCountTrue) / plastCountTrue * 100)
    # print("TruePlastCount:", plastCountTrue, "Estimated PlastCount", plastCountEstimated, "PlastError:", error)
    return error


def getNMostDifferentSpectra(assignments: List[str], spectra: np.ndarray, n: int) -> Tuple[List[str], np.ndarray]:
    """
    Takes a set of spectra and returns the indices of the n spectra that are furthest apart from each other,
    in terms of the first two PCA components.
    :param assignments: List of M-1 spectra
    :param spectra: (NxM) array of spec of M-1 spectra with N wavenumbers (wavenumbers in first column).
    :param n: Desired number of spectra to keep
    :return: Tuple[shortened AssignmentList, shortened Spectra array]
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

    assignments = [assignments[i] for i in indices]
    indices = [0] + [i + 1 for i in indices]
    spectra = spectra[:, indices]
    assert len(assignments) == spectra.shape[1] - 1
    return assignments, spectra


def mapSpectrasetsToSameWavenumbers(set1: np.ndarray, set2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matches two sets of spectra to each other, so that both have the same wavenumber axis (the shorter set is used)
    :param set1: np.ndarray shape(N, M) with N wavenumbers and M-1 spectra, wavenumbers in first column
    :param set2: np.ndarray shape(K, L) with K wavenumbers and L-1 spectra, wavenumbers in first column
    :return: Tuple of set1 and set2, with same wavenumber axis.
    """
    shorterWavenums, longerWavenums = set1[:, 0], set2[:, 0]
    set1IsShorter: bool = True
    if len(shorterWavenums) > len(longerWavenums):
        shorterWavenums, longerWavenums = longerWavenums, shorterWavenums
        set1IsShorter = False

    newSet1: np.ndarray = set1 if set1IsShorter else set2
    newSet2: np.ndarray = np.zeros((set1.shape[0], set2.shape[1])) if set1IsShorter else np.zeros((set2.shape[0], set1.shape[1]))
    newSet2[:, 0] = shorterWavenums

    for i, num in enumerate(shorterWavenums):
        correspondInd = np.argmin(np.abs(longerWavenums - num))
        newSet2[i, :] = set2[correspondInd, :] if set1IsShorter else set1[correspondInd, :]

    returnTuple: Tuple[np.ndarray, np.ndarray] = (newSet1, newSet2) if set1IsShorter else (newSet2, newSet1)
    assert returnTuple[0].shape[1] == set1.shape[1] and returnTuple[1].shape[1] == set2.shape[1]
    assert returnTuple[0].shape[0] == returnTuple[1].shape[0] == len(shorterWavenums)
    return returnTuple


def remapSpecArrayToWavenumbers(spectra: np.ndarray, wavenumbers: np.ndarray) -> np.ndarray:
    """
    Takes a spectrum array and maps it to the currently present spectra.
    :param spectra: (N, M) shape array of M-1 spectra with wavenumbs in first column
    :param wavenumbers: The wavenumbers to map to. If None, the wavenumbers of the currently present spectra set
    is used.
    :return: shape (L, M) shape spectrum with new wavenumber axis
    """
    newSpecs: np.ndarray = np.zeros((len(wavenumbers), spectra.shape[1]))
    newSpecs[:, 0] = wavenumbers
    for i in range(spectra.shape[1]-1):
        newSpecs[:, i+1] = remapSpectrumToWavenumbers(spectra[:, [0, i+1]], wavenumbers)[:, 1]
    return newSpecs


def remapSpectrumToWavenumbers(spectrum: np.ndarray, wavenumbers: np.ndarray = None) -> np.ndarray:
    """
    Takes a spectrum array and maps it to the currently present spectra.
    :param spectrum: (N, 2) shape spectrum with wavenumbs in first column
    :param wavenumbers: The wavenumbers to map to. If None, the wavenumbers of the currently present spectra set
    is used.
    :return: shape (M, 2) shape spectrum with new wavenumber axis
    """
    newSpec = np.zeros((len(wavenumbers), 2))
    newSpec[:, 0] = wavenumbers
    for i in range(len(wavenumbers)):
        clostestIndex = np.argmin(np.abs(spectrum[:, 0] - wavenumbers[i]))
        newSpec[i, 1] = spectrum[clostestIndex, 1]
    return newSpec
