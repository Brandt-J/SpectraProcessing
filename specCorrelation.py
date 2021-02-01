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


import numpy as np
from typing import List, Tuple
from importData import Database, specProc


def correlate_spectra(spectra: np.ndarray, database: Database):
    refSpecs: np.ndarray = database.getSpectra()

    # preprocess ref spectra
    for i in range(refSpecs.shape[1]-1):
        refSpecs[:, i+1] -= specProc.als_baseline(refSpecs[:, i+1], smoothness_param=1e6)
        refSpecs[:, i+1] = specProc.normalizeIntensities(refSpecs[:, i+1])

    results: List[str] = []
    sampleSpecs, refSpecs = mapSpectrasetsToSameWavenumbers(spectra, refSpecs)
    for i in range(sampleSpecs.shape[1]-1):
        spec: np.ndarray = sampleSpecs[:, i+1].copy()
        spec -= specProc.als_baseline(spec, smoothness_param=1e6)
        spec = specProc.normalizeIntensities(spec)

        corrcoeffs: np.ndarray = np.zeros(refSpecs.shape[1]-1)
        for j in range(refSpecs.shape[1]-1):
            corrcoeffs[j] = np.corrcoef(spec, refSpecs[:, j+1])[0, 1]
        results.append(database.getSpectrumNameOfIndex(np.argmax(corrcoeffs)))

    return results


def mapSpectrasetsToSameWavenumbers(set1: np.ndarray, set2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Matches two sets of spectra to each other, so that both have the same wavenumber axis (the shorter set is used)
    :param set1: np.ndarray shape(N, M) with N wavenumbers and M-1 spectra, wavenumbers in first column
    :param set2: np.ndarray shape(K, L) with K wavenumbers and L-1 spectra, wavenumbers in first column
    :return:
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
