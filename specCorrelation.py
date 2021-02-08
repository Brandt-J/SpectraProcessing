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
import processing as specProc


class Database(object):
    def __init__(self, title):
        self.title: str = title
        self._spectraNames: List[str] = []
        self._spectra: np.ndarray = None

    def addSpectrum(self, name: str, spectrum: np.ndarray) -> None:
        """
        :param name: name of spec
        :param spectrum: shape (N, 2) array of spectrum
        :return:
        """
        if self._spectra is None:
            self._spectra = spectrum
            self._spectraNames.append(name)
        else:
            if np.array_equal(spectrum[:, 0], self._spectra[:, 0]):
                addSpec: np.ndarray = spectrum[:, 1][:, np.newaxis]
                self._spectra = np.hstack((self._spectra, addSpec))
                self._spectraNames.append(name)
            else:
                print(f'remapping spectrum {name} to fitting wavenumbers')
                remappedSpec: np.ndarray = self._remapSpectrumToWavenumbers(spectrum)
                self.addSpectrum(name, remappedSpec)

    def getSpectrumOfIndex(self, index: int) -> np.ndarray:
        assert self._spectra is not None
        return self._spectra[:, [0, index+1]]

    def getSpectrumNameOfIndex(self, index: int) -> str:
        assert self._spectra is not None
        assert self._spectra.shape[1] - 1 == len(self._spectraNames)
        return self._spectraNames[index]

    def getSpectrumOfName(self, name: str) -> np.ndarray:
        assert name in self._spectraNames, f'requested spectrum {name} not in database {self.title}'
        index: int = self._spectraNames.index(name)
        return self.getSpectrumOfIndex(index)

    def getNumberOfSpectra(self) -> int:
        numSpec: int = 0
        if self._spectra is not None:
            numSpec = self._spectra.shape[1] - 1
        return numSpec

    def getSpectra(self) -> np.ndarray:
        return self._spectra

    def getIndexOfSpectrumName(self, name: str) -> int:
        assert self._spectra is not None
        return self._spectraNames.index(name)

    def preprocessSpectra(self) -> None:
        pass

    def removeSpectrumOfIndex(self, index: int) -> None:
        self._spectra = np.delete(self._spectra, index, axis=1)
        self._spectraNames.__delitem__(index)

    def _remapSpectrumToWavenumbers(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Takes a (N, 2) shape spectrum array and maps it to the currently present spectra.
        """
        wavenumbers = self._spectra[:, ]
        newSpec = np.zeros((len(wavenumbers), 2))
        newSpec[:, 0] = wavenumbers
        for i in range(len(wavenumbers)):
            clostestIndex = np.argmin(np.abs(spectrum[:, 0] - wavenumbers[i]))
            newSpec[i, 1] = spectrum[clostestIndex, 1]
        return newSpec


def correlate_spectra(spectra: np.ndarray, database: Database):
    refSpecs: np.ndarray = database.getSpectra()

    import time
    t0 = time.time()
    # preprocess ref spectra
    for i in range(refSpecs.shape[1]-1):
        refSpecs[:, i+1] -= specProc.als_baseline(refSpecs[:, i+1], smoothness_param=1e6)
        refSpecs[:, i+1] = specProc.normalizeIntensities(refSpecs[:, i+1])
    print(f"preprocessing took {round(time.time()-t0, 2)} seconds")

    results: List[str] = []
    t0 = time.time()
    sampleSpecs, refSpecs = mapSpectrasetsToSameWavenumbers(spectra, refSpecs)
    print(f"mapping spectra sets to each other took {round(time.time() - t0, 2)} seconds")

    t0 = time.time()
    for i in range(sampleSpecs.shape[1]-1):
        spec: np.ndarray = sampleSpecs[:, i+1].copy()
        spec -= specProc.als_baseline(spec, smoothness_param=1e6)
        spec = specProc.normalizeIntensities(spec)

        corrcoeffs: np.ndarray = np.zeros(refSpecs.shape[1]-1)
        for j in range(refSpecs.shape[1]-1):
            corrcoeffs[j] = np.corrcoef(spec, refSpecs[:, j+1])[0, 1]
        results.append(database.getSpectrumNameOfIndex(np.argmax(corrcoeffs)))
    print(f"specCorrelation took {round(time.time() - t0, 2)} seconds")

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
