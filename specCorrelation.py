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
from functions import mapSpectrasetsToSameWavenumbers
from scipy.stats import pearsonr
from enum import Enum
from typing import List
import processing as specProc
from cythonModules import corrCoeff


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
        for i in range(self._spectra.shape[1] - 1):
            self._spectra[:, i + 1] -= specProc.als_baseline(self._spectra[:, i + 1], smoothness_param=1e6)
            self._spectra[:, i + 1] = specProc.normalizeIntensities(self._spectra[:, i + 1])

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


class CorrelationMode(Enum):
    PEARSON = 0
    SFEC = 1


def correlate_spectra(spectra: np.ndarray, database: Database, corrMode: CorrelationMode = CorrelationMode.PEARSON,
                      cutoff: float = 0.75, preproc: bool = False) -> List[str]:
    """
    Takes a set of spectra and a database and returns a List of names of spectra with best hit
    :param spectra: NxM spectra array of M-1 spectra with N wavenumbers (wavenumbers in first column)
    :param database: The database to use
    :param corrMode: Desired mode of correlation
    :param cutoff: The cutoff for correlation, under which the procedure returns "unknwon" for the spectrum
    :param preproc: Whether to apply preprocessing (i.e., baseline removal and normalization) to the spectra.
    :return:
    """
    spectra = spectra.copy()
    refSpecs: np.ndarray = database.getSpectra()
    results: List[str] = []
    sampleSpecs, refSpecs = mapSpectrasetsToSameWavenumbers(spectra, refSpecs)

    for i in range(sampleSpecs.shape[1]-1):
        spec: np.ndarray = sampleSpecs[:, i+1].copy()
        if preproc:
            spec -= specProc.als_baseline(spec, smoothness_param=1e6)
            spec = specProc.normalizeIntensities(spec)
            spec = specProc.smooth(spec[np.newaxis, :], windowSize=21)[0]

        corrcoeffs: np.ndarray = np.zeros(refSpecs.shape[1]-1)
        for j in range(refSpecs.shape[1]-1):
            corrcoeffs[j] = getCorrelation(spec, refSpecs[:, j+1], corrMode)

        highestCorr = np.max(corrcoeffs)
        if highestCorr >= cutoff:
            maxIndex = int(np.where(corrcoeffs == highestCorr)[0])
            result = database.getSpectrumNameOfIndex(maxIndex)
        else:
            result = 'unknown'
        results.append(result)

    return results


def getCorrelation(intensities1: np.ndarray, intensities2: np.ndarray, mode: CorrelationMode) -> float:
    """
    Takes two intenity vectors and returns their correlation measure, according to the specified mode
    :param intensities1: shape (N) array of test data
    :param intensities2: shape (N) array of reference data
    :param mode: the desired correlation mode
    :return:
    """
    assert len(intensities1) == len(intensities2)
    corr: float = np.nan
    if mode == CorrelationMode.PEARSON:
        corr = pearsonr(intensities1, intensities2)[0]
    elif mode == CorrelationMode.SFEC:
        corr = corrCoeff.sfec(intensities1, intensities2)

    assert corr != np.nan
    return corr
