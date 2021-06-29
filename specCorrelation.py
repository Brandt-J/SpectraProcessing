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
from scipy.stats import pearsonr
from enum import Enum
from typing import List

import functions as fn
from Preprocessing import processing as specProc
from cythonModules import corrCoeff


class Database(object):
    def __init__(self, title='DefaultDB'):
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
            curWavenums: np.ndarray = self._spectra[:, 0]
            if np.array_equal(spectrum[:, 0], curWavenums):
                addSpec: np.ndarray = spectrum[:, 1][:, np.newaxis]
                self._spectra = np.hstack((self._spectra, addSpec))
                self._spectraNames.append(name)
            else:
                print(f'remapping spectrum {name} to fitting wavenumbers')
                remappedSpec: np.ndarray = fn.remapSpectrumToWavenumbers(spectrum, curWavenums)
                self.addSpectrum(name, remappedSpec)

    def reduceSpecsToNWavenumbers(self, n: int) -> None:
        """
        Reduces the present spectra to only have n wavenumbers
        :param n: desired number of wavenumbers
        :return:
        """
        curWavenums: np.ndarray = self._spectra[:, 0]
        newSpecs: np.ndarray = np.zeros((n, self._spectra.shape[1]))
        newWavenums: np.ndarray = np.linspace(curWavenums.min(), curWavenums.max(), n)
        newSpecs[:, 0] = newWavenums
        for i in range(self.getNumberOfSpectra()):
            curSpec: np.ndarray = self._spectra[:, [0, i+1]]
            newSpecs[:, i+1] = fn.remapSpectrumToWavenumbers(curSpec, newWavenums)[:, 1]

        self._spectra = newSpecs

    def getWavenumbers(self) -> np.ndarray:
        return self._spectra[:, 0].copy()

    def getSpectrumOfIndex(self, index: int) -> np.ndarray:
        assert self._spectra is not None
        return self._spectra[:, [0, index+1]].copy()

    def getSpectrumNameOfIndex(self, index: int) -> str:
        assert self._spectra is not None
        assert self._spectra.shape[1] - 1 == len(self._spectraNames)
        return self._spectraNames[index]

    def getSpectrumOfName(self, name: str) -> np.ndarray:
        assert name in self._spectraNames, f'requested spectrum {name} not in database {self.title}'
        index: int = self._spectraNames.index(name)
        return self.getSpectrumOfIndex(index)

    def getSpectraNames(self) -> List[str]:
        return self._spectraNames

    def getNumberOfSpectra(self) -> int:
        numSpec: int = 0
        if self._spectra is not None:
            numSpec = self._spectra.shape[1] - 1
        return numSpec

    def getSpectra(self) -> np.ndarray:
        return self._spectra.copy()

    def getIndexOfSpectrumName(self, name: str) -> int:
        assert self._spectra is not None
        return self._spectraNames.index(name)

    def preprocessSpectra(self, baseline: bool = True, normalize: bool = True) -> None:
        for i in range(self._spectra.shape[1] - 1):
            if baseline:
                self._spectra[:, i + 1] -= specProc.als_baseline(self._spectra[:, i + 1], smoothness_param=1e6)
            if normalize:
                self._spectra[:, i + 1] = specProc.normalizeIntensities(self._spectra[:, i + 1])

    def removeSpectrumOfIndex(self, index: int) -> None:
        self._spectra = np.delete(self._spectra, index, axis=1)
        self._spectraNames.__delitem__(index)


class CorrelationMode(Enum):
    PEARSON = 0
    SFEC = 1

    def getName(self) -> str:
        if self.value == 0:
            name = "Database, Pearson"
        else:
            name = "Database, SFEC"
        return name


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
    sampleSpecs, refSpecs = fn.mapSpectrasetsToSameWavenumbers(spectra, refSpecs)

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
        # vector normalizatoin:
        intensities1 /= np.linalg.norm(intensities1)
        intensities2 /= np.linalg.norm(intensities2)
        corr = corrCoeff.sfec(intensities2, intensities1)

    assert corr != np.nan
    return corr
