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
from typing import List, TYPE_CHECKING
from scipy.signal import find_peaks, peak_prominences, savgol_filter
import matplotlib.pyplot as plt
import processing as specProc
from cythonModules import corrCoeff
if TYPE_CHECKING:
    from importData import Database
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def getDescriptorSetForSpec(specName: str, spec: np.ndarray, maxNumDescriptors: int = 10,
                            minDiff: int = 5) -> 'DescriptorSet':
    intensities = spec[:, 1].copy()
    filterSize = round(len(intensities)/100)
    if filterSize % 2 == 0:
        filterSize += 1
    intensities = savgol_filter(intensities, filterSize, 3)

    intensities -= intensities.min()
    intensities /= intensities.max()

    plot = False
    if plot:
        plt.clf()
        plt.plot(intensities, alpha=0.5)

    desc: DescriptorSet = DescriptorSet(specName)
    peaks, _ = find_peaks(intensities, height=0.5)
    # remove peaks being too close to each other
    peaks = np.delete(peaks, np.argwhere(np.ediff1d(peaks) <= minDiff*2) + 1)
    prominences = peak_prominences(intensities, peaks)[0]  # i.e., peakHeights minus baseline...

    if len(peaks) > maxNumDescriptors:
        # take the maxNumDescriptors highest peaks
        peaks = peaks[np.argsort(prominences)[-maxNumDescriptors:][::-1]]

    lefts, rights = [], []
    thresh = 0.2
    for peak in peaks:
        peakInt: float = intensities[peak]

        leftFound: bool = False
        i = peak-minDiff
        if i < 0:
            lefts.append(0)
        else:
            while not leftFound:
                if intensities[i] <= thresh*peakInt or intensities[i] > intensities[i+1] or i <= 0:
                    lefts.append(i)
                    leftFound = True
                else:
                    i -= 1

        rightFound: bool = False
        i = peak+minDiff
        if i >= len(intensities):
            rights.append(len(intensities)-1)
        else:
            while not rightFound:
                if intensities[i] <= thresh*peakInt or intensities[i] > intensities[i-1] or i >= len(intensities)-1:
                    rights.append(i+1)
                    rightFound = True
                else:
                    i += 1

    for start, peak, end in zip(lefts, peaks, rights):
        if spec[peak, 1] > spec[start, 1] and spec[peak, 1] > spec[end, 1]:
            desc.add_descriptor(spec[start, 0], spec[peak, 0], spec[end, 0])
            peakshape = np.array([start, peak, end])
            if plot:
                plt.plot(peakshape, intensities[peakshape], alpha=0.8)
                plt.title(specName)
    if plot:
        plt.waitforbuttonpress()

    return desc


def getDescriptorSetForSpecIndexInSpectra(specName: str, specIndex: int, spectra: np.ndarray) -> 'DescriptorSet':
    descSet: DescriptorSet = DescriptorSet(specName)
    intensities: np.ndarray = spectra[:, 1:].copy()
    filterSize = round(len(intensities) / 50)
    if filterSize % 2 == 0:
        filterSize += 1

    # first, normalize them all:
    for i in range(intensities.shape[1]):
        intensities[:, i] = savgol_filter(intensities[:, i], filterSize, 3)
        intensities[:, i] -= specProc.als_baseline(intensities[:, i], smoothness_param=1e7)
        intensities[:, i] /= intensities[:, i].max()

    curIntensity: np.ndarray = intensities[:, specIndex]
    numSpecs: int = intensities.shape[1]
    otherIndices: np.ndarray = np.delete(np.arange(numSpecs), specIndex)
    avgOthers: np.ndarray = np.mean(intensities[:, otherIndices], axis=1)
    slopeOfOthers: np.ndarray = np.abs(np.gradient(avgOthers))
    possPeaks: np.ndarray = curIntensity.copy()
    possPeaks[slopeOfOthers > slopeOfOthers.max() * 0.3] = 0

    plt.clf()
    plt.plot(avgOthers, alpha=0.5)
    plt.plot(curIntensity + 0.5, alpha=0.5)
    plt.plot(slopeOfOthers*10 - 0.2)
    plt.plot(possPeaks + 1)
    plt.title(specName)
    plt.waitforbuttonpress()

    return descSet


def descriptorOverlapsWithSpectrum(descriptor: 'TriangleDescriptor', specToTest: np.ndarray) -> bool:
    """
    Checks, if the descriptors start, peak and end are within the range of the spectrum to test.
    :param descriptor: Triangle Descriptor
    :param specToTest: (N, 2) shape spectrum, wavenumbers in first column
    :return: True if fully overlaps, else False
    """
    return descriptor.start >= specToTest[0, 0] and descriptor.end <= specToTest[-1, 0]


class DescriptorLibrary(object):
    def __init__(self):
        super(DescriptorLibrary, self).__init__()
        self._descriptorSets: List['DescriptorSet'] = []

    def generate_from_specDatabase(self, database: 'Database', maxDescPerSet: int = 10) -> None:
        for i in range(database.getNumberOfSpectra()):
            desc: 'DescriptorSet' = getDescriptorSetForSpec(database.getSpectrumNameOfIndex(i),
                                                            database.getSpectrumOfIndex(i),
                                                            maxNumDescriptors=maxDescPerSet)

            self._descriptorSets.append(desc)

    def add_descriptorSet(self, descSet: 'DescriptorSet') -> None:
        self._descriptorSets.append(descSet)

    def apply_to_spectra(self, spectra: np.ndarray) -> List[str]:
        """
        Takes array of spectra and return list of classifications.
        :param spectra: shape (N, M) array with N wavenumbers and M-1 samples. First Column is Wavenumbers
        :return: List of Spectra Assignments
        """
        self._setUpDescriptorsToWavenumbers(spectra[:, 0])
        results: List[str] = []
        for i in range(spectra.shape[1]-1):
            corrs: List[float] = []
            maxCorr: float = 0.0
            bestHit: str = 'None'
            for desc in self._descriptorSets:
                corr = desc.get_mean_correlation_to_spectrum(spectra[:, [0, i+1]])
                corrs.append(corr)
                if corr > maxCorr:
                    bestHit = desc.name
                    maxCorr = corr

            results.append(bestHit)
        self._unsetDescriptorsFromWavenumbers()
        return results

    def getTotalNumberOfDescriptors(self) -> int:
        return int(np.sum([desc.getNumDescriptors() for desc in self._descriptorSets]))

    def getCorrelationMatrixToSpectra(self, spectra: np.ndarray, useSFEC: bool = False) -> np.ndarray:
        """
        Produces a feature matrix used for training classifiers.
        :param spectra: (N, M) shape array of spectra, first column wavenumbers, all others: intensities
        :param useSFEC: Whether to use Squared First-Difference Euclidean Cosine correlation or not (in that case, pearson correlation is used)
        :return: feature matrix rows: Features, columns: Samples
        """
        numSpectra: int = spectra.shape[1]-1
        numTotalDescriptors: int = self.getTotalNumberOfDescriptors()
        self._setUpDescriptorsToWavenumbers(spectra[:, 0])

        featureMat: np.ndarray = np.zeros((numSpectra, numTotalDescriptors))
        allDescs: List['TriangleDescriptor'] = self._getAllDescriptors()
        for i, desc in enumerate(allDescs):
            corrs = np.zeros(numSpectra)
            specSection = spectra[desc.startInd:desc.endInd, 1:]

            if desc.endInd - desc.peakInd > 2 and desc.peakInd - desc.startInd > 2:
                if useSFEC:
                    intens1 = desc.intensities / np.linalg.norm(desc.intensities)
                    intens2 = specSection / np.linalg.norm(specSection)
                    corrs = corrCoeff.getCorrelationCoefficients(intens1, intens2, useSFEC=True)
                else:
                    corrs = corrCoeff.getCorrelationCoefficients(desc.intensities, specSection, useSFEC=False)
                featureMat[:, i] = corrs

        self._unsetDescriptorsFromWavenumbers()
        return featureMat

    def optimize_descriptorSets(self, maxDescriptorsPerSet: int = 5) -> None:
        """Optimize in the sense that each descriptor Set only has descriptors for peaks that are as far as possible
        away from the peaks of all other descriptor Sets"""

        for i, descSet in enumerate(self._descriptorSets):
            if descSet.getNumDescriptors() > maxDescriptorsPerSet:
                otherDescSets: List['DescriptorSet'] = [dset for j, dset in enumerate(self._descriptorSets) if j != i]
                curPeaks: np.ndarray = np.array([desc.peak for desc in descSet.getDescriptors()])
                otherPeaks: np.ndarray = np.array([])
                for dset in otherDescSets:
                    otherPeaks = np.append(otherPeaks, np.array([desc.peak for desc in dset.getDescriptors()]))

                distances: np.ndarray = np.array([np.abs(otherPeaks - peak).min() for peak in curPeaks])
                sortInd: np.ndarray = np.argsort(distances)[::-1]  # we want max distance first
                indicesToRemove: np.ndarray = np.sort(sortInd[maxDescriptorsPerSet:])[::-1]  # from highest to lowest
                # print(f'Desc Set {descSet.name}: min Distance before: {distances[sortInd[-1]]} and after: {distances[sortInd[maxDescriptorsPerSet]]}')
                for ind in indicesToRemove:
                    descSet.remove_descriptor_of_index(ind)

    def getDescriptorPlot(self) -> 'Figure':
        fig: 'Figure' = plt.figure()
        ax: 'Axes' = fig.add_subplot()

        colorCycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        tickLength: float = 1
        for i, descSet in enumerate(self._descriptorSets):
            for j, desc in enumerate(descSet.getDescriptors()):
                if j == 0:
                    ax.plot([desc.peak, desc.peak], [(i+1)*tickLength, i*tickLength], color=colorCycle[i], label=descSet.name)
                else:
                    ax.plot([desc.peak, desc.peak], [(i+1)*tickLength, i*tickLength], color=colorCycle[i])
        ax.legend()
        return fig

    def _setUpDescriptorsToWavenumbers(self, wavenums: np.ndarray) -> None:
        for descSet in self._descriptorSets:
            for desc in descSet.getDescriptors():
                desc.set_to_wavenumbers(wavenums)

    def _unsetDescriptorsFromWavenumbers(self) -> None:
        for descSet in self._descriptorSets:
            for desc in descSet.getDescriptors():
                desc.unset_from_wavenumbers()

    def _getAllDescriptors(self) -> List['TriangleDescriptor']:
        allDesc: List['TriangleDescriptor'] = []
        for descSet in self._descriptorSets:
            for desc in descSet.getDescriptors():
                allDesc.append(desc)
        return allDesc


class DescriptorSet(object):
    """
    Set of descriptors describing a particular polymer tyoe.
    """
    def __init__(self, name: str):
        super(DescriptorSet, self).__init__()
        self.name = name
        self._descriptors: List['TriangleDescriptor'] = []

    def get_mean_correlation_to_spectrum(self, spectrum: np.ndarray) -> float:
        return np.mean(self.get_correlations_to_spectrum(spectrum))

    def get_correlations_to_spectrum(self, spectrum: np.ndarray) -> np.ndarray:
        """
        Calculates correlation vector
        :param spectrum: (N, 2) shape np array (wavenumbers in first column)
        :return:
        """
        corrs: List[float] = [np.nan]
        if len(self._descriptors) > 0:
            corrs = []
            for desc in self._descriptors:
                if descriptorOverlapsWithSpectrum(desc, spectrum):
                    newCorr = desc.get_correlation_to_spectrum(spectrum)
                else:
                    newCorr = 0.0
                    # print(f'skipping descriptor {desc.start}, {desc.end} of {self.name}')

                corrs.append(newCorr)
        else:
            print('No descriptors on descriptor set', self.name)
        assert len(corrs) == len(self._descriptors)
        return np.array(corrs)

    def add_descriptor(self, start: float, peak: float, end: float) -> None:
        newDescriptor: 'TriangleDescriptor' = TriangleDescriptor(start, peak, end)
        self._descriptors.append(newDescriptor)

    def remove_descriptor_of_index(self, index: int) -> None:
        self._descriptors.remove(self._descriptors[index])

    def getNumDescriptors(self) -> int:
        return len(self._descriptors)

    def getDescriptors(self) -> List['TriangleDescriptor']:
        return self._descriptors


class TriangleDescriptor(object):
    """
    A Basic Spectral Descriptor.
    """
    def __init__(self, start: float = 0.0, peak: float = 0.5, end: float = 1.0):
        super(TriangleDescriptor, self).__init__()
        self.start: float = start  # in Wavenumbers
        self.peak: float = peak  # in Wavenumbers
        self.end: float = end  # in Wavenumbers

        self.startInd: int = np.nan
        self.peakInd: int = np.nan
        self.endInd: int = np.nan
        self.intensities: np.ndarray = None

    def set_to_wavenumbers(self, wavenumbers: np.ndarray) -> None:
        self.startInd = np.argmin(np.abs(wavenumbers - self.start))
        self.peakInd = np.argmin(np.abs(wavenumbers - self.peak))
        self.endInd = np.argmin(np.abs(wavenumbers - self.end))

        if (self.endInd - self.startInd) % 2 == 0:
            self.intensities = np.append(np.linspace(0, 1, self.peakInd - self.startInd, endpoint=False),
                                                np.linspace(1, 0, self.endInd - self.peakInd))
        else:
            self.intensities = np.append(np.linspace(0, 1, self.peakInd - self.startInd),
                                                np.linspace(1, 0, self.endInd - self.peakInd))

    def unset_from_wavenumbers(self) -> None:
        self.startInd: int = np.nan
        self.peakInd: int = np.nan
        self.endInd: int = np.nan
        self.intensities: np.ndarray = None

    # def get_correlation_to_spectrum(self, spectrum: np.ndarray) -> float:
    #     """
    #     Spectrum section according to descriptor limits is cut out, normalized and correlated to a triangle shape
    #     determined by the descriptor limits.
    #     :param spectrum: (N, 2) shape array, first col = wavenumbers
    #     :return: correlationCoefficient
    #     """
    #     corr: float = 0
    #     assert self._intensities is not None, f'Descriptor was not yet set to wavenumbers!'
    #     if self._endInd - self._peakInd > 2 and self._peakInd - self._startInd > 2:
    #         specSection: np.ndarray = spectrum[self._startInd:self._endInd, 1].copy()
    #         # the following could seemingly be omitted, but have to test it...
    #         specSection -= np.linspace(specSection[0], specSection[-1], self._endInd-self._startInd)  # subtract baseline
    #
    #         if not np.all(specSection == 0):
    #             # scipy's version is faster than the one from numpy, result are the same
    #             corr = pearsonr(self._intensities, specSection)[0]
    #         assert not np.isnan(corr)
    #
    #     return corr
