import numpy as np
from typing import List, TYPE_CHECKING
from scipy.signal import find_peaks, peak_prominences, savgol_filter
import matplotlib.pyplot as plt
from importData import specProc
if TYPE_CHECKING:
    from importData import Database
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def getDescriptorSetForSpec(specName: str, spec: np.ndarray, maxNumDescriptors: int = 10,
                            minDiff: int = 5) -> 'DescriptorSet':
    intensities = spec[:, 1].copy()
    filterSize = round(len(intensities)/100)
    if filterSize %2 == 0:
        filterSize += 1
    intensities = savgol_filter(intensities, filterSize, 3)

    intensities -= intensities.min()
    intensities /= intensities.max()

    plot = False
    if plot:
        plt.clf()
        plt.plot(intensities, alpha=0.5)

    desc: DescriptorSet = DescriptorSet(specName)
    peaks, _ = find_peaks(spec[:, 1], height=0.5)
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
            name: str = database.getSpectrumNameOfIndex(i)
            desc: 'DescriptorSet' = getDescriptorSetForSpec(database.getSpectrumNameOfIndex(i),
                                                            database.getSpectrumOfIndex(i),
                                                            maxNumDescriptors=maxDescPerSet)
            # desc: 'DescriptorSet' = getDescriptorSetForSpecIndexInSpectra(name, i, database.getSpectra())
            self._descriptorSets.append(desc)

    def add_descriptorSet(self, descSet: 'DescriptorSet') -> None:
        self._descriptorSets.append(descSet)

    def apply_to_spectra(self, spectra: np.ndarray) -> List[str]:
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
        return results

    def getTotalNumberOfDescriptors(self) -> np.ndarray:
        return np.sum([desc.getNumDescriptors() for desc in self._descriptorSets])

    def getCorrelationMatrixToSpectra(self, spectra: np.ndarray) -> np.ndarray:
        """
        Produces a feature matrix used for training classifiers.
        :param spectra: (N, M) shape array of spectra, first column wavenumbers, all others: intensities
        :return: feature matrix rows: Features, columns: Samples
        """
        numTotalDescriptors: int = self.getTotalNumberOfDescriptors()
        featureMat: np.ndarray = np.zeros((spectra.shape[1]-1, numTotalDescriptors))
        for i in range(spectra.shape[1]-1):
            spec: np.ndarray = spectra[:, [0, i+1]]
            correlations: List[np.ndarray] = []
            for descSet in self._descriptorSets:
                curCorrs: np.ndarray = descSet.get_correlations_to_spectrum(spec)
                correlations.append(curCorrs)
            featureMat[i, :] = np.concatenate(correlations)

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


class DescriptorSet(object):
    def __init__(self, name: str, threshold: float = 0.01):
        super(DescriptorSet, self).__init__()
        self.name = name
        self._threshold = threshold
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
                    print(f'skipping descriptor {desc.start}, {desc.end} of {self.name}')

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

    def get_correlation_to_spectrum(self, spectrum: np.ndarray) -> float:
        """
        Spectrum section according to descriptor limits is cut out, normalized and correlated to a triangle shape
        determined by the descriptor limits.
        :param spectrum: (N, 2) shape array, first col = wavenumbers
        :return: correlationCoefficient
        """
        corr: float = 0
        startInd = np.argmin(np.abs(spectrum[:, 0] - self.start))
        peakInd = np.argmin(np.abs(spectrum[:, 0] - self.peak))
        endInd = np.argmin(np.abs(spectrum[:, 0] - self.end))

        if endInd - peakInd > 2 and peakInd - startInd > 2:

            if (endInd - startInd)%2 == 0:
                intensities: np.ndarray = np.append(np.linspace(0, 1, peakInd - startInd, endpoint=False),
                                                    np.linspace(1, 0, endInd - peakInd))
            else:
                intensities: np.ndarray = np.append(np.linspace(0, 1, peakInd - startInd),
                                                    np.linspace(1, 0, endInd - peakInd))

            specSection: np.ndarray = spectrum[startInd:endInd, 1].copy()
            specSection -= np.linspace(specSection[0], specSection[-1], endInd-startInd)  # subtract baseline
            if specSection.max() != 0.0:
                specSection /= specSection.max()

            corr = np.corrcoef(intensities, specSection)[0, 1] ** 2

        assert not np.isnan(corr)
        return corr
