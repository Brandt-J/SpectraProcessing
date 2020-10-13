import numpy as np
from typing import List, TYPE_CHECKING
from scipy.signal import find_peaks, peak_prominences, savgol_filter
import matplotlib.pyplot as plt
if TYPE_CHECKING:
    from importData import Database


def getPETDescriptor() -> 'DescriptorSet':
    desc: DescriptorSet = DescriptorSet('PET')
    desc.add_descriptor(695, 733, 758)
    desc.add_descriptor(945, 970, 997)
    desc.add_descriptor(997, 1021, 1034)
    desc.add_descriptor(1264, 1301, 1328)
    desc.add_descriptor(1391, 1410, 1425)
    desc.add_descriptor(1660, 1744, 1800)
    desc.add_descriptor(1900, 1955, 2035)
    desc.add_descriptor(2830, 2908, 2930)
    desc.add_descriptor(2930, 2970, 3040)
    desc.add_descriptor(3400, 3433, 3480)
    return desc


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


class DescriptorLibrary(object):
    def __init__(self):
        super(DescriptorLibrary, self).__init__()
        self._descriptorSets: List['DescriptorSet'] = []

    def generate_from_specDatabase(self, database: 'Database') -> None:
        for i in range(database.getNumberOfSpectra()):
            desc: DescriptorSet = getDescriptorSetForSpec(database.getSpectrumNameOfIndex(i),
                                                          database.getSpectrumOfIndex(i))
            self._descriptorSets.append(desc)

    def apply_to_spectra(self, spectra: np.ndarray) -> List[str]:
        results: List[str] = []
        for i in range(spectra.shape[1]-1):
            maxCorr: float = 0.0
            bestHit: str = 'None'
            for desc in self._descriptorSets:
                corr = desc.get_correlation_to_spectrum(spectra[:, [0, i+1]])
                if corr > maxCorr:
                    bestHit = desc.name
                    maxCorr = corr

            results.append(bestHit)
        return results


class DescriptorSet(object):
    def __init__(self, name: str, threshold: float = 0.5):
        super(DescriptorSet, self).__init__()
        self.name = name
        self._threshold = threshold
        self._descriptors: List['TriangleDescriptor'] = []

    def apply_to_spectra(self, spectra: np.ndarray) -> List[bool]:
        equals: List[bool] = []
        for i in range(spectra.shape[1]-1):
            curCorr: float = self.get_correlation_to_spectrum(spectra[:, [0, i+1]])
            equals.append(curCorr > self._threshold)
        return equals

    def get_correlation_to_spectrum(self, spectrum: np.ndarray) -> float:
        corr: float = 0.0
        if len(self._descriptors) > 0:
            corrs: List[float] = []
            for desc in self._descriptors:
                newCorr = desc.get_correlation_to_spectrum(spectrum)
                if not np.isnan(newCorr):
                    corrs.append(newCorr)

            if len(corrs) > 0:
                corr = np.mean(corrs)

        return corr
    
    def add_descriptor(self, start: float, peak: float, end: float) -> None:
        newDescriptor: 'TriangleDescriptor' = TriangleDescriptor(start, peak, end)
        self._descriptors.append(newDescriptor)


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
        corr: float = np.nan
        startInd = np.argmin(np.abs(spectrum[:, 0] - self.start))
        peakInd = np.argmin(np.abs(spectrum[:, 0] - self.peak))
        endInd = np.argmin(np.abs(spectrum[:, 0] - self.end))

        if endInd - peakInd > 3 and peakInd - startInd > 3:

            if (endInd - startInd)%2 == 0:
                intensities: np.ndarray = np.append(np.linspace(0, 1, peakInd - startInd, endpoint=False),
                                                    np.linspace(1, 0, endInd - peakInd))
            else:
                intensities: np.ndarray = np.append(np.linspace(0, 1, peakInd - startInd),
                                                    np.linspace(1, 0, endInd - peakInd))

            specSection: np.ndarray = spectrum[startInd:endInd, 1].copy()
            specSection -= np.linspace(specSection[0], specSection[-1], endInd-startInd)
            if specSection.max() != 0.0:
                specSection /= specSection.max()

            corr = np.corrcoef(intensities, specSection)[0, 1]

        return corr
