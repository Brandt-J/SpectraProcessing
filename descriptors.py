import numpy as np
from typing import List


def getPETDescriptor() -> 'DescriptorSet':
    desc: DescriptorSet = DescriptorSet('PET')
    desc.add_descriptor(TriangleDescriptor(695, 733, 758))
    desc.add_descriptor(TriangleDescriptor(945, 970, 997))
    desc.add_descriptor(TriangleDescriptor(997, 1021, 1034))
    desc.add_descriptor(TriangleDescriptor(1264, 1301, 1328))
    desc.add_descriptor(TriangleDescriptor(1391, 1410, 1425))
    desc.add_descriptor(TriangleDescriptor(1660, 1744, 1800))
    desc.add_descriptor(TriangleDescriptor(1900, 1955, 2035))
    desc.add_descriptor(TriangleDescriptor(2830, 2908, 2930))
    desc.add_descriptor(TriangleDescriptor(2930, 2970, 3040))
    desc.add_descriptor(TriangleDescriptor(3400, 3433, 3480))
    return desc


class DescriptorSet(object):
    def __init__(self, name: str, threshold: float = 0.5):
        super(DescriptorSet, self).__init__()
        self.name = name
        self._threshold = threshold
        self._descriptors: List['TriangleDescriptor'] = []

    def apply_to_spectra(self, spectra: np.ndarray) -> List[bool]:
        equals: List[bool] = []
        for i in range(spectra.shape[1]-1):
            curCorr: float = self._get_correlation_to_spectrum(spectra[:, [0, i+1]])
            equals.append(curCorr > self._threshold)
        return equals

    def _get_correlation_to_spectrum(self, spectrum: np.ndarray) -> float:
        corr: float = 0.0
        if len(self._descriptors) > 0:
            corr = np.mean(np.array([desc.get_correlation_to_spectrum(spectrum) for desc in self._descriptors]))
        return corr
    
    def add_descriptor(self, newDescriptor: 'TriangleDescriptor') -> None:
        self._descriptors.append(newDescriptor)


class TriangleDescriptor(object):
    """
    A Basic Spectral Descriptor.
    """
    def __init__(self, start: float = 0.0, peak: float = 0.5, end: float = 1.0):
        super(TriangleDescriptor, self).__init__()
        self.start: float = start
        self.peak: float = peak
        self.end: float = end

    def get_correlation_to_spectrum(self, spectrum: np.ndarray) -> float:
        """
        Spectrum section according to descriptor limits is cut out, normalized and correlated to a triangle shape
        determined by the descriptor limits.
        :param spectrum: (N, 2) shape array, first col = wavenumbers
        :return: correlationCoefficient
        """
        startInd = np.argmin(np.abs(spectrum[:, 0] - self.start))
        peakInd = np.argmin(np.abs(spectrum[:, 0] - self.peak))
        endInd = np.argmin(np.abs(spectrum[:, 0] - self.end))

        if (endInd - startInd)%2 == 0:
            intensities: np.ndarray = np.append(np.linspace(0, 1, peakInd - startInd, endpoint=False),
                                                np.linspace(1, 0, endInd - peakInd))
        else:
            intensities: np.ndarray = np.append(np.linspace(0, 1, peakInd - startInd),
                                                np.linspace(1, 0, endInd - peakInd))

        specSection: np.ndarray = spectrum[startInd:endInd, 1].copy()
        specSection -= np.linspace(specSection[0], specSection[-1], endInd-startInd)
        specSection /= specSection.max()
        try:
            return np.corrcoef(intensities, specSection)[0, 1]
        except ValueError:
            breakpoint()
