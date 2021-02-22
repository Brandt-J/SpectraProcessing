from typing import *
from enum import Enum
from collections import Counter

from importData import *
from distort import append_n_distorted_copies
from processing import normalizeIntensities
from functions import remapSpecArrayToWavenumbers


class TestSpectra(object):
    pathSpectra: str = r'TestData/testSpectra.npy'
    pathAssignments: str = r'TestData/testAssignments.txt'
    pathTypes: str = r'TestData/testTypes.txt'

    def __init__(self):
        super(TestSpectra, self).__init__()
        self._spectra: np.ndarray = None
        self._assignments: List[str] = []
        self._types: List[MatType] = []

    def normalizeSpectra(self) -> None:
        for i in range(self.getNumberOfSpectra()):
            self._spectra[:, i+1] = normalizeIntensities(self._spectra[:, i+1])

    def mapSpectraToWavenumbers(self, newWavenumbers: np.ndarray) -> None:
        self._spectra = remapSpecArrayToWavenumbers(self._spectra, newWavenumbers)

    def getPlasticContent(self) -> float:
        assert len(self._types) > 0, 'Requesting plastic content from empty type list'
        count: Counter = Counter(self._types)
        if MatType.NON_PLASTIC not in count.elements():
            content = 1.0
        elif MatType.PLASTIC not in count.elements():
            content = 0.0
        else:
            content = count.get(MatType.PLASTIC) / (count.get(MatType.NON_PLASTIC) + count.get(MatType.PLASTIC))
        return content

    def getAllSpectra(self) -> np.ndarray:
        return self._spectra.copy()

    def getAllAssignments(self) -> List[str]:
        return self._assignments.copy()

    def getPolymerSpectra(self) -> np.ndarray:
        polymspecs: List[np.ndarray] = [self._spectra[:, 0]]
        assert Counter(self._types).get(MatType.PLASTIC) > 0, 'NO PLASTIC SPECTRA PRESENT'
        for i, mtype in enumerate(self._types):
            if mtype == MatType.PLASTIC:
                polymspecs.append(self._spectra[:, i+1])
        return np.array(polymspecs).transpose()

    def getPolymerAssignments(self) -> List[str]:
        assert Counter(self._types).get(MatType.PLASTIC) > 0, 'NO PLASTIC SPECTRA PRESENT'

        return [self._assignments[i] for i, mtype in enumerate(self._types) if mtype == MatType.PLASTIC]

    def getNumberOfSpectra(self) -> int:
        return len(self._assignments)

    def getNumberOfPlastics(self) -> int:
        counter = Counter(self._types)
        numPlast = counter.get(MatType.PLASTIC)
        if numPlast is None:
            numPlast = 0
        return numPlast

    def getNumberOfNonPlastics(self) -> int:
        counter = Counter(self._types)
        numNonPlast = counter.get(MatType.NON_PLASTIC)
        if numNonPlast is None:
            numNonPlast = 0
        return numNonPlast

    def loadFromNPY(self) -> None:
        if os.path.exists(self.pathSpectra) and os.path.exists(self.pathAssignments) and os.path.exists(self.pathTypes):
            specs = np.load(self.pathSpectra)
            assignments = list(np.genfromtxt(self.pathAssignments, dtype=str))
            types = list(np.genfromtxt(self.pathTypes, dtype=str))
            assert len(types) == len(assignments) == specs.shape[1]-1, 'Inconsistent size of input data!'
            self._types = [MatType.fromString(i) for i in types]
            self._assignments = assignments
            self._spectra = specs
        else:
            raise FileNotFoundError(f"Not all files found, check for these files:\n"
                                    f"{self.pathSpectra}\n"
                                    f"{self.pathAssignments}\n"
                                    f"{self.pathTypes}")

    def saveToNPY(self) -> None:
        np.savetxt(self.pathAssignments, self._assignments, fmt='%s')
        np.savetxt(self.pathTypes, [i.toString() for i in self._types], fmt='%s')
        np.save(self.pathSpectra, self._spectra)

    def generateFromSampleDir(self, maxSpecPerFolder: int = 1e6, saveToDisk: bool = True) -> None:
        """
        :param maxSpecPerFolder: Max number of Spectra per folder
        :param saveToDisk: Whether or not to save direclty to disk for faster loading next time
        """
        # TODO: Currently only plastic spectra present!! Upgrade with non-plastic spectra and include plastic content
        self._assignments, self._spectra = load_specCSVs_from_subfolders(sampleDirectory, maxSpecPerFolder)
        self._types = [MatType.PLASTIC] * len(self._assignments)
        if saveToDisk:
            self.saveToNPY()

    def generateFromRefSpecs(self, plasticContent: float, maxPlastTypes: int = 1e6, numVariations: int = 1,
                             unknownNonPlast: bool = True, saveToDisk: bool = True) -> None:
        """
        Read in reference spectra, create distorted variations and mix to a given plastic content.
        :param plasticContent: Desired plastic content (0.0 to 1.0)
        :param maxPlastTypes: Maximum number of different plastic types
        :param numVariations: Number of Variations per spectrum
        :param unknownNonPlast: If True, all non plastics will be named "unknown"
        :param saveToDisk: Whether or not to save direclty to disk for faster loading next time
        :return:
        """
        # Load Plastic spectra
        plastAssignments, plastSpecs = load_specCSVs_from_directory(refPlasticDirectory)
        numPlastSpecs = len(plastAssignments)
        if numPlastSpecs > maxPlastTypes:
            plastAssignments, plastSpecs = getNMostDifferentSpectra(plastAssignments, plastSpecs, maxPlastTypes)

        # Create plastic spec variations
        plastSpecs = append_n_distorted_copies(plastSpecs, numVariations-1)
        plastAssignments *= numVariations
        types = [MatType.PLASTIC] * len(plastAssignments)
        numPlastSpecs = len(plastAssignments)

        # Get the right number of non-plastic specs
        # ratio = plast / (plast + non-plast)
        # -> ratio * plast + ratio * non-plast = plast
        # -> non-plast = (plast - ratio*plast) / ratio
        numNonPlastSpecs = round((numPlastSpecs - numPlastSpecs * plasticContent) / plasticContent)
        if numNonPlastSpecs == 0:
            self._spectra = plastSpecs
            self._assignments = plastAssignments
            self._types = types
        else:
            nonPlastAssignments, nonPlastSpecs = load_specCSVs_from_directory(refNonPlasticDirectory)
            numSpecs = len(nonPlastAssignments)
            numVariationsRequired = int(round(numNonPlastSpecs / numSpecs))
            if numVariationsRequired > 1:
                nonPlastSpecs = append_n_distorted_copies(nonPlastSpecs, numVariationsRequired-1)
                nonPlastAssignments *= numVariationsRequired
            elif numVariationsRequired < 1:
                nonPlastAssignments, nonPlastSpecs = getNMostDifferentSpectra(nonPlastAssignments, nonPlastSpecs,
                                                                              numNonPlastSpecs)
            finalNumNonPlastic = len(nonPlastAssignments)
            if unknownNonPlast:
                nonPlastAssignments = ["unknown"] * len(nonPlastAssignments)
            types += [MatType.NON_PLASTIC] * finalNumNonPlastic

            plastSpecs, nonPlastSpecs = mapSpectrasetsToSameWavenumbers(plastSpecs, nonPlastSpecs)
            self._spectra = np.hstack((plastSpecs, nonPlastSpecs[:, 1:]))
            self._assignments = plastAssignments + nonPlastAssignments
            self._types = types
            assert len(self._assignments) == len(self._types) == self._spectra.shape[1]-1
        print(f'requested ratio: {plasticContent}, actual is {self.getPlasticContent()}')
        if saveToDisk:
            self.saveToNPY()


class MatType(Enum):
    PLASTIC = 0
    NON_PLASTIC = 1

    @classmethod
    def fromString(cls, name: str) -> 'MatType':
        if name == 'plastic':
            return MatType.PLASTIC
        elif name == 'non_plastic':
            return MatType.NON_PLASTIC

    def toString(self) -> str:
        if self.value == 0:
            string = 'plastic'
        else:
            string = 'non_plastic'
        return string
