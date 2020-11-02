import numpy as np
from typing import List
import time
import os

import importData as io
from specCorrelation import correlate_spectra
from descriptors import DescriptorLibrary
from handmadeDescriptors import handMadeDescLib
from classification import test_randForestClassifier
from functions import compareResultLists

t0 = time.time()
pathSampleSpec: str = r'Sample Spectra/sampleSpectra.npy'
pathSampleAssignments: str = r'Sample Spectra/origResults.txt'

regenerate: bool = True
if regenerate or not (os.path.exists(pathSampleSpec) and os.path.exists(pathSampleAssignments)):
    print('regenerating sample spectra from files...')
    origResults, testSpectra = io.get_test_spectra()
    np.savetxt(pathSampleAssignments, origResults, fmt='%s')
    np.save(pathSampleSpec, testSpectra)
else:
    origResults: List[str] = list(np.genfromtxt(pathSampleAssignments, dtype=str))
    testSpectra: np.ndarray = np.load(pathSampleSpec)

print(f'loading {len(origResults)} spectra took {time.time()-t0} seconds')


numSpectra: int = testSpectra.shape[1] - 1
database = io.get_database()

# descriptors: DescriptorLibrary = DescriptorLibrary()
# descriptors.generate_from_specDatabase(database, maxDescPerSet=200)
descriptors: DescriptorLibrary = handMadeDescLib
# descriptors.getDescriptorPlot().show()
descriptors.optimize_descriptorSets(maxDescriptorsPerSet=4)
# descriptors.getDescriptorPlot().show()

# presentPolyms = [descSet.name for descSet in descriptors._descriptorSets]  # TODO: If I want to keep that, use a public function....
# delIndices = []
# for i in range(database.getNumberOfSpectra()):
#     if database.getSpectrumNameOfIndex(i) not in presentPolyms:
#         delIndices.append(i)
#
# origResults: List[str] = [database.getSpectrumNameOfIndex(i) for i in range(database.getNumberOfSpectra())]
# for i in reversed(delIndices):
#     origResults.remove(origResults[i])
# dbSpecs: np.ndarray = np.delete(database.getSpectra(), delIndices, axis=1)

# clf, uniqueAssignments = test_randForestClassifier(descriptors.getCorrelationMatrixToSpectra(testSpectra), origResults)
#
t0 = time.time()
dbResults = correlate_spectra(testSpectra, database)
resultQualityDB, dbResultDict = compareResultLists(origResults, dbResults)
print(f'Spec correlation took {round(time.time()-t0, 2)} seconds, {round(resultQualityDB)} % correct hits')
print(dbResultDict)

t0 = time.time()
descriptorResults = descriptors.apply_to_spectra(testSpectra)
# prediction = clf.predict(descriptors.getCorrelationMatrixToSpectra(testSpectra))
# descriptorResults = [uniqueAssignments[i] for i in prediction]
resultQualityDesc, descResultDict = compareResultLists(origResults, descriptorResults)
print(f'Spectra Descriptor Application took {round(time.time()-t0, 2)} seconds, {round(resultQualityDesc)} % correct hits')
print(descResultDict)
