import numpy as np
from typing import List
import time
from collections import Counter
import matplotlib.pyplot as plt

import importData as io
from specCorrelation import correlate_spectra
from descriptors import DescriptorLibrary
from handmadeDescriptors import handMadeDescLib
from distort import add_distortions, add_ghost_peaks, add_noise
from classification import test_randForestClassifier
from functions import compareResultLists

spectra: np.ndarray = io.get_spectra()
numSpectra: int = spectra.shape[1] - 1
database = io.get_database()

descriptors: DescriptorLibrary = DescriptorLibrary()
descriptors.generate_from_specDatabase(database, maxDescPerSet=20)
# print(descriptors.getTotalNumberOfDescriptors())
descriptors.optimize_descriptorSets(maxDescriptorsPerSet=5)
# descriptors: DescriptorLibrary = handMadeDescLib
# print(descriptors.getTotalNumberOfDescriptors())

presentPolyms = [descSet.name for descSet in descriptors._descriptorSets]  # TODO: If I want to keep that, use a public function....
delIndices = []
for i in range(database.getNumberOfSpectra()):
    if database.getSpectrumNameOfIndex(i) not in presentPolyms:
        delIndices.append(i)

origResults: List[str] = [database.getSpectrumNameOfIndex(i) for i in range(database.getNumberOfSpectra())]
for i in reversed(delIndices):
    origResults.remove(origResults[i])
dbSpecs: np.ndarray = np.delete(database.getSpectra(), delIndices, axis=1)

numOrigSpecs = dbSpecs.shape[1]-1
numVariations = 10
allSpecs: np.ndarray = np.zeros((dbSpecs.shape[0], numOrigSpecs*numVariations + 1))
allSpecs[:, :numOrigSpecs+1] = dbSpecs.copy()
trueResults: List[str] = origResults.copy()

plotIndex = 9
if plotIndex is not None:
    plt.plot(allSpecs[:, 0], allSpecs[:, plotIndex+1])

for i in range(numVariations):
    if i > 0:
        for j in range(numOrigSpecs):
            currentSpec = allSpecs[:, [0, (i-1)*numOrigSpecs + j + 1]].copy()
            altered = add_distortions(currentSpec, seed=i*numOrigSpecs+j, level=0.5)
            altered = add_ghost_peaks(altered, seed=i*numOrigSpecs+j, maxLevel=0.3)
            altered = add_noise(altered, seed=i*numOrigSpecs+j, maxLevel=0.1)

            allSpecs[:, (i*numOrigSpecs + j + 1)] = altered[:, 1]
            trueResults.append(origResults[j])
            if plotIndex is not None:
                if j == plotIndex:
                    plt.plot(allSpecs[:, 0], altered[:, 1] + i*0.5)

numALlSpecs = allSpecs.shape[1]-1
print(f'Loaded {numALlSpecs} sample spectra and {numOrigSpecs} ref spectra.')

clf, uniqueAssignments = test_randForestClassifier(descriptors.getCorrelationMatrixToSpectra(allSpecs), trueResults)


t0 = time.time()
dbResults = correlate_spectra(allSpecs, database)
resultQualityDB: float = np.count_nonzero(np.array(dbResults) == np.array(trueResults)) / numALlSpecs * 100
print(f'Spec correlation took {round(time.time()-t0, 2)} seconds, {round(resultQualityDB)} % correct hits')
print(compareResultLists(trueResults, dbResults))

t0 = time.time()
prediction = clf.predict(descriptors.getCorrelationMatrixToSpectra(allSpecs))
descriptorResults = [uniqueAssignments[i] for i in prediction]
resultQualityDesc: float = np.count_nonzero(descriptorResults == np.array(trueResults)) / numALlSpecs * 100
print(f'Spectra Descriptor Application took {round(time.time()-t0, 2)} seconds, {round(resultQualityDesc)} % correct hits')
print(compareResultLists(trueResults, descriptorResults))



# comparison: np.ndarray = np.zeros((20, 2))
#
# # spectra = io.get_spectra()
# spectra = allSpecs.copy()
# numSpectra = spectra.shape[1]-1
# plt.clf()
# for i in range(comparison.shape[0]):
#     spectra = add_distortions(spectra, level=0.4, seed=i)
#     spectra = add_ghost_peaks(spectra, maxLevel=0.2, seed=i)
#     # spectra = add_noise(spectra, maxLevel=0.05, seed=i)
#     dbResults = correlate_spectra(spectra, database)
#     resultQualityDB = Counter(dbResults).get('PET') / numSpectra * 100
#
#     # prediction = clf.predict(descriptors.getCorrelationMatrixToSpectra(spectra))
#     # descriptorResults = [uniqueAssignments[i] for i in prediction]
#     # try:
#     #     resultQualityDesc: float = Counter(descriptorResults).get('PET') / numSpectra * 100
#     # except TypeError:
#     #     resultQualityDesc = 0.0
#     resultQualityDesc = 0
#     comparison[i, 0] = resultQualityDB
#     comparison[i, 1] = resultQualityDesc
#
#     if i % 2 == 0:
#         print(f'After adding {i+1} distortion(s): {round(resultQualityDB)} % correct hits with DB search '
#               f'and {round(resultQualityDesc)} % correct hits with Spec Descriptors')
#
# plt.title('Distort + GhostPeak')
# plt.plot(comparison[:, 0], label='Database Matching')
# plt.plot(comparison[:, 1], label='Spectral Descriptor')
# plt.legend()
