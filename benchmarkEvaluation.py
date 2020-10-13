import numpy as np
from typing import List
import time
from collections import Counter
import matplotlib.pyplot as plt

import importData as io
from specCorrelation import correlate_spectra
from descriptors import DescriptorLibrary
from distort import add_distortions, add_ghost_peaks, add_noise


# spectra: np.ndarray = io.get_spectra()
# numSpectra: int = spectra.shape[1] - 1
database = io.get_database()
dbSpecs: np.ndarray = database.getSpectra()
numOrigSpecs = database.getNumberOfSpectra()
numVariations = 1

allSpecs: np.ndarray = np.zeros((dbSpecs.shape[0], numOrigSpecs*numVariations + 1))
allSpecs[:, :numOrigSpecs+1] = dbSpecs.copy()
trueResults: List[str] = [database.getSpectrumNameOfIndex(i) for i in range(numOrigSpecs)]

for i in range(numVariations):
    if i > 0:
        for j in range(numOrigSpecs):
            currentSpec = allSpecs[:, [0, (i-1)*numOrigSpecs + j + 1]].copy()
            altered = add_distortions(currentSpec, seed=i*numOrigSpecs+j, level=0.8)
            altered = add_ghost_peaks(altered, seed=i*numOrigSpecs+j, maxLevel=0.3)
            # altered = add_noise(altered, seed=i*numOrigSpecs+j)

            allSpecs[:, (i*numOrigSpecs + j + 1)] = altered[:, 1]
            trueResults.append(database.getSpectrumNameOfIndex(j))

trueResults: np.ndarray = np.array(trueResults)
numALlSpecs = allSpecs.shape[1]-1
print(f'Loaded {numALlSpecs} sample spectra and {database.getNumberOfSpectra()} ref spectra.')

descriptors: DescriptorLibrary = DescriptorLibrary()
descriptors.generate_from_specDatabase(database)

t0 = time.time()
dbResults = np.array(correlate_spectra(allSpecs, database))
resultQualityDB: float = np.count_nonzero(dbResults == trueResults) / numALlSpecs * 100
print(f'Spec correlation took {round(time.time()-t0, 2)} seconds, {round(resultQualityDB)} % correct hits')

t0 = time.time()
descriptorResults = np.array(descriptors.apply_to_spectra(allSpecs))
resultQualityDesc: float = np.count_nonzero(descriptorResults == trueResults) / numALlSpecs * 100
print(f'Spectra Descriptor Application took {round(time.time()-t0, 2)} seconds, {round(resultQualityDesc)} % correct hits')

# comparison: np.ndarray = np.zeros((20, 2))
#
# spectra = io.get_spectra()
# numSpectra = spectra.shape[1]-1
# plt.clf()
# for i in range(comparison.shape[0]):
#     spectra = add_distortions(spectra, level=0.4, seed=i)
#     spectra = add_ghost_peaks(spectra, maxLevel=0.2, seed=i)
#     # spectra = add_noise(spectra, maxLevel=0.05, seed=i)
#     dbResults = correlate_spectra(spectra, database)
#     resultQualityDB = Counter(dbResults).get('PET') / numSpectra * 100
#
#     descriptorResults = descriptors.apply_to_spectra(spectra)
#     try:
#         resultQualityDesc: float = Counter(descriptorResults).get('PET') / numSpectra * 100
#     except TypeError:
#         resultQualityDesc = 0.0
#
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
