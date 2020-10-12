import numpy as np
from typing import List
import time
from collections import Counter
import matplotlib.pyplot as plt

import importData as io
from specCorrelation import correlate_spectra
from descriptors import getPETDescriptor, DescriptorSet
from distort import add_distortions, add_noise


spectra: np.ndarray = io.get_spectra()
numSpectra: int = spectra.shape[1] - 1
database = io.get_database()
print(f'Loaded {numSpectra} sample spectra and {database.getNumberOfSpectra()} ref spectra.')
petDescriptor: DescriptorSet = getPETDescriptor()
trueResults: List[str] = ['PET'] * numSpectra  # all the spectra are from PET particles

t0 = time.time()
dbResults: List[str] = correlate_spectra(spectra, database)
resultQualityDB: float = Counter(dbResults).get('PET') / numSpectra * 100
print(f'Spec correlation took {round(time.time()-t0, 2)} seconds, {round(resultQualityDB)} % correct hits')

t0 = time.time()
descriptorResults: List[bool] = petDescriptor.apply_to_spectra(spectra)
resultQualityDesc: float = Counter(descriptorResults).get(True) / numSpectra * 100
print(f'Spectra Descriptor Application took {round(time.time()-t0, 2)} seconds, {round(resultQualityDesc)} % correct hits')

comparison: np.ndarray = np.zeros((21, 2))
comparison[0, 0] = resultQualityDB
comparison[0, 1] = resultQualityDesc

for i in range(comparison.shape[0]-1):
    spectra = add_distortions(spectra, level=0.6, seed=i)
    # spectra = add_noise(spectra, maxLevel=0.05, seed=i)
    dbResults = correlate_spectra(spectra, database)
    resultQualityDB = Counter(dbResults).get('PET') / numSpectra * 100

    descriptorResults = petDescriptor.apply_to_spectra(spectra)
    try:
        resultQualityDesc = Counter(descriptorResults).get(True) / numSpectra * 100
    except TypeError:
        resultQualityDesc = 0.0

    comparison[i+1, 0] = resultQualityDB
    comparison[i+1, 1] = resultQualityDesc

    if i%2 == 0:
        print(f'After adding {i+1} distortion(s): {round(resultQualityDB)} % correct hits with DB search '
              f'and {round(resultQualityDesc)} % correct hits with Spec Descriptors')

plt.plot(comparison[:, 0], label='Database Matching')
plt.plot(comparison[:, 1], label='Spectral Descriptor')
plt.legend()