import numpy as np
from typing import List
import time
from collections import Counter

import importData as io
from specCorrelation import correlateSpectra
from distort import addDistortions, addNoise


spectra: np.ndarray = io.getSpectra()
numSpectra: int = spectra.shape[1] - 1
database = io.getDatabase()
print(f'Loaded {numSpectra} sample spectra and {database.getNumberOfSpectra()} ref spectra.')
trueResults: List[str] = ['PET'] * numSpectra  # all the spectra are from PET particles

t0 = time.time()
dbResults: List[str] = correlateSpectra(spectra, database)
resultQuality: float = Counter(dbResults).get('PET') / numSpectra
print(f'Spec correlation took {round(time.time()-t0, 2)} seconds, {round(resultQuality*100)} % correct hits')

for i in range(20):
    spectra = addDistortions(spectra, level=0.6, seed=i)
    spectra = addNoise(spectra, seed=i)
    dbResults: List[str] = correlateSpectra(spectra, database)
    resultQuality: float = Counter(dbResults).get('PET') / numSpectra
    if i%2 == 0:
        print(f'After adding {i+1} distortion(s): {round(resultQuality * 100)} % correct hits with DB search')
