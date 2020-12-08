import matplotlib.pyplot as plt
from typing import List
import time
import os
from sklearn.preprocessing import StandardScaler

import importData as io
from specCorrelation import correlate_spectra
from descriptors import DescriptorLibrary
from handmadeDescriptors import handMadeDescLib
from classification import test_randForestClassifier, balanceDataset
from functions import compareResultLists
from distort import *

t0 = time.time()
pathSampleSpec: str = r'Sample Spectra/sampleSpectra.npy'
pathSampleAssignments: str = r'Sample Spectra/origResults.txt'

forceRegenerate: bool = False
if forceRegenerate or not (os.path.exists(pathSampleSpec) and os.path.exists(pathSampleAssignments)):
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

descriptors: DescriptorLibrary = DescriptorLibrary()
descriptors.generate_from_specDatabase(database, maxDescPerSet=200)
# descriptors: DescriptorLibrary = handMadeDescLib
# descriptors.getDescriptorPlot().show()
descriptors.optimize_descriptorSets(maxDescriptorsPerSet=5)
# descriptors.getDescriptorPlot().show()

featureMat: np.ndarray = descriptors.getCorrelationMatrixToSpectra(testSpectra)
X, y = featureMat.copy(), origResults.copy()

X = StandardScaler().fit_transform(X)
X, y = balanceDataset(X, y)
t0 = time.time()
clf, uniqueAssignments = test_randForestClassifier(featureMat, origResults)
print(f'creating rdf classifier took {round(time.time()-t0, 2)} seconds')

results = []
totalQualities = [[], []]
for i in range(10):
    print(f'----------------ITERATION {i+1} ----------------')
    if i > 0:
        testSpectra = add_noise(testSpectra, maxLevel=0.7)
        testSpectra = add_distortions(testSpectra, level=0.7)
        testSpectra = add_ghost_peaks(testSpectra, maxLevel=0.7)

    t0 = time.time()
    dbResults = correlate_spectra(testSpectra, database)
    resultQualityDB, dbResultDict = compareResultLists(origResults, dbResults)
    print(f'Spec correlation took {round(time.time()-t0, 2)} seconds, {round(resultQualityDB)} % correct hits')
    print(dbResultDict)

    t0 = time.time()
    # descriptorResults = descriptors.apply_to_spectra(testSpectra)
    featureMat = descriptors.getCorrelationMatrixToSpectra(testSpectra)
    featureMat = StandardScaler().fit_transform(featureMat)
    descriptorResults = [uniqueAssignments[i] for i in clf.predict(featureMat)]
    resultQualityDesc, descResultDict = compareResultLists(origResults, descriptorResults)
    print(f'Spectra Descriptor Application took {round(time.time()-t0, 2)} seconds, {round(resultQualityDesc)} % correct hits')
    print(descResultDict)

    totalQualities[0].append(resultQualityDB)
    totalQualities[1].append(resultQualityDesc)
    results.append([dbResultDict, descResultDict])

colorCycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.plot(totalQualities[0], label='database matching')
plt.plot(totalQualities[1], label='RDF Spec Descriptors')
plt.xlabel('-- Decreasing spectra quality -->', fontsize=15)
plt.ylabel('Hit Quai§lty (%)', fontsize=15)
plt.legend(fontsize=13)

# for i, name in enumerate(uniqueAssignments):
#     name = name.lower()
#     dbDicts = [j[0] for j in results]
#     dbData = [resDict[name] for resDict in dbDicts]
#     plt.plot(dbData, '-', label=f'db: {name}', color=colorCycle[i])
#
#     descDicts = [j[1] for j in results]
#     descData = [resDict[name] for resDict in descDicts]
#     plt.plot(descData, '--', label=f'desc: {name}', color=colorCycle[i])
# plt.legend()
