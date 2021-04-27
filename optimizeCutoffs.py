"""
SPECTRA PROCESSING
Copyright (C) 2020 Josef Brandt, University of Gothenborg.
<josef.brandt@gu.se>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program, see COPYING.
If not, see <https://www.gnu.org/licenses/>.
"""
from typing import List
import numpy as np
from scipy.optimize import least_squares
import time
import matplotlib.pyplot as plt

import importData as io
from descriptors import DescriptorLibrary
from classification import RandomDecisionForest
from specCorrelation import CorrelationMode, correlate_spectra
from functions import compareResultLists
from testSpectra import TestSpectra


def getPredictionErrorFromDB(testSpec: np.ndarray, assignments: List[str], db: 'io.Database', corrMode: CorrelationMode,
                             cutOff: float) -> np.ndarray:
    dbResults = correlate_spectra(testSpec, db, corrMode, cutoff=cutOff, preproc=False)
    precDB, recallDB, f1DB, plastErrorDB = compareResultLists(assignments, dbResults)
    errVec: np.ndarray = calculateErrorVector(precDB, recallDB, f1DB, plastErrorDB)
    print(f'Cutoff: {cutOff} Precision: {precDB} %, Recall: {recallDB} %, '
          f'F1: {f1DB} %, plastError: {plastErrorDB} %')
    return errVec


def getPredictionErrorFromRDF(testSpec: np.ndarray, assignments: List[str], clf: 'RandomDecisionForest', cutOff: float) -> np.ndarray:
    descriptorResults = clf.evaluateSpectra(testSpec, cutoff=cutOff)
    precRDF, recallRDF, f1RDF, plastErrorRDF = compareResultLists(assignments, descriptorResults)
    errVec: np.ndarray = calculateErrorVector(precRDF, recallRDF, f1RDF, plastErrorRDF)
    # print(f'Cutoff: {cutOff} Precision: {precRDF} %, Recall: {recallRDF} %, '
    #       f'F1: {f1RDF} %, plastError: {plastErrorRDF} %')
    return errVec


def calculateErrorVector(precision: float, recall: float, f1: float, plastError: float) -> np.ndarray:
    error: np.ndarray = np.array([1-precision/100, 1-recall/100, 1-f1/100, 2*abs(plastError/100)])
    return error


preprocessSpectra: bool = True  # Whether or not subtract baseline and normalize spectra for database search
correlationModes: List[CorrelationMode] = [CorrelationMode.SFEC]
nMaxDBSpecs: int = 20  # maximum number of spectra in the database
nMaxDesc: int = 20  # maximum number of descriptors per spectrum

database = io.get_database(maxSpectra=nMaxDBSpecs, includeNonPlastic=False)
database.preprocessSpectra()
descriptors: DescriptorLibrary = DescriptorLibrary()
descriptors.generate_from_specDatabase(database, maxDescPerSet=200)
descriptors.optimize_descriptorSets(maxDescriptorsPerSet=nMaxDesc)
rdf: RandomDecisionForest = RandomDecisionForest(descriptors)


cutOffs = np.linspace(0.05, 1.0, 10)
plotNum = 1
for plasticContent in [0.1, 0.25, 0.5, 0.9]:
    testSpecObj = TestSpectra()
    testSpecObj.generateFromRefSpecs(plasticContent=plasticContent, numVariations=20, maxPlastTypes=nMaxDBSpecs)

    trainSpectra: np.ndarray = testSpecObj.getPolymerSpectra()
    trainAssignments: List[str] = testSpecObj.getPolymerAssignments()
    rdf.trainWithSpectra(trainSpectra, trainAssignments)

    testSpectra = testSpecObj.getAllSpectra()
    testAssignments = testSpecObj.getAllAssignments()
    numPlast, numNonPlast = testSpecObj.getNumberOfPlastics(), testSpecObj.getNumberOfNonPlastics()
    print(f"testing with {numPlast} plastic and {numNonPlast} non-plastic spectra, plastic content: {testSpecObj.getPlasticContent()}")

    plt.subplot(2, 2, plotNum)
    f1s, plastErrors = [], []
    for cutOff in cutOffs:
        results = correlate_spectra(testSpectra, database, CorrelationMode.PEARSON, cutoff=cutOff, preproc=True)
        prec, recall, f1, plastError = compareResultLists(testAssignments, results)
        # descriptorResults = rdf.evaluateSpectra(testSpectra, cutoff=cutOff)
        # prec, recall, f1, plastError = compareResultLists(testAssignments, descriptorResults)
        f1s.append(f1)
        plastErrors.append(plastError)
    plt.plot(cutOffs, f1s, label='F1 Score', marker='o')
    plt.plot(cutOffs, plastErrors, label='Plastic Error', marker='o')
    plt.xlabel('CutOff')
    plt.legend()

    t0 = time.time()
    # clfErrorFunc = lambda x: getPredictionErrorFromRDF(testSpectra, testAssignments, rdf, cutOff=x)
    # opt = least_squares(clfErrorFunc, np.array(plasticContent), bounds=(np.array(0.0), np.array(1.0)), method='trf', diff_step=0.2)
    dbErrorFunc = lambda x: getPredictionErrorFromDB(testSpectra, testAssignments, database, corrMode=CorrelationMode.PEARSON, cutOff=x)
    opt = least_squares(dbErrorFunc, np.array(0.6), bounds=(np.array(0.0), np.array(1.0)), method='trf', diff_step=0.2)
    plt.plot([0, 1], [100, 100], color='blue', linestyle='--', alpha=0.5)
    plt.plot([0, 1], [0, 0], color='orange', linestyle='--', alpha=0.5)
    results = rdf.evaluateSpectra(testSpectra, cutoff=opt.x)


    prec, recall, f1, plastError = compareResultLists(testAssignments, results)
    print(f'\tOptimization took {round(time.time() - t0, 2)} seconds, optimized cutoff for RDF is {opt.x}')
    if f1 > plastError:
        if plastError > 0:
            plt.plot([opt.x, opt.x], [1.1*f1, 0.9*plastError], color='black')
        else:
            plt.plot([opt.x, opt.x], [1.1*f1, 1.1*plastError], color='black')
    else:
        plt.plot([opt.x, opt.x], [0.9*f1, 1.1*plastError], color='black')

    plt.title(f'Plastic Content: {plasticContent}\nAt cutoff {round(opt.x[0], 2)}: Precision: {round(prec)} %, '
              f'Recall: {round(recall)} %,\nF1: {round(f1)} %, plastError: {round(plastError)} %')
    plotNum += 1

    # for mode in correlationModes:
    #     stepSize = 0.1 if mode == CorrelationMode.PEARSON else -0.3
    #     t0 = time.time()
    #     dbErrorFunc = lambda x: getPredictionErrorFromDB(testSpectra, testAssignments, database, corrMode=mode, cutOff=x)
    #     opt = least_squares(dbErrorFunc, np.array(0.5), bounds=(np.array(0.0), np.array(1.0)), method='trf',
    #                         diff_step=stepSize)
    #     print(f'\tOptimization took {round(time.time()-t0, 2)} seconds, optimized cutoff for {mode} is {opt.x}')
    #     results = correlate_spectra(testSpectra, database, mode, cutoff=opt.x, preproc=False)
    #     prec, recall, f1, plastError = compareResultLists(testAssignments, results)
    #     print(f'\tPrecision: {prec} %, Recall: {recall} %, F1: {f1} %, plastError: {plastError} %')





plt.tight_layout()

