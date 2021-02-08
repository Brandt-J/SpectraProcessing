import matplotlib.pyplot as plt
from typing import List, TYPE_CHECKING
import time

from specCorrelation import correlate_spectra, CorrelationMode
from functions import compareResultLists
from distort import *

if TYPE_CHECKING:
    from specCorrelation import Database
    from classification import RandomDecisionForest


def testEvaluationOnSpectra(spectra: np.ndarray, assignments: List[str], database: 'Database', 
                            classifier: 'RandomDecisionForest', preprocessSpecs: bool,
                            numIterations: int = 5, plotSpectra: bool = False):
    """

    :param spectra: (NxM) shape array of M-1 spectra with N wavenumbers (wavenumbers in first column)
    :param assignments: List of M-1 known spectra assignments
    :param database: Database object to use for spectra matching
    :param classifier: Random Forest Classifier for spectra classification
    :param preprocessSpecs: Whether or not to preprocess spectra for database matching
    :param numIterations: Number of Iterations to perform
    :param plotSpectra: Whether or not to plot random spectra of every even iteration.
    :return:
    """
    results = []
    totalQualities = [[], []]
    specPlotIndices = np.random.randint(spectra.shape[1]-1, size=5)
    seed = 0
    for i in range(numIterations):
        print(f'----------------ITERATION {i+1} ----------------')
        if i > 0:
            spectra = add_noise(spectra, level=0.5, seed=seed)
            spectra = add_distortions(spectra, level=0.5, seed=seed)
            spectra = add_ghost_peaks(spectra, level=0.4, seed=seed)
            seed += 1

        if i % 2 == 0 and plotSpectra:
            fig = plt.figure()
            ax = fig.add_subplot()
            for offset, ind in enumerate(specPlotIndices):
                specToPlot = spectra[:, ind+1]
                specToPlot -= specToPlot.min()
                specToPlot /= specToPlot.max()
                ax.plot(spectra[:, 0], specToPlot + offset*0.2)
            ax.set_title(f'Random spectra of iteration {i+1}')

        t0 = time.time()
        dbResults = correlate_spectra(spectra, database, CorrelationMode.SFEC, preprocessSpecs)
        resultQualityDB, dbResultDict = compareResultLists(assignments, dbResults)
        print(f'Spec correlation took {round(time.time()-t0, 2)} seconds, {round(resultQualityDB)} % correct hits')

        t0 = time.time()
        descriptorResults = classifier.evaluateSpectra(spectra)
        resultQualityDesc, descResultDict = compareResultLists(assignments, descriptorResults)
        print(f'Spectra Descriptor Application took {round(time.time()-t0, 2)} seconds, {round(resultQualityDesc)} % correct hits')

        totalQualities[0].append(resultQualityDB)
        totalQualities[1].append(resultQualityDesc)
        results.append([dbResultDict, descResultDict])

    resultFig = plt.figure()
    ax = resultFig.add_subplot()
    ax.plot(totalQualities[0], label='database matching')
    ax.plot(totalQualities[1], label='RDF Spec descLib')
    ax.set_xlabel('-- Decreasing spectra quality -->', fontsize=15)
    ax.set_ylabel('Hit Quality (%)', fontsize=15)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=13)
    resultFig.show()
