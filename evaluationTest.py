import matplotlib.pyplot as plt
from typing import List, Tuple, TYPE_CHECKING
import time

from specCorrelation import correlate_spectra, CorrelationMode
from functions import compareResultLists
from distort import *

if TYPE_CHECKING:
    from specCorrelation import Database, CorrelationMode
    from classification import RandomDecisionForest


def testEvaluationOnSpectra(spectra: np.ndarray, assignments: List[str], database: 'Database', 
                            classifier: 'RandomDecisionForest', preprocessSpecs: bool,
                            corrModes: List['CorrelationMode'] = [CorrelationMode.PEARSON],
                            numIterations: int = 5, plotSpectra: bool = False) -> Tuple[plt.Figure, List[List[float]]]:
    """

    :param spectra: (NxM) shape array of M-1 spectra with N wavenumbers (wavenumbers in first column)
    :param assignments: List of M-1 known spectra assignments
    :param database: Database object to use for spectra matching
    :param classifier: Random Forest Classifier for spectra classification
    :param preprocessSpecs: Whether or not to preprocess spectra for database matching
    :param corrModes: List of correlation Modes to test
    :param numIterations: Number of Iterations to perform
    :param plotSpectra: Whether or not to plot random spectra of every even iteration.
    :return:
    """
    totalQualities: List[List[float]] = []
    for _ in range(len(corrModes)):
        totalQualities.append([])
    totalQualities.append([])

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

        for ind, mode in enumerate(corrModes):
            t0 = time.time()
            dbResults = correlate_spectra(spectra, database, mode, preprocessSpecs)
            resultQualityDB, dbResultDict = compareResultLists(assignments, dbResults)
            totalQualities[ind].append(resultQualityDB)
            print(f'Spec correlation with {mode} took {round(time.time()-t0, 2)} seconds, {round(resultQualityDB)} % correct hits')

        t0 = time.time()
        descriptorResults = classifier.evaluateSpectra(spectra)
        resultQualityDesc, descResultDict = compareResultLists(assignments, descriptorResults)
        print(f'Spectra Descriptor Application took {round(time.time()-t0, 2)} seconds, {round(resultQualityDesc)} % correct hits')
        totalQualities[ind+1].append(resultQualityDesc)

    resultFig = plt.figure()
    ax = resultFig.add_subplot()
    for ind, results in enumerate(totalQualities):
        if ind < len(corrModes):
            label = f'database matching, {corrModes[ind]}'
        else:
            label = 'RDF Spec descLib'
        ax.plot(np.arange(len(results))+1, results, label=label)

    ax.set_xlabel('-- Decreasing spectra quality -->', fontsize=15)
    ax.set_ylabel('Hit Quality (%)', fontsize=15)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    return resultFig, totalQualities
