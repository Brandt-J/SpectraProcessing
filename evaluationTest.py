import matplotlib.pyplot as plt
from typing import List, Tuple, TYPE_CHECKING
import time

from specCorrelation import correlate_spectra, CorrelationMode
from processing import normalizeIntensities, smooth
from functions import compareResultLists
from distort import *

if TYPE_CHECKING:
    from specCorrelation import Database, CorrelationMode
    from classification import RandomDecisionForest


def testEvaluationOnSpectra(spectra: np.ndarray, assignments: List[str], database: 'Database', 
                            classifier: 'RandomDecisionForest', preprocessSpecs: bool,
                            corrModes: List['CorrelationMode'] = [CorrelationMode.PEARSON],
                            dbCutoff: float = 0.5, numIterations: int = 5,
                            plotSpectra: bool = False) -> Tuple[plt.Figure, List[List[float]]]:
    """

    :param spectra: (NxM) shape array of M-1 spectra with N wavenumbers (wavenumbers in first column)
    :param assignments: List of M-1 known spectra assignments
    :param database: Database object to use for spectra matching
    :param classifier: Random Forest Classifier for spectra classification
    :param preprocessSpecs: Whether or not to preprocess spectra for database matching
    :param corrModes: List of correlation Modes to test
    :param numIterations: Number of Iterations to perform
    :param plotSpectra: Whether or not to plot random spectra of every even iteration.
    :param dbCutoff: Cutoff (0.0 -> 1.0) for db correlation. Correlations lower than that are considered "unknown"
    :return:
    """
    totalPrecisions: List[List[float]] = []
    totalRecalls: List[List[float]] = []
    for _ in range(len(corrModes)):
        totalPrecisions.append([])
        totalRecalls.append([])
    totalPrecisions.append([])
    totalRecalls.append([])

    specPlotIndices = np.random.randint(spectra.shape[1]-1, size=5)
    seed = 0
    for i in range(numIterations):
        print(f'----------------ITERATION {i+1} ----------------')
        if i > 0:
            spectra = add_noise(spectra, level=0.2, seed=seed)
            spectra = add_distortions(spectra, level=0.7, seed=seed)
            spectra = add_ghost_peaks(spectra, level=0.6, seed=seed)
            seed += 1

        if i % 2 == 0 and plotSpectra:
            fig = plt.figure()
            ax = fig.add_subplot()
            for offset, ind in enumerate(specPlotIndices):
                specToPlot = normalizeIntensities(spectra[:, ind+1].copy())
                ax.plot(spectra[:, 0], specToPlot + offset*0.2)
            ax.set_title(f'Random spectra of iteration {i+1}')

        for ind, mode in enumerate(corrModes):
            t0 = time.time()
            dbResults = correlate_spectra(spectra, database, mode, cutoff=dbCutoff, preproc=preprocessSpecs)
            precDB, recallDB = compareResultLists(assignments, dbResults)
            totalPrecisions[ind].append(precDB)
            totalRecalls[ind].append(recallDB)
            print(f'Spec correlation with {mode} took {round(time.time()-t0, 2)} seconds, {round(precDB)} % correct hits')

        t0 = time.time()
        descriptorResults = classifier.evaluateSpectra(spectra)
        precRDF, recallRDF = compareResultLists(assignments, descriptorResults)
        print(f'Spectra Descriptor Application took {round(time.time()-t0, 2)} seconds, {round(precRDF)} % correct hits')
        totalPrecisions[ind+1].append(precRDF)
        totalRecalls[ind + 1].append(recallRDF)

    resultFig: plt.Figure = plt.figure(figsize=(12, 5))
    ax1: plt.Axes = resultFig.add_subplot(121)
    ax2: plt.Axes = resultFig.add_subplot(122)
    for ind, (precisions, recalls) in enumerate(zip(totalPrecisions, totalRecalls)):
        if ind < len(corrModes):
            label = f'database matching, {corrModes[ind]}'
        else:
            label = 'RDF Spec descLib'
        ax1.plot(np.arange(len(precisions))+1, precisions, label=label)
        ax2.plot(np.arange(len(recalls))+1, recalls, label=label)

    ax1.set_ylabel('Precision (%)', fontsize=15)
    ax2.set_ylabel('Recall (%)', fontsize=15)
    for ax in [ax1, ax2]:
        ax.set_xlabel('-- Decreasing spectra quality -->', fontsize=15)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=10)
    resultFig.tight_layout()
    return resultFig, totalPrecisions
