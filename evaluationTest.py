from typing import List, Dict, Tuple, TYPE_CHECKING
import time

from specCorrelation import correlate_spectra, CorrelationMode
from Preprocessing.processing import normalizeIntensities
from functions import compareResultLists
from distort import *

if TYPE_CHECKING:
    from specCorrelation import Database, CorrelationMode
    from classification import BaseClassifier


def testEvaluationOnSpectra(spectra: np.ndarray, assignments: List[str], database: 'Database', 
                            classifiers: List['BaseClassifier'], preprocessSpecs: bool,
                            corrModes: List['CorrelationMode'] = [CorrelationMode.PEARSON],
                            dbCutoff: float = 0.5, rdfCutoff: float = 0.5, numIterations: int = 5,
                            plotSpectra: bool = False, plotTitle: str = '') -> Tuple[plt.Figure, Dict[str, 'ResultObj']]:
    """

    :param spectra: (NxM) shape array of M-1 spectra with N wavenumbers (wavenumbers in first column)
    :param assignments: List of M-1 known spectra assignments
    :param database: Database object to use for spectra matching
    :param classifiers: List of Random Forest and/or Neural Net Classifiers for spectra classification
    :param preprocessSpecs: Whether or not to preprocess spectra for database matching
    :param corrModes: List of correlation Modes to test
    :param numIterations: Number of Iterations to perform
    :param dbCutoff: Cutoff (0.0 -> 1.0) for db correlation. Correlations lower than that are considered "unknown"
    :param rdfCutoff: Cutoff (0.0 -> 1.0) for RDF application. Correlations lower than that are considered "unknown"
    :param plotSpectra: Whether or not to plot random spectra of every even iteration.
    :param plotTitle: If given, the string will be used as title for the result plot
    :return:
    """
    minPlastError, maxPlastError = 50, -50
    resultObjs: Dict[str, ResultObj] = {}

    for corrMode in corrModes:
        resultObjs[corrMode.getName()] = ResultObj()
    for clf in classifiers:
        resultObjs[clf.name] = ResultObj()

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

        for mode in corrModes:
            t0 = time.time()
            resultObj: 'ResultObj' = resultObjs[mode.getName()]
            dbResults = correlate_spectra(spectra, database, mode, cutoff=dbCutoff, preproc=preprocessSpecs)
            precDB, recallDB, f1DB, plastErrorDB = compareResultLists(assignments, dbResults)
            resultObj.totalPrecisions.append(precDB)
            resultObj.totalRecalls.append(recallDB)
            resultObj.totalF1.append(f1DB)
            resultObj.totalPlastErrors.append(plastErrorDB)
            if plastErrorDB < minPlastError:
                minPlastError = plastErrorDB
            if plastErrorDB > maxPlastError:
                maxPlastError = plastErrorDB
            print(f'Spec correlation with {mode} took {round(time.time()-t0, 2)} seconds, {round(precDB)} % correct hits')

        for classifier in classifiers:
            t0 = time.time()
            resultObj = resultObjs[classifier.name]
            descriptorResults = classifier.evaluateSpectra(spectra, cutoff=rdfCutoff)
            precRDF, recallRDF, f1RDF, plastErrorRDF = compareResultLists(assignments, descriptorResults)
            print(f'Application of {classifier.name} took {round(time.time()-t0, 2)} seconds, {round(precRDF)} % correct hits')
            resultObj.totalPrecisions.append(precRDF)
            resultObj.totalRecalls.append(recallRDF)
            resultObj.totalF1.append(f1RDF)
            resultObj.totalPlastErrors.append(plastErrorRDF)
            if plastErrorRDF < minPlastError:
                minPlastError = plastErrorRDF
            if plastErrorRDF > maxPlastError:
                maxPlastError = plastErrorRDF

    resultFig: plt.Figure = plt.figure(figsize=(12, 5))
    ax1: plt.Axes = resultFig.add_subplot(221)
    ax2: plt.Axes = resultFig.add_subplot(222)
    ax3: plt.Axes = resultFig.add_subplot(223)
    ax4: plt.Axes = resultFig.add_subplot(224)

    for label, resultObj in resultObjs.items():
        ax1.plot(np.arange(len(resultObj.totalPrecisions))+1, resultObj.totalPrecisions, marker='o', label=label)
        ax2.plot(np.arange(len(resultObj.totalRecalls))+1, resultObj.totalRecalls, marker='o', label=label)
        ax3.plot(np.arange(len(resultObj.totalF1)) + 1, resultObj.totalF1, marker='o', label=label)
        ax4.plot(np.arange(len(resultObj.totalPlastErrors)) + 1, resultObj.totalPlastErrors, marker='o', label=label)

    ax1.set_ylabel('Precision (%)', fontsize=15)
    ax2.set_ylabel('Recall (%)', fontsize=15)
    ax3.set_ylabel('F1 Score (%)', fontsize=15)
    ax4.set_ylabel('Plastic Count Error (%)', fontsize=15)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('-- Decreasing spectra quality -->', fontsize=15)
        ax.set_ylim(0, 105)
        ax.legend(fontsize=10)
    ax4.set_ylim(minPlastError * 1.1, maxPlastError * 1.1)
    if plotTitle != '':
        resultFig.suptitle(plotTitle, fontsize=15)
    resultFig.tight_layout()
    return resultFig, resultObjs


class ResultObj(object):
    def __init__(self):
        self.totalPrecisions: List[float] = []
        self.totalRecalls: List[float] = []
        self.totalF1: List[float] = []
        self.totalPlastErrors: List[float] = []
