import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import numpy as np
from collections import Counter
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from testSpectra import TestSpectra
    from specCorrelation import Database


def prepareSpecSet(specSet: np.ndarray, transpose: bool = True):
    specSet -= specSet.min()
    specSet /= specSet.max()
    if transpose:
        specSet = specSet.transpose()
    specSet = tf.cast(specSet, tf.float32)
    return specSet


class Denoiser(Model):
    def __init__(self, specLength: int = 512):
        super(Denoiser, self).__init__()
        self.encoder = tf.keras.Sequential([
            Dense(128, activation="relu")])

        self.decoder = tf.keras.Sequential([
            Dense(128, activation="relu"),
            Dense(specLength, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def getTrainTestSpecs(testObj: 'TestSpectra', database: 'Database', trainSplit: float = 0.4) -> tuple:
    testSpecs: np.ndarray = testObj.getAllSpectra()[:, 1:]
    trainSpectra, testSpectra, noisyTrainSpectra, noisyTestSpectra = [], [], [], []
    assignments = testObj.getAllAssignments()
    countedAssignments: Counter = Counter(assignments)
    for specType, count in countedAssignments.items():
        assert specType.lower() in database.getSpectraNames(), f'SpecType {specType} not present in Database!'
        numTrain = round(count * trainSplit)
        typeIndices = [i for i, name in enumerate(assignments) if name == specType]
        for i in typeIndices[:numTrain]:
            noisyTrainSpectra.append(testSpecs[:, i])
            trainSpectra.append(database.getSpectrumOfName(specType.lower())[:, 1])

        for i in typeIndices[numTrain:]:
            noisyTestSpectra.append(testSpecs[:, i])
            testSpectra.append(database.getSpectrumOfName(specType.lower())[:, 1])

    trainSpectra = np.array(trainSpectra)
    testSpectra = np.array(testSpectra)
    noisyTrainSpectra = np.array(noisyTrainSpectra)
    noisyTestSpectra = np.array(noisyTestSpectra)

    trainSpectra = prepareSpecSet(trainSpectra, transpose=False)
    testSpectra = prepareSpecSet(testSpectra, transpose=False)
    noisyTrainSpectra = prepareSpecSet(noisyTrainSpectra, transpose=False)
    noisyTestSpectra = prepareSpecSet(noisyTestSpectra, transpose=False)

    return trainSpectra, noisyTrainSpectra, testSpectra, noisyTestSpectra
