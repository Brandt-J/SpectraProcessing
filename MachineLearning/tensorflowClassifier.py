import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import List, TYPE_CHECKING
import tensorflow as tf
from tensorflow.keras import models, layers, losses

from classification import BaseClassifier
if TYPE_CHECKING:
    from descriptors import DescriptorLibrary


class NNClassifier(BaseClassifier):
    def __init__(self, descriptorLibrary: 'DescriptorLibrary'):
        super(NNClassifier, self).__init__("Neural Net")
        self._descLib: 'DescriptorLibrary' = descriptorLibrary
        self._clf: models.Sequential = models.Sequential()
        self._clf.add(layers.InputLayer(input_shape=(self._descLib.getTotalNumberOfDescriptors())))
        # self._clf.add(layers.Dense(1024, activation="relu"))
        # self._clf.add(layers.Dropout(0.2))
        # self._clf.add(layers.Dense(512, activation="relu"))
        # self._clf.add(layers.Dropout(0.2))
        self._clf.add(layers.Dense(64, activation="relu"))
        self._clf.add(layers.Dense(self._descLib.getNumberOfDescriptorSets(), activation="softmax"))
        self._clf.compile(optimizer="adam", loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
        self._clf.summary()

    def trainWithSpectra(self, trainSpecs: np.ndarray, assignments: List[str]) -> None:
        self._setUniqueAssignments(assignments)
        featureMat: np.ndarray = self._descLib.getCorrelationMatrixToSpectra(trainSpecs)
        featureMat = StandardScaler().fit_transform(featureMat)
        X_train, X_test, y_train, y_test = _balanceAndSplit(featureMat, assignments)
        trainSpecs = prepareSpecSet(X_train, transpose=False)
        testSpecs = prepareSpecSet(X_test, transpose=False)
        y_train = tf.cast(y_train, tf.int32)
        y_test = tf.cast(y_test, tf.int32)
        history = self._clf.fit(trainSpecs, y_train, epochs=50, validation_data=(testSpecs, y_test),
                                batch_size=32, shuffle=True)

    def evaluateSpectra(self, spectra: np.ndarray, cutoff: float = 0.0) -> List[str]:
        featureMat: np.ndarray = self._descLib.getCorrelationMatrixToSpectra(spectra)
        featureMat = StandardScaler().fit_transform(featureMat)
        featureMat = prepareSpecSet(featureMat, transpose=False)
        prediction = self._clf.predict(featureMat)
        return self._probabilityMatrix2Prediction(prediction, cutoff)


def prepareSpecSet(specSet: np.ndarray, transpose: bool = True, addDimension: bool = False):
    if transpose:
        specSet = specSet.transpose()

    specSet = normalizeSpecSet(specSet)
    if addDimension:
        specSet = specSet.reshape(specSet.shape[0], specSet.shape[1], 1)

    specSet = tf.cast(specSet, tf.float32)
    return specSet


def normalizeSpecSet(specSet: np.ndarray) -> np.ndarray:
    """
    Normalizing Specset to 0.0 -> 1.0 range, for each spectrum individually
    :param specSet: (N x M) array of N spectra with M wavenumbers
    :return: normalized specset
    """
    for i in range(specSet.shape[0]):
        intens: np.ndarray = specSet[i, :]
        intens -= intens.min()
        if intens.max() != 0:
            intens /= intens.max()
        specSet[i, :] = intens
    return specSet
