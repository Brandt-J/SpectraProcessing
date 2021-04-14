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


import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn import over_sampling, under_sampling
from typing import List, Tuple, TYPE_CHECKING
import tensorflow as tf
from tensorflow.keras import models, layers, losses

if TYPE_CHECKING:
    from descriptors import DescriptorLibrary


class BaseClassifier(object):
    def __init__(self, name: str = 'BaseClassifier'):
        super(BaseClassifier, self).__init__()
        self.name = name
        self._uniqueAssignments: List[str] = []

    def trainWithSpectra(self, trainSpecs: np.ndarray, assignments: List[str]) -> None:
        """
        Train with the spectra given in trainSpecs.
        :param trainSpecs: (N x M) array of M-1 Spectra with N wavenumbers
        :param assignments: The according assignments
        :return:
        """
        raise NotImplementedError

    def evaluateSpectra(self, spectra: np.ndarray, cutoff: float = 0.0) -> List[str]:
        """
        Takes an array of spectra and returns the determined assignments for each spectrum. "Unknown" is returned
        if class probability dropbs below the cutoff.
        :param spectra: (NxM) shape array of spectra of M-1 spectra with N wavenumbers (wavenumbers in first column)
        :param cutoff: (0...1) value for required class probability for assigmnent. Will be "unknown" otherwise.
        :return:
        """
        raise NotImplementedError

    def _setUniqueAssignments(self, assignmentList: List[str]) -> None:
        self._uniqueAssignments = list(np.unique(assignmentList))

    def _balanceAndSplit(self, data: np.ndarray, assignments: List[str],
                         testSize: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """

        :param data: (N X M) array with M spectra of N wavenumbers
        :param assignments: List of M Assignments
        :return: Tuple[X_train, X_test, y_train, y_test]
        """
        y: List[int] = [self._uniqueAssignments.index(ass) for ass in assignments]
        X, y = balanceDataset(data, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, random_state=42)
        return X_train, X_test, y_train, y_test

    def _probabilityMatrix2Prediction(self, probabilityMatrix: np.ndarray, cutoff: float) -> List[str]:
        """
        Gets the final prediction
        :param probabilityMatrix: (N x M) array for N spectra with the probability to fall in each of the M categories, each.
        :param cutoff: Cutoff under which a specturm will be labelled "unknnown"
        :return:
        """
        descriptorResults: List[str] = []
        assert probabilityMatrix.shape[1] == len(self._uniqueAssignments)
        for i in range(probabilityMatrix.shape[0]):
        # for i, prob in enumerate(probabilityMatrix):
            maxProb = np.max(probabilityMatrix[i, :])
            if maxProb >= cutoff:
                maxIndices: np.ndarray = np.where(probabilityMatrix[i, :] == maxProb)[0]
                descriptorResults.append(self._uniqueAssignments[int(maxIndices[0])])
            else:
                descriptorResults.append("unknown")
        return descriptorResults


class RandomDecisionForest(BaseClassifier):
    def __init__(self, descLib: 'DescriptorLibrary'):
        super(RandomDecisionForest, self).__init__("Random Forest")
        self._descLib: 'DescriptorLibrary' = descLib
        self._clf: RandomForestClassifier = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)

    def trainWithSpectra(self, trainSpecs: np.ndarray, assignments: List[str]) -> None:
        t0 = time.time()
        self._setUniqueAssignments(assignments)
        featureMat: np.ndarray = self._descLib.getCorrelationMatrixToSpectra(trainSpecs)
        assert len(assignments) == featureMat.shape[0]
        featureMat = StandardScaler().fit_transform(featureMat)
        X_train, X_test, y_train, y_test = self._balanceAndSplit(featureMat, assignments)

        print(f'creating and training rdf on {len(y_train)} samples with {X_train.shape[1]} features')
        self._clf.fit(X_train, y_train)
        score = self._clf.score(X_test, y_test)
        print(f'Classifier score is {round(score, 2)}, training and testing took {round(time.time()-t0, 2)} seconds.')

    def evaluateSpectra(self, spectra: np.ndarray, cutoff: float = 0.0) -> List[str]:
        featureMat: np.ndarray = self._descLib.getCorrelationMatrixToSpectra(spectra)
        featureMat = StandardScaler().fit_transform(featureMat)
        probabilities: np.ndarray = self._clf.predict_proba(featureMat)
        return self._probabilityMatrix2Prediction(probabilities, cutoff)


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
        X_train, X_test, y_train, y_test = self._balanceAndSplit(featureMat, assignments)
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


def balanceDataset(featureMat: np.ndarray, assignments: List[str]) -> Tuple[np.ndarray, List[str]]:
    sampler = over_sampling.SMOTE(random_state=42)
    # sampler = over_sampling.ADASYN(random_state=42)
    # sampler = over_sampling.RandomOverSampler(random_state=42)
    # sampler = under_sampling.ClusterCentroids(random_state=42)

    print('balancing with:', sampler)
    newData, newAssignments = sampler.fit_resample(featureMat, assignments)
    return newData, newAssignments


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


def prepareSpecSet(specSet: np.ndarray, transpose: bool = True, addDimension: bool = False):
    if transpose:
        specSet = specSet.transpose()

    specSet = normalizeSpecSet(specSet)
    if addDimension:
        specSet = specSet.reshape(specSet.shape[0], specSet.shape[1], 1)

    specSet = tf.cast(specSet, tf.float32)
    return specSet
