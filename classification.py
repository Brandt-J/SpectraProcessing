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
import tensorflow as tf
from tensorflow.keras import models, layers, losses


from typing import List, TYPE_CHECKING


if TYPE_CHECKING:
    from descriptors import DescriptorLibrary


class BaseClassifier(object):
    def __init__(self, name: str = 'BaseClassifier'):
        super(BaseClassifier, self).__init__()
        self.name = name
        self._uniqueAssignments: List[str] = []

    def trainWithSpectra(self, trainSpecs: np.ndarray, assignments: List[str], testSize: float = 0.1) -> None:
        """
        Train with the spectra given in trainSpecs.
        :param trainSpecs: (M x N) array of M Spectra with N wavenumbers
        :param assignments: The according assignments
        :param testSize: Fraction of data used for testing
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
            maxProb = np.max(probabilityMatrix[i, :])
            if maxProb >= cutoff:
                maxIndices: np.ndarray = np.where(probabilityMatrix[i, :] == maxProb)[0]
                descriptorResults.append(self._uniqueAssignments[int(maxIndices[0])])
            else:  # also triggers, when probability is nan
                descriptorResults.append("unknown")
        return descriptorResults


class NeuralNetClassifier(BaseClassifier):
    def __init__(self, inputShape: int, numClasses: int):
        super(NeuralNetClassifier, self).__init__("Neural Net")
        self._model = models.Sequential()
        self._model.add(layers.InputLayer(input_shape=(inputShape)))
        self._model.add(layers.Dense(32, activation="relu"))
        self._model.add(layers.Dense(16, activation="relu"))
        self._model.add(layers.Dense(numClasses, activation="softmax"))
        self._model.compile(optimizer="adam", loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                            metrics=['accuracy'])

    def trainWithSpectra(self, spectra: np.ndarray, assignments: List[str], testSize: float = 0.1) -> None:
        t0 = time.time()
        self._setUniqueAssignments(assignments)
        assignments: List[int] = [self._uniqueAssignments.index(ass) for ass in assignments]

        X_train, X_test, y_train, y_test = train_test_split(spectra, assignments, test_size=testSize, random_state=42)
        X_train = tf.cast(X_train, tf.float32)
        X_test = tf.cast(X_test, tf.float32)
        y_train = tf.cast(y_train, tf.int32)
        y_test = tf.cast(y_test, tf.int32)
        history = self._model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
                                  batch_size=32, shuffle=True, verbose=0)
        print(f'training took {round(time.time()-t0, 2)} seconds')

    def evaluateSpectra(self, spectra: np.ndarray, cutoff: float = 0.0) -> List[str]:
        spectra = tf.cast(spectra, tf.float32)
        probabilities: np.ndarray = self._model.predict(spectra)
        return self._probabilityMatrix2Prediction(probabilities, cutoff)


class RandomDecisionForest(BaseClassifier):
    def __init__(self, descLib: 'DescriptorLibrary'):
        super(RandomDecisionForest, self).__init__("Random Forest")
        self._descLib: 'DescriptorLibrary' = descLib
        self._wavenums: np.ndarray = None
        self._clf: RandomForestClassifier = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)

    def setWavenumbers(self, wavenumbers: np.ndarray) -> None:
        self._wavenums = wavenumbers

    def trainWithSpectra(self, trainSpecs: np.ndarray, assignments: List[str], testSize=0.1) -> None:
        t0 = time.time()
        self._setUniqueAssignments(assignments)
        specsForDescLib = self._getSpecsForDesclib(trainSpecs)
        featureMat: np.ndarray = self._descLib.getCorrelationMatrixToSpectra(specsForDescLib)
        assert len(assignments) == featureMat.shape[0]

        featureMat = StandardScaler().fit_transform(featureMat)
        assignments: List[int] = [self._uniqueAssignments.index(ass) for ass in assignments]
        X_train, X_test, y_train, y_test = train_test_split(featureMat, assignments, test_size=testSize, random_state=42)
        print(f'creating and training rdf on {len(y_train)} samples with {X_train.shape[1]} features')
        self._clf.fit(X_train, y_train)
        score = self._clf.score(X_test, y_test)
        print(f'Classifier score is {round(score, 2)}, training and testing took {round(time.time()-t0, 2)} seconds.')

    def evaluateSpectra(self, spectra: np.ndarray, cutoff: float = 0.0) -> List[str]:
        specsForDescLib = self._getSpecsForDesclib(spectra)
        featureMat: np.ndarray = self._descLib.getCorrelationMatrixToSpectra(specsForDescLib)
        featureMat = StandardScaler().fit_transform(featureMat)
        probabilities: np.ndarray = self._clf.predict_proba(featureMat)
        return self._probabilityMatrix2Prediction(probabilities, cutoff)

    def _getSpecsForDesclib(self, specArray: np.ndarray) -> np.ndarray:
        """
        Takes (MxN) array spectra of M spectra with N wavenumbers and returns
        (N x M+1) array of M spectra with N wavenumbers, wavenumbers in first column
        :return: columnWise spec array
        """
        assert self._wavenums is not None, "Wavenumbers haven't been yet set for RDF classifier"
        return np.vstack((self._wavenums, specArray)).transpose()
