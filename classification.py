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
if TYPE_CHECKING:
    from descriptors import DescriptorLibrary


class RandomDecisionForest(object):
    def __init__(self, descLib: 'DescriptorLibrary'):
        super(RandomDecisionForest, self).__init__()
        self._descLib: 'DescriptorLibrary' = descLib
        self._clf: RandomForestClassifier = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)
        self._uniqueAssignments: List[str] = []  #

    def trainWithSpectra(self, trainSpecs: np.ndarray, assignments: List[str]) -> None:
        t0 = time.time()
        featureMat: np.ndarray = self._descLib.getCorrelationMatrixToSpectra(trainSpecs)
        self._uniqueAssignments = list(np.unique(assignments))
        assert len(assignments) == featureMat.shape[0]
        featureMat = StandardScaler().fit_transform(featureMat)
        y: List[int] = [self._uniqueAssignments.index(ass) for ass in assignments]
        X, y = balanceDataset(featureMat, y)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

        print(f'creating and training rdf on {len(y_train)} samples with {X_train.shape[1]} features')
        self._clf.fit(X_train, y_train)
        score = self._clf.score(X_test, y_test)
        print(f'Classifier score is {round(score, 2)}, training and testing took {round(time.time()-t0, 2)} seconds.')

    def evaluateSpectra(self, spectra: np.ndarray, cutoff: float = 0.0) -> List[str]:
        """
        Takes an array of spectra and returns the determined assignments for each spectrum. "Unknown" is returned
        if class probability dropbs below the cutoff.
        :param spectra: (NxM) shape array of spectra of M-1 spectra with N wavenumbers (wavenumbers in first column)
        :param cutoff: (0...1) value for required class probability for assigmnent. Will be "unknown" otherwise.
        :return:
        """
        featureMat: np.ndarray = self._descLib.getCorrelationMatrixToSpectra(spectra)
        featureMat = StandardScaler().fit_transform(featureMat)
        probabilities: np.ndarray = self._clf.predict_proba(featureMat)
        descriptorResults: List[str] = []
        for i, prob in enumerate(probabilities):
            maxProb = np.max(prob)
            if maxProb >= cutoff:
                maxIndices: np.ndarray = np.where(prob == maxProb)[0]
                descriptorResults.append(self._uniqueAssignments[int(maxIndices[0])])
            else:
                descriptorResults.append("unknown")
        return descriptorResults


def balanceDataset(featureMat: np.ndarray, assignments: List[str]) -> Tuple[np.ndarray, List[str]]:
    sampler = over_sampling.SMOTE(random_state=42)
    # sampler = over_sampling.ADASYN(random_state=42)
    # sampler = over_sampling.RandomOverSampler(random_state=42)
    # sampler = under_sampling.ClusterCentroids(random_state=42)

    print('balancing with:', sampler)
    newData, newAssignments = sampler.fit_resample(featureMat, assignments)
    return newData, newAssignments

