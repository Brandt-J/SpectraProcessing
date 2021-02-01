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


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn import over_sampling, under_sampling
import numpy as np
from collections import Counter
from typing import List, Tuple


def balanceDataset(featureMat: np.ndarray, assignments: List[str]) -> Tuple[np.ndarray, List[str]]:
    sampler = over_sampling.SMOTE(random_state=42)
    # sampler = over_sampling.ADASYN(random_state=42)
    # sampler = over_sampling.RandomOverSampler(random_state=42)
    # sampler = under_sampling.ClusterCentroids(random_state=42)

    print('balancing with:', sampler)
    newData, newAssignments = sampler.fit_resample(featureMat, assignments)
    return newData, newAssignments


def test_randForestClassifier(featureMatrix: np.ndarray, assignments: List[str]) -> Tuple[RandomForestClassifier, List[str]]:
    assert len(assignments) == featureMatrix.shape[0]
    X: np.ndarray = featureMatrix
    uniqueAssignments: List[str] = list(np.unique(assignments))
    y: List[int] = [uniqueAssignments.index(ass) for ass in assignments]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

    print(f'creating and training rdf on {len(y_train)} samples with {X_train.shape[1]} features')
    clf: RandomForestClassifier = RandomForestClassifier(n_estimators=100, max_features='sqrt')
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f'Classifier score is {score}')

    return clf, uniqueAssignments
