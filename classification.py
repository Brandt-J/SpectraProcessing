from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from typing import List, Tuple


def test_randForestClassifier(featureMatrix: np.ndarray, assignments: List[str]) -> Tuple[RandomForestClassifier, List[str]]:
    assert len(assignments) == featureMatrix.shape[0]
    X: np.ndarray = StandardScaler().fit_transform(featureMatrix)
    uniqueAssignments: List[str] = list(np.unique(assignments))
    y: List[int] = [uniqueAssignments.index(ass) for ass in assignments]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

    print(f'creating and training rdf on {len(y_train)} samples with {X_train.shape[1]} features')
    clf: RandomForestClassifier = RandomForestClassifier()
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print(f'Classifier score is {score}')
    return clf, uniqueAssignments
