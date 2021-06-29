from imblearn import over_sampling, under_sampling
from enum import Enum
from typing import Tuple, List, TYPE_CHECKING
if TYPE_CHECKING:
    import numpy as np


class BalanceMode(Enum):
    NONE = 0
    UNDER_RAND = 1
    OVER_RAND = 2
    OVER_SMOTE = 3
    OVER_ADASYN = 4


def balanceDataset(dataArray: 'np.ndarray', assignments: List[str],
                   balanceMode: BalanceMode = BalanceMode.NONE) -> Tuple['np.ndarray', List[str]]:

    if balanceMode == BalanceMode.NONE:
        newData, newAssignments = dataArray.copy(), assignments.copy()
    else:
        if balanceMode == BalanceMode.UNDER_RAND:
            sampler = under_sampling.RandomUnderSampler(random_state=42)
        elif balanceMode == BalanceMode.OVER_SMOTE:
            sampler = over_sampling.SMOTE(random_state=42)
        elif balanceMode == BalanceMode.OVER_ADASYN:
            sampler = over_sampling.ADASYN(random_state=42)
        elif balanceMode == BalanceMode.OVER_RAND:
            sampler = over_sampling.RandomOverSampler(random_state=42)

        newData, newAssignments = sampler.fit_resample(dataArray, assignments)

    return newData, newAssignments
