import numpy as np
from collections import Counter
from typing import *


def compareResultLists(trueList: List[str], estimatedList: List[str]) -> Dict[str, float]:
    assert len(trueList) == len(estimatedList)
    resultDict: Dict[str, float] = {}
    for name in np.unique(trueList):
        numCorrect = Counter(trueList).get(name)
        estCounter = Counter(estimatedList)
        numEstimated = estCounter.get(name) if name in estCounter.keys() else 0
        resultDict[name] = numEstimated / numCorrect * 100

    sortedList: List[Tuple[str, str]] = sorted(resultDict.items(), key=lambda x: x[1], reverse=True)
    resultDict = {entry[0]: entry[1] for entry in sortedList}
    return resultDict
