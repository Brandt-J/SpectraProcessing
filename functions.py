import numpy as np
from collections import Counter
from typing import List, Tuple, Dict


def compareResultLists(trueList: List[str], estimatedList: List[str]) -> Tuple[float, Dict[str, float]]:
    """
    Compares two assignment lists.
    :param trueList: List of true assignments
    :param estimatedList: List of estimated assignments
    :returns: Tuple(agreement in percent, Dictionary of agreements in percent per type of assignment)
    """
    assert len(trueList) == len(estimatedList)

    replaceDict: dict = {'ldpe': 'pe',
                         'hdpe': 'pe'}

    for i in range(len(trueList)):
        trueList[i] = trueList[i].lower()
        estimatedList[i] = estimatedList[i].lower()

        if trueList[i] in replaceDict.keys():
            trueList[i] = replaceDict[trueList[i]]

        if estimatedList[i] in replaceDict.keys():
            estimatedList[i] = replaceDict[estimatedList[i]]

    totalQuality: float = np.count_nonzero(np.array(estimatedList) == np.array(trueList)) / len(trueList) * 100

    resultDict: Dict[str, float] = {}
    for name in np.unique(trueList):
        numCorrect = Counter(trueList).get(name)
        estCounter = Counter(estimatedList)
        numEstimated = estCounter.get(name) if name in estCounter.keys() else 0
        resultDict[name] = numEstimated / numCorrect * 100
        # print(name, numCorrect, numEstimated, resultDict[name])

    sortedList: List[Tuple[str, str]] = sorted(resultDict.items(), key=lambda x: x[1], reverse=True)
    resultDict = {entry[0]: entry[1] for entry in sortedList}
    return totalQuality, resultDict
