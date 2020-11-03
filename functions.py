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

    resultDict: Dict[str, List[int]] = {}  # key: name, value: [numCorrect, numFalse]
    totalErrors: int = 0
    for expected, actual in zip(trueList, estimatedList):
        correct = expected == actual
        if not correct:
            totalErrors += 1

        if expected not in resultDict.keys():
            resultDict[expected] = [1, 0] if correct else [0, 1]
        else:
            if correct:
                resultDict[expected][0] += 1
            else:
                resultDict[expected][1] += 1

    for name, trueFalse in resultDict.items():
        resultDict[name] = trueFalse[0] / (trueFalse[0] + trueFalse[1]) * 100

    sortedList: List[Tuple[str, str]] = sorted(resultDict.items(), key=lambda x: x[1], reverse=True)
    resultDict = {entry[0]: entry[1] for entry in sortedList}
    totalQuality = 100 - (totalErrors / len(trueList)) * 100
    return totalQuality, resultDict
