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


import os
import sys
from pathlib import Path

curPath: Path = Path(os.path.realpath(__file__))
projectPath: Path = curPath.parent
sys.path.append(str(projectPath.parent))
from gepardevaluation.analysis import spectraProcessing
from gepardevaluation.analysis.specDataBases import Database
import numpy as np
from typing import List, Tuple

from functions import getNMostDifferentSpectraIndices

specProc = spectraProcessing


def get_database(maxSpectra: int = np.inf) -> Database:
    newDB: Database = Database('StandardPolymers')
    specNames, spectra = read_from_directory(os.path.join(projectPath, 'Reference Spectra'))
    if len(specNames) > maxSpectra:
        indices = getNMostDifferentSpectraIndices(spectra, maxSpectra)
        specNames = [specNames[i] for i in indices]
        print('chosen Spectra:', specNames)
        indices = [0] + [i+1 for i in indices]
        spectra = spectra[:, indices]

    for index, name in enumerate(specNames):
        newDB.addSpectrum(name, spectra[:, [0, index + 1]])

    return newDB


def get_test_spectra(maxSpectraPerFolder=1e6) -> Tuple[List[str], np.ndarray]:
    sampleDirectory = 'Sample Spectra'
    wd = os.getcwd()
    # just to make compatible with unittest...
    if os.path.basename(wd) == 'unittests':
        sampleDirectory = os.path.join(os.path.dirname(wd), sampleDirectory)

    subfolders = [x[1] for x in os.walk(sampleDirectory)][0]
    spectraSets: List[Tuple[List[str], np.ndarray]] = []
    for folderName in subfolders:
        if not folderName.endswith('_skip'):
            path = os.path.join(sampleDirectory, folderName)
            spectraSets.append(read_from_directory(path, fixName=folderName, maxSpectra=maxSpectraPerFolder))

    wavenums: List[np.ndarray] = [spectra[:, 0] for _, spectra in spectraSets]
    minCommon, maxCommon = max([curWavenums[0] for curWavenums in wavenums]), \
                           max([curWavenums[-1] for curWavenums in wavenums])
    diffs: List[np.ndarray] = [np.diff(curWavenums) for curWavenums in wavenums]
    meanDiff = np.mean(np.concatenate(diffs))

    finalWavenums: np.ndarray = np.linspace(minCommon, maxCommon, round((maxCommon-minCommon) / meanDiff))
    allNames: np.ndarray = np.concatenate([names for names, _ in spectraSets])
    allSpectra: np.ndarray = np.zeros((finalWavenums.shape[0], allNames.shape[0]+1))
    allSpectra[:, 0] = finalWavenums

    spectraAdded: int = 0
    for _, spectra in spectraSets:
        numSpecs: int = spectra.shape[1] - 1
        finalSpecAdded: int = spectraAdded + numSpecs
        curWavenums: np.ndarray = spectra[:, 0]
        for i, wavenum in enumerate(finalWavenums):
            closestIndex = np.argmin(np.abs(curWavenums - wavenum))
            allSpectra[i, spectraAdded+1:finalSpecAdded+1] = spectra[closestIndex, 1:]

        spectraAdded = finalSpecAdded

    return allNames, allSpectra


def read_from_directory(path: str, fixName: str = None, maxSpectra=1e6) -> Tuple[List[str], np.ndarray]:
    spectra: np.ndarray = None
    names: list = []
    numSpectra: int = 0
    for file in os.listdir(path):
        if file.lower().endswith('.csv') and numSpectra < maxSpectra:
            curSpec: list = []
            specName = fixName if fixName is not None else file.lower().split('.csv')[0]
            names.append(specName)

            with open(os.path.join(path, file), 'r') as fp:

                if spectra is None:
                    wavenumbers = []
                    # for index, row in enumerate(reader):
                    for line in fp.readlines():
                        wavenum, intensity = get_numbers_from_line(line)
                        curSpec.append(intensity)
                        wavenumbers.append(wavenum)

                    spectra = np.array(wavenumbers)
                else:
                    tmpSpec = []
                    tmpWavenumber = []
                    for line in fp.readlines():
                        wavenum, intensity = get_numbers_from_line(line)
                        tmpSpec.append(intensity)
                        tmpWavenumber.append(wavenum)

                    tmpSpec = np.array(tmpSpec)
                    tmpWavenumber = np.array(tmpWavenumber)
                    for number in spectra[:, 0]:
                        index = np.argmin(np.abs(tmpWavenumber - number))
                        curSpec.append(tmpSpec[index])

                if len(spectra.shape) == 1:
                    spectra = np.append(spectra[:, np.newaxis], np.array(curSpec)[:, np.newaxis], axis=1)
                else:
                    spectra = np.append(spectra, np.array(curSpec)[:, np.newaxis], axis=1)

                numSpectra += 1

    return names, spectra


def get_numbers_from_line(line: str) -> Tuple[float, float]:
    """
    Takes a line from a csv or txt document and checks for delimiter and decimal separator to yield exactly
    two float numbers
    :param line:
    :return: the two float numbers as Tuple
    """
    origline = line.strip()
    try:
        line = origline.split(';')
        assert len(line) == 2
        numbers: Tuple[float, float] = float(line[0].replace(',', '.')), float(line[1].replace(',', '.'))
    except AssertionError:
        line = origline.split(',')
        assert len(line) == 2
        numbers: Tuple[float, float] = float(line[0]), float(line[1])
    except AssertionError as e:
        print(e)
        raise
    return numbers
