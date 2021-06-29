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
from specCorrelation import Database
import numpy as np
from typing import List, Tuple
import functions as fn

# Path to folders for regenerating data
sampleDirectory: str = 'Sample Spectra Plastic'
refPlasticDirectory: str = 'Reference Spectra Plastic'
refNonPlasticDirectory: str = 'Reference Spectra Non Plastic'


def get_database(maxSpectra: int = np.inf, includeNonPlastic: bool = False) -> 'Database':
    newDB: Database = Database('StandardPolymers')
    projectPath = os.getcwd()
    if projectPath.endswith("unittests"):
        projectPath = os.path.dirname(projectPath)

    specNames, spectra = load_specCSVs_from_directory(os.path.join(projectPath, 'Reference Spectra Plastic'))
    if includeNonPlastic:
        nonPlastNames, nonPlastSpectra = load_specCSVs_from_directory(os.path.join(projectPath, 'Reference Spectra Non Plastic'))
        spectra, nonPlastSpectra = fn.mapSpectrasetsToSameWavenumbers(spectra, nonPlastSpectra)
        assert spectra.shape[0] == nonPlastSpectra.shape[0]
        spectra = np.hstack((spectra, nonPlastSpectra[:, 1:]))
        specNames = specNames + nonPlastNames

    if len(specNames) > maxSpectra:
        specNames, spectra = fn.getNMostDifferentSpectra(specNames, spectra, maxSpectra)

    for index, name in enumerate(specNames):
        newDB.addSpectrum(name, spectra[:, [0, index + 1]])

    return newDB


def load_specCSVs_from_subfolders(parentDirectory: str, maxSpectraPerFolder: int = 1e6) -> Tuple[List[str], np.ndarray]:
    """
    Goes through the subdirectories of the given parentDirectory and loads all the spectra there.
    :param parentDirectory:
    :param maxSpectraPerFolder:
    :return:
    """
    wd = os.getcwd()
    # just to make compatible with unittest...
    if os.path.basename(wd) == 'unittests':
        directory = os.path.join(os.path.dirname(wd), parentDirectory)

    subfolders = [x[1] for x in os.walk(parentDirectory)][0]
    spectraSets: List[Tuple[List[str], np.ndarray]] = []
    for folderName in subfolders:
        if not folderName.endswith('_skip'):
            path = os.path.join(parentDirectory, folderName)
            spectraSets.append(load_specCSVs_from_directory(path, fixName=folderName, maxSpectra=maxSpectraPerFolder))

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


def load_specCSVs_from_directory(path: str, fixName: str = None, maxSpectra=1e6) -> Tuple[List[str], np.ndarray]:
    """
    Reads Spectra from CSV viles in path. If given, a fix name is assigned to each spectrum
    :param path: Directory path
    :param fixName: If None, each spectrum has the filename as name, otherwise the indicated fixName
    :param maxSpectra: Max number of spectra to take.
    :return: Tuple[Assignment List, spectra array]
    """
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


def getNonPlasticNames() -> List[str]:
    files = os.listdir(refNonPlasticDirectory)
    return [name.split('.csv')[0] for name in files]


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
        numbers = float(line[0].replace(',', '.')), float(line[1].replace(',', '.'))
    except AssertionError as e:
        line = origline.split(',')
        assert len(line) == 2
        numbers = float(line[0]), float(line[1])
    return numbers


if __name__ == '__main__':
    names, specs = load_specCSVs_from_subfolders(r"C:\Users\xbrjos\Desktop\Python\SpectraProcesing\Sample Spectra Plastic")
    np.savetxt(r"C:\Users\xbrjos\Desktop\Python\SpecReconstruction\MicroFTIRSpectra\assignments.txt", names, fmt='%s')
    np.save(r"C:\Users\xbrjos\Desktop\Python\SpecReconstruction\MicroFTIRSpectra\polymers.npy", specs)