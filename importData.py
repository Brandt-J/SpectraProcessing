import os
import sys
from pathlib import Path

curPath: Path = Path(os.path.realpath(__file__))
projectPath: Path = curPath.parent
sys.path.append(str(projectPath.parent))
from gepardevaluation.analysis import spectraProcessing
from gepardevaluation.analysis.specDataBases import Database
import numpy as np
import csv


specProc = spectraProcessing


def get_database() -> Database:
    def read_from_directory(path: str) -> np.ndarray:
        spectra: np.ndarray = None
        names: list = []
        for file in os.listdir(path):
            if file.endswith('.CSV'):
                curSpec: list = []
                names.append(file.split('.CSV')[0])
                with open(os.path.join(path, file), 'r') as fp:
                    reader = csv.reader(fp, delimiter=';')
                    if spectra is None:
                        wavenumbers = []
                        for index, row in enumerate(reader):
                            curWavenumber = float(row[0].replace(',', '.'))
                            curSpec.append(float(row[1].replace(',', '.')))
                            wavenumbers.append(curWavenumber)

                        spectra = np.array(wavenumbers)
                    else:
                        tmpSpec = []
                        tmpWavenumber = []
                        for index, row in enumerate(reader):
                            curWavenumber = float(row[0].replace(',', '.'))
                            tmpSpec.append(float(row[1].replace(',', '.')))
                            tmpWavenumber.append(curWavenumber)
                        tmpSpec = np.array(tmpSpec)
                        tmpWavenumber = np.array(tmpWavenumber)
                        for number in spectra[:, 0]:
                            index = np.argmin(np.abs(tmpWavenumber - number))
                            curSpec.append(tmpSpec[index])

                    if len(spectra.shape) == 1:
                        spectra = np.append(spectra[:, np.newaxis], np.array(curSpec)[:, np.newaxis], axis=1)
                    else:
                        spectra = np.append(spectra, np.array(curSpec)[:, np.newaxis], axis=1)

        return names, spectra

    newDB: Database = Database('StandardPolymers')
    specNames, spectra = read_from_directory(os.path.join(projectPath, 'Reference Spectra'))

    for index, name in enumerate(specNames):
        newDB.addSpectrum(name, spectra[:, [0, index + 1]])

    return newDB


def get_spectra() -> np.ndarray:
    return np.loadtxt(os.path.join(projectPath, 'Sample Spectra', 'FTIR PET.txt'))
