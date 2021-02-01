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
import mysql.connector
import hashlib
import json
import os
from io import StringIO
from typing import List, TYPE_CHECKING
if TYPE_CHECKING:
    from mysql.connector.connection import MySQLConnection
    from mysql.connector.cursor import MySQLCursor


def uploadSpectra(spectra: np.ndarray, assigment: str, specType: str, numAcc: int = 1, intTime: float = 0.0, magn: int = 20) -> None:
    """

    :param spectra: (NxM) shape array of M spectra with N wavenumbers
    :param assigment:
    :param specType:
    :param numAcc:
    :param intTime:
    :param magn:
    :return:
    """
    configPath = getConfigPath()
    print(configPath)
    if not os.path.exists(configPath):
        raise FileNotFoundError('SQL Login data not found.')

    with open(configPath, "r") as fp:
        config = json.load(fp)

    connection: 'MySQLConnection' = None
    connection = mysql.connector.connect(**config)
    assert connection is not None, f'Failed to connect to SQL database with config {config}'
    assert assigment in getPolymerNames(connection)
    assert specType in getSpectraTypes(connection)

    for i in range(spectra.shape[0]-1):
        curSpec: np.ndarray = spectra[[0, i+1], :]
        if np.count_nonzero(curSpec[1, :]) == 0:
            print(f'Skipping upload of  {assigment}, {intTime}, {magn}, spec is all zeros..')
        else:
            specToInsert: str = specToString(curSpec)
            specHash: str = hashSpecArray(curSpec)
            cursor = connection.cursor()
            try:
                cursor.execute(f"""INSERT INTO spectrum (spec_type, assignment, specdata, spechash, AcquisitionTime, numAcummulations, Magnification)
                               VALUES ("{specType}", "{assigment}", "{specToInsert}", "{specHash}", "{intTime}", "{numAcc}", "{magn}");""")
                connection.commit()
            except mysql.connector.Error as err:
                print(f'Error in uploading {assigment}, {intTime}, {magn}\n', err)

    if connection is not None:
        connection.disconnect()


def getPolymerNames(conn: 'MySQLConnection') -> List[str]:
    cursor: 'MySQLCursor' = conn.cursor()
    cursor.execute("SELECT material_name FROM material_type")
    return [row[0] for row in cursor]


def getSpectraTypes(conn: 'MySQLConnection') -> List[str]:
    cursor: 'MySQLCursor' = conn.cursor()
    cursor.execute("SELECT spectrum_type FROM spec_type")
    return [row[0] for row in cursor]


def specToString(spec: np.ndarray) -> str:
    fp = StringIO()
    np.savetxt(fp, spec)
    fp.seek(0)
    return fp.read()


def hashSpecArray(arr: np.ndarray) -> str:
    specString = arr.copy().tobytes()
    hashed = hashlib.sha256(specString)
    return hashed.hexdigest()


def getConfigPath() -> str:
    curPath: str = os.path.dirname(__file__)
    return os.path.join(os.path.dirname(curPath), "SQLCredentials.txt")