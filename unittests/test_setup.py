import unittest
import numpy as np
from importData import getDatabase, getSpectra, Database


class TestSetup(unittest.TestCase):
    def test_dbSetup(self) -> None:
        newDB: Database = getDatabase()
        self.assertEqual(type(newDB), Database)
        self.assertEqual(newDB.getNumberOfSpectra(), 16)

    def test_testSpecSetup(self) -> None:
        spectra: np.ndarray = getSpectra()
        self.assertEqual(type(spectra), np.ndarray)
        self.assertEqual(spectra.shape, (863, 54))
