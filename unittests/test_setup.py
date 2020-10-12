import unittest
import numpy as np
from importData import get_database, get_spectra, Database, specProc


class TestSetup(unittest.TestCase):
    def test_dbSetup(self) -> None:
        newDB: Database = get_database()
        self.assertEqual(type(newDB), Database)
        self.assertEqual(newDB.getNumberOfSpectra(), 16)

    def test_testSpecSetup(self) -> None:
        spectra: np.ndarray = get_spectra()
        self.assertEqual(type(spectra), np.ndarray)
        self.assertEqual(spectra.shape, (863, 54))
