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


import unittest
import numpy as np
from typing import List

import os
from importData import get_database, load_specCSVs_from_subfolders
from specCorrelation import correlate_spectra
from functions import mapSpectrasetsToSameWavenumbers


class TestSpecEvaluation(unittest.TestCase):
    def test_match_specSets(self) -> None:
        specs1: np.ndarray = np.zeros((10, 21))  # i.e., 20 spectra with 10 wavenumbers
        specs1[:, 0] = np.arange(10)
        specs1[:, 1:] = np.random.rand(10, 20)
        specs2: np.ndarray = np.zeros((5, 11))  # i.e. 10 spectra with 5 wavenumbers
        specs2[:, 0] = np.arange(5)
        specs2[:, 1:] = np.random.rand(5, 10)

        newSpecs1, newSpecs2 = mapSpectrasetsToSameWavenumbers(specs1, specs2)
        self.assertTrue(newSpecs1.shape[0] == newSpecs2.shape[0] == specs2.shape[0])
        self.assertTrue(np.array_equal(newSpecs2, specs2))
        self.assertTrue(np.array_equal(newSpecs1[:, 0], newSpecs2[:, 0]))

        newSpecs1, newSpecs2 = mapSpectrasetsToSameWavenumbers(specs2, specs1)
        self.assertTrue(newSpecs1.shape[0] == newSpecs2.shape[0] == specs2.shape[0])
        self.assertTrue(np.array_equal(newSpecs1, specs2))
        self.assertTrue(np.array_equal(newSpecs1[:, 0], newSpecs2[:, 0]))

    def test_specCorr(self) -> None:
        projectFolder = os.path.dirname(os.getcwd())
        specFolder = os.path.join(projectFolder, "Sample Spectra Plastic")
        names, specs = load_specCSVs_from_subfolders(specFolder, maxSpectraPerFolder=5)  # we don't need aaaall the spectra...
        db = get_database()

        # just to see that no errors occur..
        results: List[str] = correlate_spectra(specs, db, cutoff=0.0)

