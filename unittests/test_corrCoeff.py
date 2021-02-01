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
from scipy.stats import pearsonr
from cythonModules import corrCoeff


class TestCorrCoeff(unittest.TestCase):
    def test_corrcoeff(self):
        np.random.seed(42)
        for _ in range(10):
            pattern = np.random.rand(10)
            specs = np.random.rand(10, 4)
            cythonCorrs = corrCoeff.getCorrelationCoefficients(pattern, specs)
            for i, corr in enumerate(cythonCorrs):
                self.assertAlmostEqual(corr, pearsonr(pattern, specs[:, i])[0])

