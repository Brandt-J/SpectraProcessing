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

