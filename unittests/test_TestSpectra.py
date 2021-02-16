import unittest
from testSpectra import TestSpectra, MatType


class TestTestSpectra(unittest.TestCase):
    def test_MaterialType(self) -> None:
        plastType: MatType = MatType.PLASTIC
        nonPlastType: MatType = MatType.NON_PLASTIC
        plastString: str = "plastic"
        nonPlastString: str = "non_plastic"
        self.assertEqual(plastType.toString(), plastString)
        self.assertEqual(nonPlastType.toString(), nonPlastString)

        self.assertEqual(MatType.fromString(plastString), plastType)
        self.assertEqual(MatType.fromString(nonPlastString), nonPlastType)

    def test_getPlasticContent(self) -> None:
        testSpec: TestSpectra = TestSpectra()
        testSpec._types = [MatType.PLASTIC] * 10 + [MatType.NON_PLASTIC] * 10
        self.assertEqual(testSpec.getPlasticContent(), 0.5)

        testSpec._types = [MatType.PLASTIC] * 1 + [MatType.NON_PLASTIC] * 9
        self.assertEqual(testSpec.getPlasticContent(), 0.1)

        testSpec._types = [MatType.PLASTIC] * 1 + [MatType.NON_PLASTIC] * 99
        self.assertEqual(testSpec.getPlasticContent(), 0.01)
