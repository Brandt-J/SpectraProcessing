import unittest
from importData import *


class TestSetup(unittest.TestCase):
    def test_dbSetup(self) -> None:
        newDB: Database = get_database()
        self.assertEqual(type(newDB), Database)
        self.assertEqual(newDB.getNumberOfSpectra(), 10)

    def test_get_numbers_from_line(self):
        numbers: Tuple[float, float] = get_numbers_from_line('3,992094e+002;2,818067e-001\n')
        assert len(numbers) == 2
        self.assertEqual(numbers[0], 3.992094e+002)
        self.assertEqual(numbers[1], 2.818067e-001)

        numbers: Tuple[float, float] = get_numbers_from_line('674.991821289, 0.007900199\n')
        assert len(numbers) == 2
        self.assertEqual(numbers[0], 674.991821289)
        self.assertEqual(numbers[1], 0.007900199)


