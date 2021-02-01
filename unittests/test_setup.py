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


