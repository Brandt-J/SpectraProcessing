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
from scipy.signal import gaussian
import matplotlib.pyplot as plt

from distort import *
from cythonModules.corrCoeff import sfec
from processing import normalizeIntensities


peakLength: int = 101
peak: np.ndarray = gaussian(peakLength, 7)

np.random.seed(42)
noisy: np.ndarray = normalizeIntensities(peak + np.random.rand(peakLength)*0.5)
offset: np.ndarray = peak + 0.4
tilted: np.ndarray = normalizeIntensities(peak + np.linspace(0, 0.4, peakLength))
bended: np.ndarray = normalizeIntensities(peak - gaussian(peakLength, 21)*0.5)

shifted: np.ndarray = np.zeros_like(peak)
shiftAmount = int(round(peakLength / 30))
shifted[shiftAmount:] = peak[:peakLength-shiftAmount]

for i, distorted in enumerate([noisy, offset, tilted, bended, shifted]):
    plt.subplot(2, 3, i+1)
    plt.plot(peak)
    plt.plot(distorted)
    corr_pearson = np.corrcoef(peak, distorted)[0, 1]
    corr_sfec = sfec(peak, distorted)
    plt.title(f'Pearson: {round(corr_pearson, 3)}, SFEC: {round(corr_sfec, 3)}')

plt.show()
plt.tight_layout()
