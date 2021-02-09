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
import time
import os
from typing import List
import numpy as np

import importData as io
from descriptors import DescriptorLibrary
from classification import RandomDecisionForest
from evaluationTest import testEvaluationOnSpectra
from specCorrelation import CorrelationMode


pathSampleSpec: str = r'Sample Spectra/sampleSpectra.npy'
pathSampleAssignments: str = r'Sample Spectra/origResults.txt'
preprocessSpectra: bool = True  # Whether or not subtract baseline and normalize spectra for database search
correlationModes: List[CorrelationMode] = [CorrelationMode.PEARSON, CorrelationMode.SFEC]
nMaxDBSpecs: int = 10  # maximum number of spectra in the database
nMaxDesc: int = 20  # maximum number of descriptors per spectrum

testSpectra, origResults = io.getTestSpectra(pathSampleSpec, pathSampleAssignments, forceRegenerate=True,
                                             maxSpecPerFolder=100)

database = io.get_database(maxSpectra=nMaxDBSpecs)
database.preprocessSpectra()

descriptors: DescriptorLibrary = DescriptorLibrary()
descriptors.generate_from_specDatabase(database, maxDescPerSet=200)
descriptors.optimize_descriptorSets(maxDescriptorsPerSet=nMaxDesc)
rdf: RandomDecisionForest = RandomDecisionForest(descriptors)
rdf.trainWithSpectra(testSpectra, origResults)

figure, results = testEvaluationOnSpectra(testSpectra, origResults, database, rdf, preprocessSpectra, numIterations=5,
                                          corrModes=correlationModes)
figure.show()
