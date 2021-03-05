import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import savgol_filter

import importData as io
from Denoising import Denoiser, getTrainTestSpecs, prepareSpecSet
from testSpectra import TestSpectra


os.chdir(os.path.dirname(os.getcwd()))


specLength = 512

numTrainSpectra, numVariationsTrain, numTestSpectra, numVariationsTest = 100, 10000, 20, 20
db: 'io.Database' = io.get_database(maxSpectra=numTrainSpectra + numTestSpectra)
db.preprocessSpectra()
db.reduceSpecsToNWavenumbers(specLength)

# Get From Database Spectra
# specs: np.ndarray = db.getSpectra()[:, 1:]
# trainSpectra = np.tile(specs[:, :numTrainSpectra], (1, numVariationsTrain))
# testSpectra = np.tile(specs[:, numTrainSpectra:], (1, numVariationsTest))
#
# noiseLevel = 0.2
# np.random.seed(42)
# noisyTrainSpectra = trainSpectra + np.random.rand(trainSpectra.shape[0], trainSpectra.shape[1])
# noisyTestSpectra = testSpectra + np.random.rand(testSpectra.shape[0], testSpectra.shape[1])
#
# trainSpectra = prepareSpecSet(trainSpectra)
# testSpectra = prepareSpecSet(testSpectra)
# noisyTrainSpectra = prepareSpecSet(noisyTrainSpectra)
# noisyTestSpectra = prepareSpecSet(noisyTestSpectra)


# Get TestData from TestSpectraObject
testObj: TestSpectra = TestSpectra()
# testObj.loadFromNPY()
testObj.generateFromSampleDir()
testObj.normalizeSpectra()
testObj.mapSpectraToWavenumbers(db.getWavenumbers())
testObj.saveToNPY()
trainSpectra, noisyTrainSpectra, testSpectra, noisyTestSpectra = getTrainTestSpecs(testObj, db, trainSplit=0.3)


print(f'Training with {len(trainSpectra)} spectra, testing with {len(testSpectra)} spectra')
denoiser = Denoiser(specLength=specLength)
denoiser.compile(optimizer='adam', loss='mae')
denoiser.fit(noisyTrainSpectra, trainSpectra,
             epochs=5,
             validation_data=(noisyTestSpectra, testSpectra), shuffle=True)
reconstructedSpecs = denoiser.call(noisyTestSpectra)

n = 6
plotIndices = np.random.randint(0, len(reconstructedSpecs), n)
corrs = np.zeros((len(reconstructedSpecs), 2))
plotNumber = 1
fig: plt.Figure = plt.figure(figsize=(20, 7))
for i in range(len(reconstructedSpecs)):
    orig = testSpectra[i]
    noisy = noisyTestSpectra[i]
    reconst = reconstructedSpecs[i]
    if np.array_equal(reconst, np.zeros(specLength)):
        print('all zeros at index', i)
    savgol = savgol_filter(noisy, window_length=21, polyorder=5)

    corrNN = np.round(np.corrcoef(orig, reconst)[0, 1] * 100)
    if np.isnan(corrNN):
        corrNN = 0

    corrSavGol = np.round(np.corrcoef(orig, savgol)[0, 1] * 100)
    corrs[i, 0] = corrNN
    corrs[i, 1] = corrSavGol
    # if corrSavGol > corrNN and plotNumber <= n:
    if i in plotIndices:
        print(f'Spec {i+1}: Corr NN: {corrNN} %, corr savgol: {corrSavGol} %')
        ax = fig.add_subplot(3, 2, plotNumber)
        ax.plot(noisy, color='blue', alpha=0.4)

        ax.plot(orig, color='blue', label='original')
        ax.plot(reconst + 0.6, color='green', label=f'neuronal net\n{corrNN} % Correlation')
        ax.plot(savgol + 0.8, color='orange', label=f'savgol filter\n{corrSavGol} % Correlation')
        ax.legend()
        plotNumber += 1

fig.tight_layout()
fig.show()


summary = f'mean NN: {np.mean(corrs[:, 0])}, mean savgol: {np.mean(corrs[:, 1])}'
print(summary)
boxfig = plt.figure()
box_ax: plt.Axes = boxfig.add_subplot()
box_ax.boxplot(corrs, labels=['Neuronal\nNet', 'Salvitzky-\nGolay'], widths=[0.6, 0.6], showfliers=False)
box_ax.set_title(summary)
box_ax.set_ylabel('Pearson Correlation (%)')
boxfig.show()
