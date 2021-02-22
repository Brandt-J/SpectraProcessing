import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Input, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import savgol_filter

import importData as io


os.chdir(os.path.dirname(os.getcwd()))

numTrainSpectra, numVariationsTrain, numTest = 100, 30000, 10
specLength = 512
db: 'io.Database' = io.get_database(maxSpectra=numTrainSpectra + numTest)
db.preprocessSpectra()
db.reduceSpecsToNWavenumbers(specLength)

spectra = db.getSpectra()[:, 1:]
trainSpectra, testSpectra = spectra[:, :numTrainSpectra], spectra[:, numTrainSpectra:]
trainSpectra = np.tile(trainSpectra, (1, numVariationsTrain))
numTotalSpecs = trainSpectra.shape[1]


trainSpectra = trainSpectra.transpose()
testSpectra = testSpectra.transpose()

trainSpectra = tf.cast(trainSpectra, tf.float32)
testSpectra = tf.cast(testSpectra, tf.float32)

trainSpectra = tf.clip_by_value(trainSpectra, clip_value_min=0.0, clip_value_max=1.0)
testSpectra = tf.clip_by_value(testSpectra, clip_value_min=0.0, clip_value_max=1.0)

noiseLevel = 0.2
noisyTrainSpectra = trainSpectra + noiseLevel * tf.random.normal(shape=trainSpectra.shape, seed=42)
noisyTestSpectra = testSpectra + noiseLevel * tf.random.normal(shape=testSpectra.shape, seed=42)

noisyTrainSpectra = tf.clip_by_value(noisyTrainSpectra, clip_value_min=0.0, clip_value_max=1.0)
noisyTestSpectra = tf.clip_by_value(noisyTestSpectra, clip_value_min=0.0, clip_value_max=1.0)


class Denoiser(Model):
    def __init__(self):
        super(Denoiser, self).__init__()
        self.encoder = tf.keras.Sequential([
            # Dense(256, activation="relu"),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            # Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(8, activation="relu")])

        self.decoder = tf.keras.Sequential([
            Dense(16, activation="relu"),
            # Dense(32, activation="relu"),
            Dense(64, activation="relu"),
            Dense(128, activation="relu"),
            # Dense(256, activation="relu"),
            Dense(specLength, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


denoiser = Denoiser()
denoiser.compile(optimizer='adam', loss='mae')
history = denoiser.fit(noisyTrainSpectra, trainSpectra,
          epochs=5,
          validation_data=(noisyTestSpectra, testSpectra),
          shuffle=True)


reconstructedSpecs = denoiser.call(noisyTestSpectra)

n = 3
fig: plt.Figure = plt.figure(figsize=(20, 7))
for i in range(n):
    orig = testSpectra[i]
    noisy = noisyTestSpectra[i]
    reconst = reconstructedSpecs[i]
    savgol = savgol_filter(noisy, window_length=21, polyorder=5)

    corrNN = np.round(np.corrcoef(orig, reconst)[0, 1] * 100)
    corrSavGol = np.round(np.corrcoef(orig, savgol)[0, 1] * 100)
    print(f'Spec {i+1}: Corr NN: {corrNN} %, corr savgol: {corrSavGol} %')

    ax = fig.add_subplot(2, n, i+1)
    ax.set_title("noisy")
    ax.plot(noisy)

    bx = fig.add_subplot(2, n, i+n+1)
    bx.plot(orig, label='original')
    bx.plot(reconst + 0.4, label=f'neuronal net\n{corrNN} % Correlation')
    bx.plot(savgol + 0.8, label=f'savgol filter\n{corrSavGol} % Correlation')
    bx.legend()

fig.tight_layout()
fig.show()
