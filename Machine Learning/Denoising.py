import tensorflow as tf
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Input
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import gaussian, savgol_filter

import importData as io


os.chdir(os.path.dirname(os.getcwd()))

numTrainSpectra, numVariationsTrain, numTest = 20, 1000, 10
specLength = 512
db: 'io.Database' = io.get_database(maxSpectra=numTrainSpectra + numTest)
db.preprocessSpectra()
db.reduceSpecsToNWavenumbers(specLength)

spectra = db.getSpectra()[:, 1:]
trainSpectra, testSpectra = spectra[:, :numTrainSpectra], spectra[:, numTrainSpectra:]
trainSpectra = np.tile(trainSpectra, (1, numVariationsTrain))
numTotalSpecs = trainSpectra.shape[1]
kernelSize = (3, 3)
print('Kernelsize:', kernelSize)
numRows = 4

x_train: np.ndarray = np.zeros((numTotalSpecs, numRows, specLength))
for i in range(numTotalSpecs):
    x_train[i, :, :] = np.tile(trainSpectra[:, i], (numRows, 1))

x_test: np.ndarray = np.zeros((numTest, numRows, specLength))
for i in range(numTest):
    x_test[i, :, :] = np.tile(testSpectra[:, i], (numRows, 1))

x_train = x_train[:, :, :, tf.newaxis]  # TODO: WHY???
x_test = x_test[:, :, :, tf.newaxis]

noise_factor = 0.2
x_train_noisy = x_train - noise_factor/2 + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test - noise_factor/2 + noise_factor * tf.random.normal(shape=x_test.shape)
x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)


class NoiseReducer(tf.keras.Model):
    def __init__(self):
        super(NoiseReducer, self).__init__()

        self.encoder = tf.keras.Sequential([
            Input(shape=(numRows, specLength, 1)),
            Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
            Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

        self.decoder = tf.keras.Sequential([
            Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
            Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = NoiseReducer()
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(x_train_noisy,
                x_train,
                epochs=2,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))

encoded_imgs = autoencoder.encoder(x_test_noisy).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs)

n = 5
fig: plt.Figure = plt.figure(figsize=(20, 7))
for i in range(n):
    orig = np.array(tf.squeeze(x_test[i]))[0, :]
    noisy = np.array(tf.squeeze(x_test_noisy[i]))[0, :]
    reconst = np.array(tf.squeeze(decoded_imgs[i]))[0, :]
    savgol = savgol_filter(noisy, window_length=21, polyorder=5)

    corrNN = np.round(np.corrcoef(orig, reconst)[0, 1] * 100)
    corrSavGol = np.round(np.corrcoef(orig, savgol)[0, 1] * 100)
    print(f'Spec {i+1}: Corr NN: {corrNN} %, corr savgol: {corrSavGol} %')

    ax = fig.add_subplot(2, n, i+1)
    ax.set_title("noisy")
    ax.plot(noisy)

    bx = fig.add_subplot(2, n, i+n+1)
    bx.plot(orig, label='original')
    bx.plot(reconst + 0.2, label=f'neuronal net\n{corrNN} % Correlation')
    bx.plot(savgol + 0.4, label=f'savgol filter\n{corrSavGol} % Correlation')
    bx.legend()

fig.tight_layout()
fig.show()
