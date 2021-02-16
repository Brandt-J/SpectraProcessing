import numpy as np

spectra = np.load(r"PMMA/spectra/Spectra Series 0.npy")
for i in range(spectra.shape[1]-1):
    spec = spectra[:, [0, i+1]]

    if np.mean(spec[:, 1]) < np.median(spec[:, 1]):  # make sure it's in "absorption" mode
        spec[:, 1] *= -1

    np.savetxt(f'pmma spec {i +1 }.csv', spec, delimiter=',')
