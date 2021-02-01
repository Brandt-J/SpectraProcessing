import numpy as np
cimport numpy as np
cimport cython


NP_INT32 = np.int32
ctypedef np.int32_t INT32_t

DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)           # assume: no index larger than N-1
@cython.wraparound(False)            # assume: no neg. index
def getCorrelationCoefficients(np.ndarray[np.float_t, ndim=1] pattern, np.ndarray[np.float_t, ndim=2] spectra):
    """
    Pattern: Array with M intensities
    Spectra: (MxN) Spectra, N spectra at M wavenumbers, each
    """
    cdef int i
    cdef int numSpectra = spectra.shape[1]
    cdef int numPoints = spectra.shape[0]
    cdef np.ndarray[np.float_t, ndim=1] corrs = np.zeros(numSpectra)
    cdef double patternMean = getMean(pattern, numPoints)

    if numPoints > 0:
        for i in range(numSpectra):
            corrs[i] = getCorrelation(pattern, spectra[:, i], patternMean, numPoints)

    return corrs


@cython.boundscheck(False)           # assume: no index larger than N-1
@cython.wraparound(False)            # assume: no neg. index
@cython.cdivision(True)              # assume: No division by 0
cdef double getCorrelation(np.ndarray[np.float_t, ndim=1] pattern, np.ndarray[np.float_t, ndim=1] curSpec, double patternMean, int numPoints):
    cdef int i
    cdef double corr, sum1, sum2, sum3, diffPattern, diffSample, quotient
    cdef double specMean = getMean(curSpec, numPoints)
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(numPoints):
        diffPattern = (pattern[i] - patternMean)
        diffSample = (curSpec[i] - specMean)
        sum1 += diffSample * diffPattern
        sum2 += diffSample ** 2
        sum3 += diffPattern ** 2

    quotient = (sum2**0.5 * sum3**0.5)
    if quotient != 0:
        corr = sum1 / quotient
    else:
        corr = 0

    # if corr >= 0.5:
    #     corr *= np.trapz(curSpec)  # multiply with peak area
    # else:
    #     corr = 0  # set to zero

    return corr


@cython.boundscheck(False)           # assume: no index larger than N-1
@cython.wraparound(False)            # assume: no neg. index
@cython.cdivision(True)              # assume: No division by 0
cdef double getMean(np.ndarray[np.float_t, ndim=1] intensities, int numPoints):
    cdef double sum = 0
    cdef int i = 0
    cdef double mean
    for i in range(numPoints):
        sum += intensities[i]
    mean = sum / numPoints
    return mean
