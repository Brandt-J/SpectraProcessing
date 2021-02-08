import numpy as np
cimport numpy as np
cimport cython


NP_INT32 = np.int32
ctypedef np.int32_t INT32_t

DTYPE = np.float
ctypedef np.float_t DTYPE_t

@cython.boundscheck(False)           # assume: no index larger than N-1
@cython.wraparound(False)            # assume: no neg. index
def getCorrelationCoefficients(np.float_t[:] pattern, np.float_t[:, :] spectra):
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
def sfec(np.float_t[:] intens1, np.float_t[:] intens2):
    """
    Calculates Squared First-Difference Euclidean Cosine, according to equation 6 in DOI: 10.1021/acsomega.0c05041
    :param intens1: First set of intensities of length N
    :param intens2: Second set of intensities of length N
    :return:
    """
    assert len(intens1) == len(intens2)

    cdef double sum1 = 0.0
    cdef double sum2 = 0.0
    cdef double sum3 = 0.0
    cdef double corr = np.nan
    cdef double delta1, delta2, nextVal
    cdef int n = len(intens1)

    for i in range(n - 1):
        delta1 = intens1[i + 1] - intens1[i]
        delta2 = intens2[i + 1] - intens2[i]
        sum1 += delta1 * delta2
        sum2 += delta1 ** 2
        sum3 += delta2 ** 2

    if sum2*sum3 != 0:
        corr = sum1 ** 2 / (sum2 * sum3)

    return corr


@cython.boundscheck(False)           # assume: no index larger than N-1
@cython.wraparound(False)            # assume: no neg. index
@cython.cdivision(True)              # assume: No division by 0
cdef double getCorrelation(np.float_t[:] pattern, np.float_t[:] curSpec, double patternMean, int numPoints):
    cdef int i
    cdef double corr, sum1, sum2, sum3, diffPattern, diffSample, quotient
    cdef double specMean = getMean(curSpec, numPoints)
    sum1 = 0
    sum2 = 0
    sum3 = 0
    for i in range(numPoints):
        diffPattern = pattern[i] - patternMean
        diffSample = curSpec[i] - specMean
        sum1 += diffSample * diffPattern
        sum2 += diffSample ** 2
        sum3 += diffPattern ** 2

    quotient = (sum2**0.5 * sum3**0.5)
    if quotient != 0:
        corr = sum1 / quotient
    else:
        corr = 0

    # if corr >= 0.5:   # This is, what the Hufnagl et al do, but it makes results worse in my case...
    #     corr *= np.trapz(curSpec)  # multiply with peak area
    # else:
    #     corr = 0  # set to zero

    return corr


@cython.boundscheck(False)           # assume: no index larger than N-1
@cython.wraparound(False)            # assume: no neg. index
@cython.cdivision(True)              # assume: No division by 0
cdef double getMean(np.float_t[:] intensities, int numPoints):
    cdef double sum = 0
    cdef int i = 0
    cdef double mean
    for i in range(numPoints):
        sum += intensities[i]
    mean = sum / numPoints
    return mean
