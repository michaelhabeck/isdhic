import numpy
cimport numpy
cimport cython

DTYPE_FLOAT = numpy.float
ctypedef numpy.float_t DTYPE_FLOAT_t

DTYPE_INT = numpy.int
ctypedef numpy.int_t DTYPE_INT_t

DTYPE_DOUBLE = numpy.double
ctypedef numpy.double_t DTYPE_DOUBLE_t

DTYPE_LONG = numpy.long
ctypedef numpy.long_t DTYPE_LONG_t

cdef extern from "math.h":
    double sqrt(double)
    double log(double)
    

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def log_prob(double [::1] data,
             double [::1] mock,
             double [::1] lower,
             double [::1] upper):

    cdef Py_ssize_t i
    cdef int n = len(data)
    cdef double x, logp = 0.

    for i in range(n):

        if mock[i] < lower[i]:
            x = mock[i] - lower[i]
            logp -= x * x

        elif mock[i] > upper[i]:
            x = mock[i] - upper[i]
            logp -= x * x

    return logp

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def logZ(double [::1] lower,
         double [::1] upper,
         double precision):

    cdef Py_ssize_t i
    cdef int n = len(lower)
    cdef double logZ = 0., x = sqrt(2 * numpy.pi / precision)

    for i in range(n):

        logZ += log(x + upper[i] - lower[i])

    return logZ

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def update_derivatives(double [::1] mock,
                       double [::1] grad,
                       double [::1] lower,
                       double [::1] upper,
                       double precision):

    cdef Py_ssize_t i
    cdef int n = len(mock)

    for i in range(n):

        if mock[i] < lower[i]:
            grad[i] = precision * (lower[i] - mock[i])
        elif mock[i] > upper[i]:
            grad[i] = precision * (upper[i] - mock[i])
        else:
            grad[i] = 0.
