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
    double log(double)
    double exp(double)
    

@cython.boundscheck(True)
@cython.wraparound(False)
def log_prob(double [::1] data,
             double [::1] mock,
             double steepness):

    cdef Py_ssize_t i
    cdef int n = len(data)
    cdef double x, lgp = 0.
    
    for i in range(n):

        x = mock[i] - data[i]
        if x > 0.: lgp -= x

    return steepness * lgp

@cython.boundscheck(True)
@cython.wraparound(False)
def update_derivatives(double [::1] data,
                       double [::1] mock,
                       double [::1] grad,
                       double steepness):

    cdef Py_ssize_t i
    cdef int n = len(data)
    cdef double x
    
    for i in range(n):

        x = mock[i] - data[i]

        if x > 0.:
            grad[i] = -steepness
        else:
            grad[i] = 0.

