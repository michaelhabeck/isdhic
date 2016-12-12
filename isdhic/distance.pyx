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

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def calc_distances(double [::1] coords,
                   int [::1] indices1,
                   int [::1] indices2):

    cdef Py_ssize_t i, j, n
    cdef int N = len(indices1)
    cdef double d, x
    cdef double [::1] distances = numpy.zeros(N)

    for n in range(N):

        i = indices1[n]
        j = indices2[n]

        x = coords[3*i+0] - coords[3*j+0]
        d = x * x

        x = coords[3*i+1] - coords[3*j+1]
        d+= x * x

        x = coords[3*i+2] - coords[3*j+2]
        d+= x * x

        distances[n] = sqrt(d)
        
    return numpy.array(distances)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def calc_data(double [::1] coords,
              int [::1] first_index,
              int [::1] second_index,
              double [::1] mock):

    cdef Py_ssize_t i, j, n
    cdef int N = len(first_index)
    cdef double d, x

    for n in range(N):

        i = first_index[n]
        j = second_index[n]

        x = coords[3*i+0] - coords[3*j+0]
        d = x * x

        x = coords[3*i+1] - coords[3*j+1]
        d+= x * x

        x = coords[3*i+2] - coords[3*j+2]
        d+= x * x

        mock[n] = sqrt(d)

@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def update_forces(double [::1] coords,
                  int [::1] first_index,
                  int [::1] second_index,
                  double [::1] mock,
                  double [::1] gradient,
                  double [::1] forces):

    cdef Py_ssize_t i, j, n
    cdef int N = len(first_index)
    cdef double c, d, x

    for n in range(N):

        i = first_index[n]
        j = second_index[n]
        c = gradient[n] / mock[n]

        x = coords[3*i+0] - coords[3*j+0]
        forces[3*i+0] += c * x
        forces[3*j+0] -= c * x
        
        x = coords[3*i+1] - coords[3*j+1]
        forces[3*i+1] += c * x
        forces[3*j+1] -= c * x
        
        x = coords[3*i+2] - coords[3*j+2]
        forces[3*i+2] += c * x
        forces[3*j+2] -= c * x
        

