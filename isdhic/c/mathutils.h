#ifndef MATHUTILS_H
#define MATHUTILS_H

#include "isdhic.h"

typedef double vector_double[3];

#define vector vector_double

#define array vector

typedef double vector6[6];

double map_angle(double angle);
double angle(double sine, double cosine);

PyObject *PyArray_CopyFromDimsAndData(int n_dimensions, int *dimensions, 
				      int type_num, char *data);

double * PyArray_DOUBLE_as_double(PyArrayObject *a);         

void   vector_print(vector x);
void   vector_scale(vector, vector, double);
double vector_dot(vector a, vector b);
double vector_norm(vector);
void   vector_add(vector dest, vector a, vector b);
void   vector_iadd(vector dest, vector a);
void   vector_imul(vector x, double c);
void   vector_sub(vector dest, vector a, vector b);
void   vector_normalize(vector v);
void   vector_set(vector, double);
void   vector_average(vector dest, vector *v, int n);
int    vector_less(vector a, vector b);
int    vector_greater(vector a, vector b);
void   vector_cross(vector dest, vector a, vector b);
double vector_spat(vector a, vector b, vector c);
double vector_dihedral(vector, vector, vector, vector);
void   vector_copy(vector, vector);

#endif
