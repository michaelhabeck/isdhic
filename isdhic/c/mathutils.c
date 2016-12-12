#define NO_IMPORT_ARRAY

#include "isdhic.h"

void vector_print(vector x) {
  int i;
  
  printf("[ ");
  for (i=0;i<3;i++) printf("%e ",x[i]);
  printf("]\n");
}

void vector_scale(vector b, vector a, double scale) {

  int i;

  for (i = 0; i < 3; i++) b[i] = scale * a[i];
}

double vector_dot(vector a, vector b) {

  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

void vector_imul(vector x, double c) {
  
  int i;

  for (i = 0; i < 3; i++) x[i] *= c;
}

void vector_add(vector dest, vector a, vector b) {
  
  int i;

  for (i = 0; i < 3; i++)
    dest[i] = a[i] + b[i];
}

void vector_iadd(vector dest, vector a) {
  
  int i;

  for (i = 0; i < 3; i++) dest[i] += a[i];
}

void vector_sub(vector dest, vector a, vector b) {
  
  int i;

  for (i = 0; i < 3; i++)
    dest[i] = a[i] - b[i];
}

void vector_normalize(vector v) {
  int i;
  double sum;

  sum = 0.;

  for (i = 0; i < 3; i++) sum += v[i] * v[i];

  sum = sqrt(sum);

  if (sum != 0.0)
    for (i = 0; i < 3; i++) v[i] /= sum;
}

double vector_norm(vector v) {
  int i;
  double sum = 0.;

  for (i = 0; i < 3; i++) sum += v[i] * v[i];

  return sqrt(sum);
}

void vector_copy(vector v, vector w) {
  int i;

  for (i = 0; i < 3; i++) v[i] = w[i];
}

void vector_cross(vector c, vector a, vector b) {

  /* Cross product between two 3D-vectors. */

  c[0] = a[1]*b[2] - a[2]*b[1];
  c[1] = a[2]*b[0] - a[0]*b[2];
  c[2] = a[0]*b[1] - a[1]*b[0]; 
}

double vector_spat(vector a, vector b, vector c) {
  vector x;
  vector_cross(x, b, c);
  return vector_dot(a, x);
}

double vector_dihedral(vector x1, vector x2, vector x3, vector x4) {

  /* Returns the dihedral angle (in radian!) defined by
     the four vectors a, b, c, d. */

  double sinus, cosinus;

  vector u, r, s;
  vector a, b, c;

  vector_sub(u, x1, x2);
  vector_sub(r, x2, x3);
  vector_sub(s, x4, x3);

  vector_cross(a, u, r);
  vector_normalize(a);
    
  vector_cross(b, s, r);
  vector_normalize(b);

  vector_cross(c, b, a);
  vector_normalize(r);

  cosinus = vector_dot(a, b);
  sinus = vector_dot(c, r);
    
  return angle(sinus, cosinus);
}

double angle(double sine, double cosine) {

  /* Calculate angle from its sine and cosine. */

  double angle;
  
  if (sine < -1.) sine = -1.;
  else if (sine > 1.) sine = 1.;

  angle = asin(fabs(sine));
  
  if (sine * cosine < 0.) angle = -angle;
        
  if (cosine < 0.) angle += PI;
        
  if (angle < 0.) angle += TWO_PI;
        
  return angle;

}

void vector_set(vector v, double val) {
  int i;

  for (i = 0; i < 3; i++) v[i] = val;
}

void vector_average(vector dest, vector *v, int n) {
  int i;

  vector_set(dest, 0.);

  for (i = 0; i < n; i++) 
    vector_add(dest, dest, v[i]);

  for (i = 0; i < 3; i++) dest[i] /= n;
}

void vector_weighted_average(vector dest, vector *v, double *w, int n) {
  int i;
  vector x;

  vector_set(dest, 0.);

  for (i = 0; i < n; i++) {
    vector_scale(x, v[i], w[i]);
    vector_add(dest, dest, x);
  }
}

int vector_less(vector a, vector b) {

  /* return 1 if a[i] < b[i] for all i */

  if (a[0] > b[0] || a[1] > b[1] || a[2] > b[2])
    return 0;
  else
    return 1;
}

int vector_greater(vector a, vector b) {

  /* return 1 if a[i] > b[i] for all i */

  if (a[0] < b[0] || a[1] < b[1] || a[2] < b[2])
    return 0;
  else
    return 1;
}

double * PyArray_DOUBLE_as_double(PyArrayObject *a) {

  /* ravels a into an C-double field. memory is allocated.
     good for rank-1 and rank-2 tensors. If error occurs, NULL
     is returned. */

  int i, j, s1, s2, n, m;
  double *d;

  if (a->nd < 1 || a->nd > 2)
    return NULL;

  n = a->dimensions[0];
  s1 = a->strides[0];

  if (a->nd == 1) {
    
    if (!(d = (double*) malloc(n * sizeof(double))))
      return NULL;


    for (i = 0; i < n; i++)
      d[i] = * (double*) (a->data + i * s1);

  } else {

    m = a->dimensions[1];

    if (!(d = (double*) malloc(n * m * sizeof(double))))
      return NULL;

    s2 = a->strides[1];

    for (i = 0; i < n; i++) for (j = 0; j < m; j++)
      d[i * m + j] = * (double*) (a->data + i * s1 + j * s2);

  }

  return d;
}
  
PyObject *PyArray_CopyFromDimsAndData(int n_dimensions, int *dimensions, 
				      int type_num, char *data) {
  /*
    This method is similar to PyArray_FromDimAndData. It creates a new
    PyArrayObject, but instead of referencing 'data', it returns a
    copy of it.
   */

  PyObject *a1, *a2;

  npy_intp dims[n_dimensions];

  int i;
  for (i = 0; i < n_dimensions; i++)
      dims[i] = dimensions[i];

  a1 = PyArray_SimpleNewFromData(n_dimensions, dims, type_num, data);
  a2 = PyArray_Copy((PyArrayObject*) a1);

  Py_DECREF(a1);

  return PyArray_Return((PyArrayObject*) a2);
}

