#ifndef __ISD_H__
#define __ISD_H__

#ifdef __cplusplus
 extern "C" {
#endif

#include <Python.h>
#include <numpy/arrayobject.h>
#include "mathutils.h"
#include "nblist.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define RAISE(a,b,c) {PyErr_SetString(a, b); return c;}
#define HERE {printf("%s: %d\n",__FILE__,__LINE__);}
#define MALLOC(n, t) ((t*) malloc((n) * sizeof(t)))
#define REALLOC(p, n, t) ((t*) realloc((void*) (p), (n) * sizeof(t)))
#define MEMCPY(a,b,n,t) memcpy((void*) (a), (void*) (b), (n) * sizeof(t))
#define RETURN_PY_NONE {Py_INCREF(Py_None);return Py_None;}
#define INC_AND_RETURN(a) {Py_INCREF((PyObject*) (a)); return (PyObject*) (a);}

/* TODO: change to SET_PYOBJECT(dest, src); that's more standard */

#define SET_PYOBJECT(op, dest) {if (dest) Py_DECREF(dest); if (op) Py_INCREF(op); dest = (op);}

typedef double (*forcefield_energyfunc) (PyObject*, PyObject*);
typedef int (*forcefield_gradientfunc) (PyObject*, PyObject*, double *);

typedef double (*forcefield_energyterm) (PyObject*, double, double, double);
typedef double (*forcefield_gradenergyterm) (PyObject*, double, double, double, double *);

#define PyPriorObject_HEAD \
        PyObject_HEAD \
        int id, enabled;

#define PyForceFieldObject_HEAD \
        PyObject_HEAD \
        double K;                /* overall force constant */ \
	int enabled;             /* switch for turning ON / OFF the interaction */\ 
	int n_types;             /* number of atom-types supported by the force-field */\
        double *k;               /* matrix of pairwise force-constants */\
        double *d;               /* matrix of pairwise sums of vwd-radii */\
        PyNBListObject *nblist;  /* neighbor list */\
        forcefield_energyterm f; \
        forcefield_gradenergyterm grad_f; \
	forcefield_energyfunc   energy; \
        forcefield_gradientfunc gradient;

#define PyDatumObject_HEAD \
        PyObject_HEAD \
	int serial; \
        double value;

typedef struct {
  PyPriorObject_HEAD
} PyPriorObject;

// Universe: container for all particles

typedef struct _PyUniverseObject {

  PyObject_HEAD

  int n_particles;

  // Cartesian coordinates and gradient

  PyArrayObject *coords;
  PyArrayObject *forces;

} PyUniverseObject;

PyObject * PyUniverse_New(PyObject *self, PyObject *args);

extern PyTypeObject PyUniverse_Type;

   /* Force field */

typedef struct {
  PyForceFieldObject_HEAD
} PyForceFieldObject;

typedef struct {
  PyForceFieldObject_HEAD
} PyProlsqObject;

typedef struct {
  PyForceFieldObject_HEAD

  /* parameters for ramping the Lennard-Jones potential */

  double r_max;            /* distance cutoff */
  double r_lin;            /* distance beyond which linear approximation starts */
  double r_sw;             /* factor < 1 below which linear approximation starts */

} PyRosettaObject;

extern PyTypeObject PyRosetta_Type;
extern PyTypeObject PyProlsq_Type;

// general force field routines

int        forcefield_set_k(PyForceFieldObject *self, PyObject *op);
int        forcefield_set_d(PyForceFieldObject *self, PyObject *op);
void       forcefield_init(PyForceFieldObject *self);
void       forcefield_dealloc(PyForceFieldObject *self);
int        forcefield_setattr(PyForceFieldObject *self, char *name, PyObject *op);
PyObject * forcefield_getattr(PyForceFieldObject *self, char *name);
double     forcefield_energy(PyForceFieldObject *self, double *coords, int *types, int n_particles);
double     forcefield_gradient(PyForceFieldObject *self, double *coords, double *forces, int *types, int n_particles, double *E);

PyObject * PyProlsq_New(PyObject *self, PyObject *args);
PyObject * PyRosetta_New(PyObject *self, PyObject *args);

#define PI 3.141592653589793115997963468544
#define TWO_PI 6.283185307179586231995926937088
#define MAX_EXP 709.
#define MIN_EXP -709.
#define INDEX(i, j, k, n) (n * (n * i + j) + k)
#define BOX_MARGIN 1e-5
#define SET_PYARRAY(dest, op) {if (dest) Py_DECREF(dest); if (op == Py_None) dest = NULL; else {Py_INCREF(op); dest = (PyArrayObject*) (op);}}
#define CREATE_ARRAY(x, t, n, initialize) {int i;if (x) free(x); x = (t*) malloc((n) * sizeof(t)); if (initialize) for (i=0; i<n; i++) x[i] = 0.;}

#include "boltzmann.h"

#ifdef __cplusplus
 }
#endif

#endif
