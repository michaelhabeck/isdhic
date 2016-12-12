#ifndef __BOLTZMANN_H__
#define __BOLTZMANN_H__

typedef struct {
  PyPriorObject_HEAD
  
  double beta;

  PyForceFieldObject *forcefield;

} PyBoltzmannPriorObject;

typedef struct {
  PyPriorObject_HEAD

  /* TODO: have a BoltzmannPrior_HEAD? */
  
  double beta;

  PyForceFieldObject *forcefield;

  double E_min, q;

} PyTsallisPriorObject;

extern PyTypeObject PyTsallisPrior_Type;
extern PyTypeObject PyBoltzmannPrior_Type;

void       boltzmann_dealloc(PyBoltzmannPriorObject *self);
int        boltzmann_setattr(PyBoltzmannPriorObject *self, char *name, PyObject *op);
PyObject * boltzmann_getattr(PyBoltzmannPriorObject *self, char *name);
PyObject * boltzmann_init(PyBoltzmannPriorObject *self);
double     boltzmann_energy(PyBoltzmannPriorObject *self, PyUniverseObject* universe);
int        boltzmann_update_gradient(PyBoltzmannPriorObject *self, 
				     PyUniverseObject *universe, double *E);
int        boltzmann_error(void);

PyObject * PyBoltzmannPrior_New(PyObject *self, PyObject *args);
PyObject * PyTsallisPrior_New(PyObject *self, PyObject *args);

#endif
