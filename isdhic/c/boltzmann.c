#define NO_IMPORT_ARRAY

#include "isdhic.h"

double boltzmann_energy(PyBoltzmannPriorObject *self, PyUniverseObject* universe) {

  if (!self->enabled) return 0.;

  PyForceFieldObject *forcefield = (PyForceFieldObject*) self->forcefield;
  double E = forcefield->energy((PyObject*) forcefield, (PyObject*) universe);

  return self->beta * E;
}  

int boltzmann_update_gradient(PyBoltzmannPriorObject *self, PyUniverseObject *universe, double *E) {

  PyForceFieldObject *forcefield = (PyForceFieldObject*) self->forcefield;
  double k = forcefield->K;

  forcefield->K *= self->beta;
  forcefield->gradient((PyObject*) forcefield, (PyObject*) universe, E);
  forcefield->K = k;

  return 0;
}

PyObject* PyBoltzmannPrior_Energy(PyBoltzmannPriorObject *self, PyObject *args) {

  PyUniverseObject *universe;

  if (!PyArg_ParseTuple(args, "O!", &PyUniverse_Type, &universe)) {
    RAISE(PyExc_TypeError, "universe is expected", NULL);
  }
  if (!self->forcefield) {
    RAISE(PyExc_StandardError, "no force field set", NULL);
  }
  return Py_BuildValue("d", boltzmann_energy(self, universe));
}

PyObject* PyBoltzmannPrior_UpdateGradient(PyBoltzmannPriorObject *self, PyObject *args) {

  PyUniverseObject *universe;
  int calculate_energy = 0;
  double E;

  if (!PyArg_ParseTuple(args, "O!|i", &PyUniverse_Type, &universe, &calculate_energy)) {
    RAISE(PyExc_TypeError, "universe and integer expected.", NULL);
  }
  if (!self->forcefield) {
    RAISE(PyExc_StandardError, "no force field set.", NULL);
  }
  if (calculate_energy) {
    boltzmann_update_gradient(self, universe, &E);
    return Py_BuildValue("d", E);
  }
  else {
    boltzmann_update_gradient(self, universe, NULL);
    RETURN_PY_NONE;
  }
}

static PyMethodDef prior_methods[] = {
  {"energy", (PyCFunction) PyBoltzmannPrior_Energy, 1},
  {"update_gradient", (PyCFunction) PyBoltzmannPrior_UpdateGradient, 1},
  {NULL, NULL }
};

PyObject * boltzmann_getattr(PyBoltzmannPriorObject *self, char *name) {

  if (!strcmp(name, "forcefield")) {
    if (self->forcefield) {
      Py_INCREF(self->forcefield);
      return (PyObject*) self->forcefield;
    }
    RETURN_PY_NONE;
  }
  else if (!strcmp(name, "beta")) {
    return Py_BuildValue("d", self->beta);
  }
  else if (!strcmp(name, "enabled")) {
    return Py_BuildValue("i", self->enabled);
  }
  else {
    return Py_FindMethod(prior_methods, (PyObject *)self, name);
  }
}

int boltzmann_setattr(PyBoltzmannPriorObject *self, char *name, PyObject *op) {
  
  if (!strcmp(name, "forcefield")) {    
    if (self->forcefield) {
      Py_DECREF(self->forcefield);
    }
    if (op == Py_None) {
      self->forcefield = NULL;
    }
    else {
      Py_INCREF(op);
      self->forcefield = (PyForceFieldObject*) op;
    }
  }      
  else if (!strcmp(name, "beta")) {
    self->beta = (double) PyFloat_AsDouble(op);
  }
  else if (!strcmp(name, "enabled")) {
    self->enabled = (int) PyInt_AsLong(op);
  }
  else {
    PyErr_SetString(PyExc_AttributeError, "Attribute does not exist or cannot be set");
    return -1;
  }
  return 0;
}

void boltzmann_dealloc(PyBoltzmannPriorObject *self) {

  if (self->forcefield) {
    Py_DECREF(self->forcefield);
  }
  PyObject_Del(self);
}

static char BoltzmannPriortype__doc__[] = ""; 

PyTypeObject PyBoltzmannPrior_Type = { 
	PyObject_HEAD_INIT(0)
	0,			       /*ob_size*/
	"boltzmannprior",	       /*tp_name*/
	sizeof(PyBoltzmannPriorObject),/*tp_basicsize*/
	0,			       /*tp_itemsize*/

	(destructor)boltzmann_dealloc, /*tp_dealloc*/
	(printfunc)NULL,	       /*tp_print*/
       	(getattrfunc)boltzmann_getattr,/*tp_getattr*/
	(setattrfunc)boltzmann_setattr,/*tp_setattr*/
	(cmpfunc)NULL,	               /*tp_compare*/
	(reprfunc)NULL,	               /*tp_repr*/

	NULL,		               /*tp_as_number*/
	NULL,	                       /*tp_as_sequence*/
	NULL,		 	       /*tp_as_mapping*/

	(hashfunc)0,		       /*tp_hash*/
	(ternaryfunc)0, 	       /*tp_call*/
	(reprfunc)0,		       /*tp_str*/
		
	0L,0L,0L,0L,
	BoltzmannPriortype__doc__               /* Documentation string */
};

PyObject * boltzmann_init(PyBoltzmannPriorObject *self) {

  self->forcefield = NULL;
  self->enabled    = 1;
  self->beta       = 1.;

  return (PyObject*) self;
}

PyObject * PyBoltzmannPrior_New(PyObject *self, PyObject *args) {

  PyBoltzmannPriorObject *ob;

  ob = PyObject_NEW(PyBoltzmannPriorObject, &PyBoltzmannPrior_Type);

  return boltzmann_init(ob);
}

