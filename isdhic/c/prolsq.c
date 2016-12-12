#define NO_IMPORT_ARRAY

#include "isdhic.h"

static double prolsq_energy(PyForceFieldObject *self, double d, double d0, double k) {
  
  double a = d0 - d;

  if (a < 0.) {
    return 0.;
  }
  else {

    a *= a;
    a *= a;

    return k * a / 2.;
  }
}

static double prolsq_gradient(PyForceFieldObject *_self, double d, double d0, double k, double *E) {

  double c=0.;

  if (d < d0) {
	
    c = (d0 - d);
    c *= k * c * c;

    *E += (d0 - d) * c;

    c *= - 2 / d;
  }
  return c;
}

static double energy(PyProlsqObject *self, PyUniverseObject *universe, int *types) {
  return forcefield_energy((PyForceFieldObject*)self, universe, types);
}

static int gradient(PyProlsqObject *self, PyUniverseObject *universe, int *types, double *E_ptr) {
  return forcefield_gradient((PyForceFieldObject*)self, universe, types, E_ptr);
}

static PyObject * py_energy(PyProlsqObject *self, PyObject *args) {
  
  PyUniverseObject *universe;
  PyArrayObject *types;

  if (!PyArg_ParseTuple(args, "O!O!", &PyUniverse_Type, &universe, &PyArray_Type, &types)) {
    RAISE(PyExc_StandardError, "universe and numpy array storing atom types expected.", NULL);
  }
  return Py_BuildValue("d", forcefield_energy((PyForceFieldObject*)self, universe, (int*) types->data));
}

static PyObject * py_update_gradient(PyProlsqObject *self, PyObject *args) {
  
  int calculate_energy = 0;
  PyUniverseObject *universe;
  PyArrayObject *types;
  double E;

  if (!PyArg_ParseTuple(args, "O!O!i", &PyUniverse_Type, &universe, &PyArray_Type, &types, &calculate_energy)) {
    RAISE(PyExc_TypeError, "universe and numpy array storing atom types expected.", NULL);
  }
  if (calculate_energy) {
    gradient(self, universe, (int*) types->data, &E);
    return Py_BuildValue("d", E);
  }
  else {
    gradient(self, universe, (int*) types->data, NULL);
    RETURN_PY_NONE;
  }
}

static PyMethodDef methods[] = {
  {"update_gradient", (PyCFunction) py_update_gradient, 1},
  {"energy", (PyCFunction) py_energy, 1},
  {NULL, NULL }
};

static void dealloc(PyProlsqObject *self) {

  forcefield_dealloc((PyForceFieldObject*)self);  

  self->k = NULL;
  self->d = NULL;
  self->nblist = NULL; 

  PyObject_Del(self);
}

static PyObject *getattr(PyProlsqObject *self, char *name) {

  PyObject *attr = forcefield_getattr((PyForceFieldObject*)self, name);

  if (!attr) {
    return Py_FindMethod(methods, (PyObject *)self, name);
  }
  else {
    return attr;
  }
}

static int setattr(PyProlsqObject *self, char *name, PyObject *op) {

  if (!forcefield_setattr((PyForceFieldObject*)self, name, op)) {
    RAISE(PyExc_AttributeError, "Attribute does not exist or cannot be set", -1);
  }  
  return 0;
}

static char __doc__[] = "prolsq forcefield"; 

PyTypeObject PyProlsq_Type = { 
  PyObject_HEAD_INIT(0)
  0,			       /*ob_size*/
  "prolsq",	               /*tp_name*/
  sizeof(PyProlsqObject),      /*tp_basicsize*/
  0,			       /*tp_itemsize*/
  
  (destructor)dealloc,         /*tp_dealloc*/
  (printfunc)NULL,	       /*tp_print*/
  (getattrfunc)getattr,        /*tp_getattr*/
  (setattrfunc)setattr,        /*tp_setattr*/
  (cmpfunc)NULL,               /*tp_compare*/
  (reprfunc)NULL,	       /*tp_repr*/
  
  NULL,		               /*tp_as_number*/
  NULL,	                       /*tp_as_sequence*/
  NULL,		 	       /*tp_as_mapping*/

  (hashfunc)0,		       /*tp_hash*/
  (ternaryfunc)0,	       /*tp_call*/
  (reprfunc)0,		       /*tp_str*/
  
  0L,0L,0L,0L,
  __doc__                        /* Documentation string */
};

PyObject * PyProlsq_New(PyObject *self, PyObject *args) {

  PyProlsqObject *ob;

  if (!PyArg_ParseTuple(args, "")) return NULL;

  ob = PyObject_NEW(PyProlsqObject, &PyProlsq_Type);

  forcefield_init((PyForceFieldObject*)ob);

  /* set energy and gradient function pointers */

  ob->energy   = (forcefield_energyfunc) energy;
  ob->gradient = (forcefield_gradientfunc) gradient;

  ob->f      = (forcefield_energyterm) prolsq_energy;
  ob->grad_f = (forcefield_gradenergyterm) prolsq_gradient;

  return (PyObject*) ob;
}
