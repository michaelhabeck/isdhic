#define NO_IMPORT_ARRAY

#include "isdhic.h"

static double rosetta_energy(PyForceFieldObject *_self, double d, double d0, double k) {

  double a;

  PyRosettaObject *self = (PyRosettaObject*) _self;

  double r_max = self->r_max;
  double r_lin = self->r_lin;
  double r_sw  = self->r_sw;      

  if (d > r_max) {
    return 0.;
  }
  else if (d >= r_lin) {

    a = d0 / r_lin;
    a*= a;
    a*= a*a;	
    a*= (2-a) / (r_max-r_lin);

    return k * a * (d - r_max);
  }
  else if (r_sw * d0 < d) {

    a = d0 / d;
    a*= a * a;
    a*= a;

    return k * ((a - 2) * a);
  }
  else {

    a = 1 / r_sw;
    a*= a * a;
    a*= a;

    d *= -12 * (a - 1) * a / (r_sw * d0);

    return k * (d + (13 * a - 14) * a);
  }
}

static double rosetta_gradient(PyForceFieldObject *_self, double d, double d0, double k, double *E) {

  double a, dE=0.;

  PyRosettaObject *self = (PyRosettaObject*) _self;

  double r_max = self->r_max;
  double r_lin = self->r_lin;
  double r_sw  = self->r_sw;      

  if (d > r_max) {
    dE = 0.;
  }
  else if (d >= r_lin) {

    a = d0 / r_lin;
    a*= a;
    a*= a*a;	
    a*= (2-a) / (r_max-r_lin);
    
    *E += k * a * (d - r_lin);    
    dE  = k * a / d;
  }
  else if (r_sw * d0 < d) {

    a = d0 / d;
    a*= a * a;
    a*= a;
    
    *E += k * ((a - 2) * a);
    dE  = -12 * k * (a - 1) * a / d / d;
  }
  else {
    
    a = 1. / r_sw;
    a*= a * a;
    a*= a;
    
    dE  = -12 * (a - 1) * a / (r_sw * d0);    
    *E += k * (dE * d + (13 * a - 14) * a);
    dE *= k / d;
  }
  return dE;
}

static double energy(PyRosettaObject *self, 
		     double *coords, 
		     int *types, 
		     int n_particles) {

  return forcefield_energy((PyForceFieldObject*)self, coords, types, n_particles);
}

static int gradient(PyRosettaObject *self, 
		    double *coords, 
		    double *forces, 
		    int *types, 
		    int n_particles, 
		    double *E_ptr) {

  return forcefield_gradient((PyForceFieldObject*)self, coords, forces, types, n_particles, E_ptr);
}

static PyObject * py_energy(PyRosettaObject *self, PyObject *args) {
  
  PyArrayObject *coords, *types;

  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &coords, &PyArray_Type, &types)) {
    RAISE(PyExc_StandardError, "numpy arrays storing coordinates and atom types expected.", NULL);
  }
  return Py_BuildValue("d", forcefield_energy((PyForceFieldObject*)self, 
					      (double*) coords->data, 
					      (int*) types->data,
					      types->dimensions[0]));
}

static PyObject * py_update_gradient(PyRosettaObject *self, PyObject *args) {
  
  int calculate_energy = 0;
  PyArrayObject *coords, *forces, *types;
  double E;

  if (!PyArg_ParseTuple(args, "O!O!O!i", 
			&PyArray_Type, &coords, 
			&PyArray_Type, &forces, 
			&PyArray_Type, &types, 
			&calculate_energy)) {
    RAISE(PyExc_TypeError, "numpy arrays storing coordinates, forces and atom types expected.", NULL);
  }
  if (calculate_energy) {
    gradient(self, 
	     (double*) coords->data, 
	     (double*) forces->data, 
	     (int*) types->data, 
	     types->dimensions[0],
	     &E);
    return Py_BuildValue("d", E);
  }
  else {
    gradient(self, 
	     (double*) coords->data, 
	     (double*) forces->data, 
	     (int*) types->data, 
	     types->dimensions[0],
	     NULL);
    RETURN_PY_NONE;
  }
}

static PyMethodDef methods[] = {
  {"update_gradient", (PyCFunction) py_update_gradient, 1},
  {"energy", (PyCFunction) py_energy, 1},
  {NULL, NULL }
};

static void dealloc(PyRosettaObject *self) {
  forcefield_dealloc((PyForceFieldObject*) self);  
  PyObject_Del(self);
}

static PyObject *getattr(PyRosettaObject *self, char *name) {

  PyObject *attr=NULL;

  if (!strcmp(name, "r_max")) {
    return Py_BuildValue("d", self->r_max);
  }
  else if (!strcmp(name, "r_lin")) {
    return Py_BuildValue("d", self->r_lin);
  }
  else if (!strcmp(name, "r_sw")) {
    return Py_BuildValue("d", self->r_sw);
  }
  else {
    attr = forcefield_getattr((PyForceFieldObject*)self, name);
    if (!attr) {
      return Py_FindMethod(methods, (PyObject *)self, name);
    }
    else {
      return attr;
    }
  }
}

static int setattr(PyRosettaObject *self, char *name, PyObject *op) {

  if (!strcmp(name, "r_max")) {
    self->r_max = (double) PyFloat_AsDouble(op);
  }
  else if (!strcmp(name, "r_lin")) {
    self->r_lin = (double) PyFloat_AsDouble(op);
  }
  else if (!strcmp(name, "r_sw")) {
    self->r_sw = (double) PyFloat_AsDouble(op);
  }
  else if (!forcefield_setattr((PyForceFieldObject*)self, name, op)) {
    RAISE(PyExc_AttributeError, "Attribute does not exist or cannot be set", -1);
  }  
  return 0;
}

static char __doc__[] = "rosetta forcefield"; 

PyTypeObject PyRosetta_Type = { 
  PyObject_HEAD_INIT(0)
  0,			       /*ob_size*/
  "rosetta",	               /*tp_name*/
  sizeof(PyRosettaObject),     /*tp_basicsize*/
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

PyObject * PyRosetta_New(PyObject *self, PyObject *args) {

  PyRosettaObject *ob;

  if (!PyArg_ParseTuple(args, "")) return NULL;

  ob = PyObject_NEW(PyRosettaObject, &PyRosetta_Type);

  forcefield_init((PyForceFieldObject*)ob);

  ob->r_max = 5.5;
  ob->r_lin = 5.0;
  ob->r_sw  = 0.6;

  /* set energy and gradient function pointers */

  ob->energy   = (forcefield_energyfunc) energy;
  ob->gradient = (forcefield_gradientfunc) gradient;

  ob->f      = (forcefield_energyterm) rosetta_energy;
  ob->grad_f = (forcefield_gradenergyterm) rosetta_gradient;

  return (PyObject*) ob;
}
