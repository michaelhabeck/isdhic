#define NO_IMPORT_ARRAY

#include "isdhic.h"

static void free_attrs(PyUniverseObject *self) {

  if (self->coords) {
    Py_DECREF((PyObject*) self->coords);
  }
  if (self->forces) {
    Py_DECREF((PyObject*) self->forces);
  }

  self->n_particles = 0;

  self->coords = NULL;
  self->forces = NULL;
}

static void dealloc(PyUniverseObject *self) {
  free_attrs(self);
  PyObject_Del(self);
}

static PyMethodDef methods[] = {
  {NULL, NULL}
};

static PyObject *getattr(PyUniverseObject *self, char *name) {

  if (!strcmp(name, "n_particles")) {
    return Py_BuildValue("i", self->n_particles);
  }
  else if (!strcmp(name, "coords")) {
    if (!self->coords) {
      RETURN_PY_NONE;
    }
    else {
      Py_INCREF(self->coords);
      return (PyObject*) self->coords;
    }
  }
  else if (!strcmp(name, "forces")) {
    if (!self->forces) {
      RETURN_PY_NONE;
    }
    else {
      Py_INCREF(self->forces);
      return (PyObject*) self->forces;
    }
  }
  else {
    return Py_FindMethod(methods, (PyObject *)self, name);
  }
}

static int setattr(PyUniverseObject *self, char *name, PyObject *op) {

  PyArrayObject *coords;

  if (!strcmp(name, "coords")) {

    /* TODO: assert that array type is DOUBLE */

    if (!PyArray_Check(op)) {
      RAISE(PyExc_TypeError, "numpy array expected.", -1);
    }

    coords = (PyArrayObject*) op;

    if (coords->nd != 2) {
      RAISE(PyExc_TypeError, "array of rank 2 expected.", -1);
    }

    if ((coords->dimensions[0] != self->n_particles) || (coords->dimensions[1] != 3)) {
      RAISE(PyExc_TypeError, "array of rank (n_atoms, 3) expected.", -1);
    }

    if (self->coords) {
      Py_DECREF((PyObject*) self->coords);
    }

    self->coords = coords;

    Py_INCREF((PyObject*) coords);     
  }
    
  else if (!strcmp(name, "forces")) {

    if (self->forces) Py_DECREF(self->forces);

    self->forces = (PyArrayObject*) op;
    Py_INCREF(op);
  }

  else {
    RAISE(PyExc_AttributeError, "Attribute does not exist or cannot be set", -1);
  }
  return 0;
}

static char __doc__[] = "Universe. Container for all particles."; 

PyTypeObject PyUniverse_Type = { 
	PyObject_HEAD_INIT(0)
	0,				/*ob_size*/
	"universe",		        /*tp_name*/
	sizeof(PyUniverseObject),   	/*tp_basicsize*/
	0,				/*tp_itemsize*/

		/* methods */

	(destructor)dealloc,	        /*tp_dealloc*/
	(printfunc)NULL,		/*tp_print*/
	(getattrfunc)getattr,	        /*tp_getattr*/
	(setattrfunc)setattr,	        /*tp_setattr*/
	(cmpfunc)NULL,			/*tp_compare*/
	(reprfunc)NULL,			/*tp_repr*/

	NULL,				/*tp_as_number*/
	NULL,				/*tp_as_sequence*/
	NULL,		 		/*tp_as_mapping*/

	(hashfunc)0,			/*tp_hash*/
	(ternaryfunc)0,			/*tp_call*/
	(reprfunc)0,			/*tp_str*/
		
	0L,0L,0L,0L,
	__doc__ 	                /*Documentation string */
};

PyObject * PyUniverse_New(PyObject *self, PyObject *args) {

  int n;
  PyUniverseObject *ob;

  if (!PyArg_ParseTuple(args, "i", &n)) {
    RAISE(PyExc_TypeError, "number of particles expected.", NULL);
  }
  if (!(ob = PyObject_NEW(PyUniverseObject, &PyUniverse_Type))) {
    RAISE(PyExc_StandardError, "PyUniverse constructor failed.", NULL);
  }
  ob->n_particles = n;

  ob->coords = NULL;
  ob->forces = NULL;

  return (PyObject*) ob;
}
