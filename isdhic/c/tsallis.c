#define NO_IMPORT_ARRAY

#include "isdhic.h"

#define MINLOG 1.e-307

static int _result;

int tsallis_error(void) {
  return _result;
}
			   
static double energy(PyTsallisPriorObject *self, PyUniverseObject *universe, double *E_boltzmann) {

  double arg, E=0.;
  double q = self->q;

  *E_boltzmann = boltzmann_energy((PyBoltzmannPriorObject*) self, universe);

  if (fabs(q - 1.) < 1.e-10) {
    return *E_boltzmann;
  }

  E = *E_boltzmann - self->beta * self->E_min;

  if ((q-1) * E < -1){
    printf("tsallis_energy: E= %3.e\n", E);
    PyErr_SetString(PyExc_OverflowError, "tsallis_energy: E <= 0.");
    _result = -1;
    return 0.;
  }

  arg = 1. + (q - 1.) * E;

  if (arg < MINLOG) {
    PyErr_SetString(PyExc_OverflowError, "log argument underflow");
    _result = -1.;
    return 0.;
  }

  _result = 0;

  E = log(arg) * q / (q - 1.) + self->beta * self->E_min;

  return E;
}

static int gradient_update(PyTsallisPriorObject *self, PyUniverseObject *universe) {

  double E, arg, q;
  int result, i;
  vector *grad;

  q = self->q;

  if (fabs(q - 1.) < 1.e-10) {
    return boltzmann_update_gradient((PyBoltzmannPriorObject*) self, universe, NULL);
  }
  result = boltzmann_update_gradient((PyBoltzmannPriorObject*) self, universe, &E);

  if (result){
    return result;
  }

  E -= self->beta * self->E_min;

  arg = 1. + (q - 1.) * E;

  if (arg < MINLOG) {
    PyErr_SetString(PyExc_OverflowError, "log argument underflow");
    return -1;
  }

  arg = q / arg;
  grad = (vector*) universe->forces->data;
      
  for (i = 0; i < universe->n_particles; i++)
    vector_imul(grad[i], arg);

  return 0;
}

PyObject* PyTsallisPrior_Energy(PyTsallisPriorObject *self,
				PyObject *args) {

  PyUniverseObject *universe;
  double E, E_boltzmann;

  if (!PyArg_ParseTuple(args, "O!", &PyUniverse_Type, &universe)){
    RAISE(PyExc_TypeError, "an universe is expected.", NULL);
  }
  if (!self->forcefield) {
    RAISE(PyExc_StandardError, "no force field set.", NULL);
  }

  E = energy(self, universe, &E_boltzmann);

  if (tsallis_error()) {
    return NULL;
  }
  return Py_BuildValue("(d,d)", E, E_boltzmann);
}

PyObject* PyTsallisPrior_UpdateGradient(PyTsallisPriorObject *self,
					PyObject *args) {

  PyUniverseObject *universe;
  int result;

  if (!PyArg_ParseTuple(args, "O!", &PyUniverse_Type, &universe)) {
    RAISE(PyExc_TypeError, "universe expected.", NULL);
  }
  if (!self->forcefield) {
    RAISE(PyExc_StandardError, "no force field set.", NULL);
  }

  result = gradient_update(self, universe);
  
  if (result < 0) {
    return NULL;
  }
  else {
    RETURN_PY_NONE;
  }
}

static PyMethodDef methods[] = {
  {"energy", (PyCFunction) PyTsallisPrior_Energy, 1},
  {"update_gradient", (PyCFunction) PyTsallisPrior_UpdateGradient, 1},
  {NULL, NULL }
};

static PyObject * getattr(PyTsallisPriorObject *self, char *name) {

  PyObject *ob;

  if (!strcmp(name, "q"))
    return Py_BuildValue("d", self->q);

  else if (!strcmp(name, "E_min"))
    return Py_BuildValue("d", self->E_min);

  else 
    ob = Py_FindMethod(methods, (PyObject*) self, name);

  if (ob)
    return ob;

  PyErr_Clear();

  return boltzmann_getattr((PyBoltzmannPriorObject*) self, name);
}

static int setattr(PyTsallisPriorObject *self, char *name, PyObject *op) {

  int result = 0;
  
  if (!strcmp(name, "q")) {
    self->q = (double) PyFloat_AsDouble(op);
  }
  else if (!strcmp(name, "E_min")) {
    self->E_min = (double) PyFloat_AsDouble(op);
  }
  else {
    result = boltzmann_setattr((PyBoltzmannPriorObject*) self, name, op);
  }
  return result;
}

static void dealloc(PyTsallisPriorObject *self) {
  boltzmann_dealloc((PyBoltzmannPriorObject*) self);
}

static char __doc__[] = ""; 

PyTypeObject PyTsallisPrior_Type = { 
	PyObject_HEAD_INIT(0)
	0,			       /*ob_size*/
	"tsallisprior",	               /*tp_name*/
	sizeof(PyTsallisPriorObject),  /*tp_basicsize*/
	0,			       /*tp_itemsize*/

	(destructor)dealloc,           /*tp_dealloc*/
	(printfunc)NULL,	       /*tp_print*/
       	(getattrfunc)getattr,          /*tp_getattr*/
	(setattrfunc)setattr,          /*tp_setattr*/
	(cmpfunc)NULL,	               /*tp_compare*/
	(reprfunc)NULL,	               /*tp_repr*/

	NULL,		               /*tp_as_number*/
	NULL,	                       /*tp_as_sequence*/
	NULL,		 	       /*tp_as_mapping*/

	(hashfunc)0,		       /*tp_hash*/
	(ternaryfunc)0, 	       /*tp_call*/
	(reprfunc)0,		       /*tp_str*/
		
	0L,0L,0L,0L,
	__doc__               /* Documentation string */
};

PyObject * PyTsallisPrior_New(PyObject *self, PyObject *args) {

  PyTsallisPriorObject *ob;

  ob = PyObject_NEW(PyTsallisPriorObject, &PyTsallisPrior_Type);

  ob->q = 1.000001;
  ob->E_min = 0.;

  return boltzmann_init((PyBoltzmannPriorObject*) ob);
}

