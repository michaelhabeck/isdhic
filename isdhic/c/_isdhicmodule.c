#include "isdhic.h"

#ifdef __cplusplus
 extern "C" {
#endif

static PyMethodDef methods[] = {
  {"nblist", (PyCFunction) PyNBList_nblist, 1},
  {"universe", (PyCFunction) PyUniverse_New, 1},
  {"prolsq", (PyCFunction) PyProlsq_New, 1},
  {"boltzmannprior", (PyCFunction) PyBoltzmannPrior_New, 1},
  {"tsallisprior", (PyCFunction) PyTsallisPrior_New, 1},
  {"rosetta", (PyCFunction) PyRosetta_New, 1,},
  /*  {"dataset", (PyCFunction) PyDataSet_New, 1},
  {"datum", (PyCFunction) PyDatum_New, 1},
  {"noe", (PyCFunction) PyNOE_New, 1},
  {"centroiddistance", (PyCFunction) PyCentroidDistance_New, 1},
  {"ispa", (PyCFunction) PyISPA_New, 1},
  {"distancerestraint", (PyCFunction) PyDistanceRestraint_New, 1},
  {"errormodel", (PyCFunction) PyErrorModel_New, 1},
  {"lognormal", (PyCFunction) PyLognormal_New, 1},
  {"normal", (PyCFunction) PyNormal_New, 1},
  {"lowerupper", (PyCFunction) PyLowerUpper_New, 1}, */
  {NULL, NULL}
};


void init_isdhic(void) {

  import_array();
  Py_InitModule("_isdhic", methods);

  /* set object types correctly */

  PyUniverse_Type.ob_type = &PyType_Type;
  PyProlsq_Type.ob_type = &PyType_Type;
  PyBoltzmannPrior_Type.ob_type = &PyType_Type;
  PyTsallisPrior_Type.ob_type = &PyType_Type;
  /*  PyDatum_Type.ob_type = &PyDatum_Type;
  PyDataSet_Type.ob_type = &PyDataSet_Type;
  PyNOE_Type.ob_type = &PyType_Type;
  PyISPA_Type.ob_type = &PyType_Type;
  PyDistanceRestraint_Type.ob_type = &PyType_Type;
  PyErrorModel_Type.ob_type = &PyType_Type;
  PyLognormal_Type.ob_type = &PyType_Type;
  PyNormal_Type.ob_type = &PyType_Type;
  PyLowerUpper_Type.ob_type = &PyType_Type;*/
}

#ifdef __cplusplus
}
#endif
