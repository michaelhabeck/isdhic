#include "isdhic.h"

#ifdef __cplusplus
 extern "C" {
#endif

static PyMethodDef methods[] = {
  {"nblist", (PyCFunction) PyNBList_nblist, 1},
  {"prolsq", (PyCFunction) PyProlsq_New, 1},
  {"rosetta", (PyCFunction) PyRosetta_New, 1,},
  {NULL, NULL}
};


void init_isdhic(void) {

  import_array();
  Py_InitModule("_isdhic", methods);

  /* set object types correctly */

  PyProlsq_Type.ob_type = &PyType_Type;
  PyRosetta_Type.ob_type = &PyType_Type;
  PyNBList_Type.ob_type = &PyType_Type;
}

#ifdef __cplusplus
}
#endif
