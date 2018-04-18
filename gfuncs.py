#from tables import openFile,IsDescription,createGroup,Int32Col,StrCol

#try:
#   from IPython import ColorANSI
#   from IPython.genutils import Term
#   tc = ColorANSI.TermColors()
#except:
#   pass

#from Operable import Operable

#import numpy as np
#import pylab as pl
# import inspect

def isBuiltinListType(typ):
   builtin_list_types = [list, tuple]
   return typ in builtin_list_types

def isBuiltinScalarType(typ):
   builtin_scalar_types = [float, long, int, complex]
   return typ in builtin_scalar_types

def isBuiltinType(typ):
   builtin_types = [float, long, int, complex, str, bool]
   return typ in builtin_types

def isNpScalarType(typ):
   import numpy as np
   #np_types = [
   #   np.int,   np.int8,  np.int16,  np.int32,   np.int64,
   #   np.uint,  np.uint8, np.uint16, np.uint32,  np.uint64,
   #   np.float,                      np.float32, np.float64,   np.float128,
   #   np.complex,                                np.complex64, np.complex128, np.complex256]
   
   np_types = [
      np.int,   np.int8,  np.int16,  np.int32,   np.int64,
      np.uint,  np.uint8, np.uint16, np.uint32,  np.uint64,
      np.float,                      np.float32, np.float64,
      np.complex,                                np.complex64, np.complex128]

   return typ in np_types

def isNpArrayType(typ):
   import numpy as np
   np_array_types = [np.ndarray]
   return typ in np_array_types or issubclass(typ, np.ndarray)

def isNpType(typ):
   return isNpScalarType(typ) or isNpArrayType(typ)

def isScalar(typ):
   return isBuiltinScalarType(typ) or isNpScalarType(typ)

def isArray(typ):
   return isNpArrayType(typ)



