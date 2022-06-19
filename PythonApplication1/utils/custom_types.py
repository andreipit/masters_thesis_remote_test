import numpy as np
#class NDArray(np.ndarray):
#    pass


from typing import Generic, TypeVar

Shape = TypeVar("Shape")
DType = TypeVar("DType")


class NDArray(np.ndarray, Generic[Shape, DType]):
    
    def getx(self, obj):
        #print('before NDArray get')
        return self

    def setx(self, obj, value, debug = False):
        old: str = str(self.shape) + str(self.dtype)
        newbie: str = str(value.shape) + str(value.dtype)
        if old != newbie or debug:
            print('old', old)
            print('newbie', newbie)
        assert newbie == old
        #obj = value
        self[:]=value[:]
        #self[self!=-999999]=value[self!=-999999]



    def delx(self, obj):
        del self

    #@staticmethod
    #def set_value(variable, value, debug = False):
    #    old: str = str(variable.shape) + str(variable.dtype)
    #    newbie: str = str(value.shape) + str(value.dtype)
    #    if old != newbie or debug:
    #        print('old', old)
    #        print('newbie', newbie)
    #    assert newbie == old
    #    variable = value
    #    #def workspace_limits(self, value): assert  str(value.shape) + str(value.dtype) == str(self._workspace_limits.shape) + str(self._workspace_limits.dtype); self._workspace_limits = value
    

    """  
    Source: https://stackoverflow.com/a/64032593/10820672
    https://stackoverflow.com/questions/52729346/is-there-a-compact-way-to-declare-several-similar-properties-in-a-python-class
    https://docs.python.org/3/library/functions.html#property
    Use this to type-annotate numpy arrays, e.g. 
        image: Array['H,W,3', np.uint8]
        xy_points: Array['N,2', float]
        nd_mask: Array['...', bool]
    """
    """
    Example:
    from utils.types import NDArray
    import numpy as np
    workspace_limits: NDArray["3,2", float] = NDArray(shape=(3,2), dtype=float)# [[-7.24e-01 -2.76e-01] [-2.24e-01  2.24e-01] [-1.00e-04  4.00e-01]] = np.zeros((3, 3), dtype=np.complex64)
    a22 = np.zeros((3, 2), dtype=int) # 3,2 int is also possible, (3,3) - just recommendation
    print(a22)

    workspace_limits: NDArray["3,2", float] = NDArray(shape=(3,2), dtype=float)# [[-7.24e-01 -2.76e-01] [-2.24e-01  2.24e-01] [-1.00e-04  4.00e-01]] = np.zeros((3, 3), dtype=np.complex64)
    @property
    def workspace_limits_prop(self): return self.workspace_limits
    @workspace_limits_prop.setter
    def workspace_limits_prop(self, value):
        assert  str(value.shape) + str(value.dtype) == str(self.workspace_limits.shape) + str(self.workspace_limits.dtype)
        self.workspace_limits = value
        #old: string = str(self.workspace_limits.shape) + str(self.workspace_limits.dtype)
        #newbie: string = str(value.shape) + str(value.dtype)
        #print('old', old)
        #print('newbie', newbie)
        #assert newbie == old
        #self.workspace_limits = value

    """

"""
Backup:


#@workspace_limits_prop.setter
#def workspace_limits_prop(self, value):
#    old: string = str(self.workspace_limits.shape) + str(self.workspace_limits.dtype)
#    newbie: string = str(value.shape) + str(value.dtype)
#    print('old', old)
#    print('newbie', newbie)
#    assert newbie == old
#    self.workspace_limits = value

#import numpy.typing as npt
#from typing import List, Tuple
from utils.custom_types import NDArray
#from nptyping import NDArray, Int, Shape, Float, Complex64 # https://github.com/ramonhagenaars/nptyping/blob/master/USERDOCS.md#Examples
#from nptyping import assert_isinstance

# --------------- Workspace setting -----------------
#workspace_limits: NDArray[Shape["3, 2"], Float] # [[-7.24e-01 -2.76e-01] [-2.24e-01  2.24e-01] [-1.00e-04  4.00e-01]] = np.zeros((3, 3), dtype=np.complex64)
workspace_limits: NDArray["3,2", float] = NDArray(shape=(3,2), dtype=float)# [[-7.24e-01 -2.76e-01] [-2.24e-01  2.24e-01] [-1.00e-04  4.00e-01]] = np.zeros((3, 3), dtype=np.complex64)
#workspace_limits: Dict[Tuple[int, int], float] # [[-7.24e-01 -2.76e-01] [-2.24e-01  2.24e-01] [-1.00e-04  4.00e-01]] = np.zeros((3, 3), dtype=np.complex64)
#workspace_limits: List[Tuple[int, int]] # [[-7.24e-01 -2.76e-01] [-2.24e-01  2.24e-01] [-1.00e-04  4.00e-01]] = np.zeros((3, 3), dtype=np.complex64)
#workspace_limits: NDArray
#workspace_limits = None # ndarray((3,2), dtype=float64)
#workspace_limits : npt.NDArray[np.complex64]

 # --------------- Workspace setting -----------------
#self.workspace_limits = np.array([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        
print('orig:', type(self.workspace_limits))
print('orig1:', self.workspace_limits.shape, self.workspace_limits.dtype)
self.workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
print('orig2:', self.workspace_limits.shape, self.workspace_limits.dtype)
assert self.workspace_limits.shape == (3, 2) and self.workspace_limits.dtype == 'float64'
#assert isinstance(self.workspace_limits, NDArray[Shape["3, 2"], Float])
#self.workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]], dtype=float64) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)



"""


#def workspace_limits(self, value): self._workspace_limits.set_value(self._workspace_limits, value)

#class NDArray(np.ndarray, Generic[Shape, DType]):
    
#    shape_orig = None
#    dtype_orig = None

#    def __init__(self, shape, dtype):
#        self.shape_orig = str(shape)
#        self.dtype_orig = dtype
#        #print('init was done', self.shape_orig, self.dtype_orig)

#    #@staticmethod
#    def set_value(self, variable, value, debug = True):
#        #old: str = str(variable.shape) + str(variable.dtype)
#        old: str = self.shape_orig + self.dtype_orig
#        newbie: str = str(value.shape) + str(value.dtype)
#        if old != newbie or debug:
#            print('old', old)
#            print('newbie', newbie)
#        assert newbie == old
#        variable = value
#        #def workspace_limits(self, value): assert  str(value.shape) + str(value.dtype) == str(self._workspace_limits.shape) + str(self._workspace_limits.dtype); self._workspace_limits = value



#_workspace_limits: NDArray["3,2", float] = NDArray(shape=(3,2), dtype=float)# [[-7.24e-01 -2.76e-01] [-2.24e-01  2.24e-01] [-1.00e-04  4.00e-01]] = np.zeros((3, 3), dtype=np.complex64)
#@property
#def workspace_limits(self): return self._workspace_limits
#@workspace_limits.setter 
#def workspace_limits(self, value): NDArray.set_value(self._workspace_limits, value)
    