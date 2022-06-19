class C:
    def __init__(self):
        self._x = None

    def getx(self):
        return self._x

    def setx(self, value):
        print('before set')
        self._x = value

    def delx(self):
        del self._x

    x = property(getx, setx, delx, "I'm the 'x' property.")


import numpy as np
from typing import Generic, TypeVar
Shape = TypeVar("Shape")
DType = TypeVar("DType")
class NDArray(np.ndarray, Generic[Shape, DType]):
    def __init__(self, shape, dtype):
        self._x = None

    def getx(self):
        print('before NDArray get')
        return self._x

    def setx(self, value):
        print('before NDArray set')
        self._x = value

    def delx(self):
        del self._x

    #y = property(getx, setx, delx, "I'm the 'x' property.")

class SomeModel():
    _y: NDArray["3,2", float] = NDArray(shape=(3,2), dtype=float)

    def getx(self):
        print('before NDArray get')
        return self._y

    def setx(self, value):
        print('before NDArray set')
        self._y = value

    def setx(self, value):
        print('before NDArray set')
        self._y = value

    def delx(self):
        del self._x

    y = property(getx, setx, delx, "I'm the 'x' property.")


    #def __init__(self, shape, dtype):
    #    self.shape_orig = str(shape)
    #    self.dtype_orig = dtype
    #    #print('init was done', self.shape_orig, self.dtype_orig)


    def __init__(self):
        self.y = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]], dtype = float)
        print(type(self.y))

class Library():

    @staticmethod
    def getx(model2):
        print('before Library get', model2)
        return model2._y
    
    @staticmethod
    def setx(model2, value):
        print('before Library set', model2)
        model2._y = value
        
    @staticmethod
    def delx(model2):
        del model2._y

        
class SomeModel2():

    _y: NDArray["3,2", float] = NDArray(shape=(3,2), dtype=float)
    #def getx(self): print('before NDArray get'); return self._y
    #def setx(self, value): print('before NDArray set'); self._y = value
    #def delx(self): del self._x
    y = property(Library.getx, Library.setx, Library.delx, "I'm the 'x' property.")
    #y = property(getx, setx, delx, "I'm the 'x' property.")

    def __init__(self):
        print('1:\n',self.y)
        self.y = 15
        print('2:\n',self.y)
        self.y = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]], dtype = float)
        print('3:\n',self.y)

        #print(type(self.y), self.y)
        #self.y = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]], dtype = float)
        #print(type(self.y), self.y)


class NDArray3(np.ndarray, Generic[Shape, DType]):
    def getx(self, obj):
    #def getx(self):
        #print('before NDArray get')
        return self

    #def setx(self, obj, value, debug = True):
    def setx(self, obj, value): # obj is property owner
        old: str = str(self.shape) + str(self.dtype)
        newbie: str = str(value.shape) + str(value.dtype)
        #if old != newbie:# or debug:
        print('old', old, self)
        print('newbie', newbie, value)
        assert newbie == old
        #self = value
        #self.__dict__.update(value.__dict__)
        #self.__init__(value)
        #print('obj=',obj)
        #self[0][0] = value[1][1]

        #mask = (self == -999)
        #self[mask] = value[mask]

        #self[self!=-999999]=value[self!=-999999]
        self[:]=value[:]

        #obj = value

    def delx(self, obj):
        #del self._x
        del self


class SomeModel3():

    #_y: NDArray3["3,2", float] = NDArray3(shape=(3,2), dtype=float)
    _y: NDArray3["2,3,2", float] = NDArray3(shape=(2,3,2), dtype=float)
    y = property(_y.getx, _y.setx, _y.delx, "_y")

    def __init__(self):
        print(self.y.__doc__)
        print('1st property set:-------------')
        #self.y = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]], dtype = float) # OK
        #self.y = np.asarray([[1, 2], [3, 4], [5, 6]], dtype = float) # OK
        self.y = np.asarray([
            [[11, 12], [13, 14], [15, 16]],
            [[1, 2], [3, 4], [5, 6]] ], dtype = float) # OK
        print('2nd property get:-------------')
        print('get=', self.y)
        #self.y = np.asarray([[1, 2], [3, 4]], dtype = float) # ERROR
        #self.y = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]], dtype = int) # ERROR


if __name__ == '__main__':
    #print('ex1---------------------')
    #obj = C()
    #obj.x = 14
    #print(obj.x)

    #print('ex2---------------------')
    #m = SomeModel()

    #print('ex3---------------------')
    #m = SomeModel2()

    print('ex4---------------------')
    m = SomeModel3()

    print('end---------------------')
