from simulation.api.coppelia_api import CoppeliaAPI
from simulation.api.unity3d_api import Unity3dAPI
from simulation import vrep

class Simulator(object):

    api:CoppeliaAPI = CoppeliaAPI()
    #api:CoppeliaAPI = Unity3dAPI()

    def __init__(self):
        pass

    def test(self):
        self.connect()

        #return_code, handle = self.gameobject_find('Floor')
        #print('Find: fun executed fine:', return_code == 0, 'GameObject ID:', handle)
        
        #self.position_set('Floor', (2.0, 4.1, -2))
        
        #ret, intDataOut, floatDataOut, stringDataOut, bufferOut = self.getcomponent_and_run(_ObjName = 'Floor', _FunName = 'myPrinter')
        #print('GetComponent',_ObjName, _FunName, ' Return:',ret, intDataOut, floatDataOut, stringDataOut, bufferOut)


    def connect(self):
        self.api.connect()

    def gameobject_find(self, _Name):
        return self.api.gameobject_find(_Name)

    def global_position_set(self, _NewPos3D, _ObjID = "-1", _ObjName = ""):
        self.api.global_position_set(_NewPos3D, _ObjID, _ObjName)
        
    def global_position_get(self, _ObjID = "-1", _ObjName = ""):
        return self.api.global_position_get(_ObjID, _ObjName)

    def global_position_get_joint(self, _ObjID = "-1", _ObjName = ""):
        return self.api.global_position_get_joint(_ObjID, _ObjName)
            
    def global_rotation_get(self, _ObjID = "-1", _ObjName = ""):
        return self.api.global_rotation_get(_ObjID, _ObjName)

    def global_rotation_set(self, _NewRot3D, _ObjID = "-1", _ObjName = ""):
        self.api.global_rotation_set(_NewRot3D, _ObjID, _ObjName)
    
    def joint_force_set(self, _ObjID, _Value):
        self.api.joint_force_set(_ObjID, _Value)

    def joint_target_velocity_set(self, _ObjID, _Value):
        self.api.joint_target_velocity_set(_ObjID, _Value)

    def getcomponent_and_run(self, _ObjName, _FunName, _Input):
        return self.api.getcomponent_and_run(_ObjName, _FunName, _Input)

    def stop(self):
        self.api.stop()

    def start(self):
        self.api.start()

    def sleep(self, _Time):
        self.api.sleep(_Time)

    def restart_hard(self, _ObjName):
        RG2_tip_handle = self.api.restart_hard(_ObjName) # we use _ObjName to check, that restart was sucsessfull
        return RG2_tip_handle

    def camera_image_rgb_get(self, _ObjID):
        return self.api.camera_image_rgb_get(_ObjID)

    def camera_image_depth_get(self, _ObjID):
        return self.api.camera_image_depth_get(_ObjID)
