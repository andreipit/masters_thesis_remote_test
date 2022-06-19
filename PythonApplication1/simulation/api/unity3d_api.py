class Unity3dAPI(object):
    # public methods ########################################################

    def __init__(self):
        pass

    def connect(self):
        # Connect to simulator
        vrep.simxFinish(-1) # Just in case, close all opened connections   # reason for only one vrep opening?????
        self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to V-REP on port 19997
        if self.sim_client == -1:
            print('Failed to connect to simulation (V-REP remote API server). Exiting.')
            exit()
        else:
            print('Connected to simulation.')

    def gameobject_find(self, _Name):
        #clientID: the client ID. refer to simxStart.
        #objectPath: the path of the object. See the section on accessing scene objects for details.
        #operationMode: a remote API function operation mode. Recommended operation mode for this function is simx_opmode_blocking
        # return: number returnCode, number handle; return_code=0 => found, handle=ID of obj
        # see bottom Readme for returnCode interpretation
        return_code, handle = vrep.simxGetObjectHandle(
            clientID = self.sim_client, 
            objectName = _Name, 
            operationMode = vrep.simx_opmode_blocking) # like coroutine doesn't block loop
        return return_code, handle
    
    def global_position_get(self, _ObjID = "-1", _ObjName = ""):
        if _ObjID == -1:
            code, _ObjID = self.gameobject_find(_ObjName)
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, _ObjID, -1, vrep.simx_opmode_blocking)
        return sim_ret, gripper_position

    def global_position_get_joint(self, _ObjID = "-1", _ObjName = ""):
        if _ObjID == -1:
            code, _ObjID = self.gameobject_find(_ObjName)
        sim_ret, gripper_position = vrep.simxGetJointPosition(
            clientID = self.sim_client, 
            jointHandle = _ObjID, 
            operationMode = vrep.simx_opmode_blocking)
        return sim_ret, gripper_position
    
    def global_rotation_get(self, _ObjID = "-1", _ObjName = ""):
        if _ObjID == -1:
            code, _ObjID = self.gameobject_find(_ObjName)
        sim_ret, cam_orientation = vrep.simxGetObjectOrientation(self.sim_client, _ObjID, -1, vrep.simx_opmode_blocking)
        return sim_ret, cam_orientation

    def joint_force_set(self, _ObjID, _Value):
        #vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
        vrep.simxSetJointForce(
            clientID = self.sim_client,  # was created at init
            jointHandle = _ObjID,       # ID of gameobject in scene
            force = _Value,
            operationMode = vrep.simx_opmode_blocking)
    
    def joint_target_velocity_set(self, _ObjID, _Value):
        #vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetVelocity(
            clientID = self.sim_client,  # was created at init
            jointHandle = _ObjID,       # ID of gameobject in scene
            targetVelocity = _Value,
            operationMode = vrep.simx_opmode_blocking)

    def global_rotation_set(self, _NewRot3D, _ObjID = "-1", _ObjName = ""):
        if _ObjID == -1:
            code, _ObjID = self.gameobject_find(_ObjName)
        vrep.simxSetObjectOrientation(
            clientID = self.sim_client,  # was created at init
            objectHandle = _ObjID,       # ID of gameobject in scene
            relativeToObjectHandle = -1, # -1 == global space
            eulerAngles = _NewRot3D,          # (-0.5,0,0.3), 
            operationMode = vrep.simx_opmode_blocking) # just default enum (coroutine settings)

    def global_position_set(self, _NewPos3D, _ObjID = "-1", _ObjName = ""):
        if _ObjID == -1:
            code, _ObjID = self.gameobject_find(_ObjName)
        vrep.simxSetObjectPosition(
            clientID = self.sim_client,  # was created at init
            objectHandle = _ObjID,       # ID of gameobject in scene
            relativeToObjectHandle = -1, # -1 == global space
            position = _NewPos3D,          # (-0.5,0,0.3), 
            operationMode = vrep.simx_opmode_blocking) # just default enum (coroutine settings)

    def getcomponent_and_run(self, _ObjName, _FunName, _Input):
        """
        ex: add child non-threaded script to Floor (in top place)
            function myPrinter(inInts,inFloats,inStrings,inBuffer)
                -- put your actuation code here
                print("myPrinter called")
                -- return("return of myPrinter")  
                return {8,5,3},{},{},''
            end
        return:
            when all is ok:
            => 0 [8, 5, 3] [] [] bytearray(b'')
            rename method myPriner -> myPrinter2
            => 8 [] [] [] bytearray(b'')   => 8 means not found
            => CoppeliaSim:error] External call to simCallScriptFunction failed (myPrinter@Floor): failed calling script function.
            rename gameobject Floor -> Floor2
            => 8 [] [] [] bytearray(b'')   => 8 means not found
            => [CoppeliaSim:error] External call to simCallScriptFunction failed (myPrinter@Floor): script does not exist.
        """
        ret, intDataOut, floatDataOut, stringDataOut, bufferOut = vrep.simxCallScriptFunction(
            clientID = self.sim_client,                   # was created at init
            scriptDescription = _ObjName,                 # gameobject name 'remoteApiCommandServer'
            options = vrep.sim_scripttype_childscript,    # like MonoBehaviour (ie not Editor or plugin)
            functionName = _FunName,                      # fun inside child script
            inputInts = _Input[0],    # [],               # arg1 - ints -> [0,0,255,0], 
            inputFloats = _Input[1],  # [],               # arg2 - floats
            inputStrings = _Input[2], # ['hello'],        # arg3 - strings
            inputBuffer = _Input[3],  # bytearray(),      # arg4 - bytes
            operationMode = vrep.simx_opmode_blocking,    # coroutine or not
        )
        return ret, intDataOut, floatDataOut, stringDataOut, bufferOut

    def stop(self):
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)

    def start(self):
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)

    def sleep(self, _Time):
        time.sleep(1)

    def restart_hard(self, _ObjName):

        sim_ret, RG2_tip_handle = self.gameobject_find(_ObjName)
        # gripper_position = RG2_tip_handle.position. Pos of the gripper end.
        #sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        sim_ret, gripper_position = self.global_position_get(_ObjID = RG2_tip_handle)
        
        # stop/start many times until gripper will be lower then 0.4 meters 
        # I've rotated a bit arm to make it lower 0.4 m. Angle between vertical = 70.
        while gripper_position[2] > 0.4: # V-REP bug requiring multiple starts and stops to restart
            self.stop()
            self.start()
            self.sleep(1)
            sim_ret, gripper_position = self.global_position_get(_ObjID = RG2_tip_handle)
            #vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            #vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            #time.sleep(1)
            #sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        return RG2_tip_handle

    def camera_image_rgb_get(self, _ObjID):
        sim_ret, resolution, raw_image = vrep.simxGetVisionSensorImage(self.sim_client, _ObjID, 0, vrep.simx_opmode_blocking)
        return sim_ret, resolution, raw_image

    def camera_image_depth_get(self, _ObjID):
        sim_ret, resolution, depth_buffer = vrep.simxGetVisionSensorDepthBuffer(self.sim_client, _ObjID, vrep.simx_opmode_blocking)
        return sim_ret, resolution, depth_buffer

    # private methods ########################################################

    def _restart_sim(self):
        # UR5_target_handle = GameObject.Find("UR5_target") (link is just ID). UR5_target - dummy, init pos for gripper end.
        sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        
        # transform.position
        vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
        
        # stop, play, wait 1 sec
        vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
        time.sleep(1)

        # RG2_tip_handle = GameObject.Find("UR5_tip"). It is the end of whole arm with gripper.
        sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)

        # gripper_position = RG2_tip_handle.position. Pos of the gripper end.
        sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
        
        # stop/start many times until gripper will be lower then 0.4 meters 
        # I've rotated a bit arm to make it lower 0.4 m. Angle between vertical = 70.
        while gripper_position[2] > 0.4: # V-REP bug requiring multiple starts and stops to restart
            print('restart', gripper_position[2])
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)





