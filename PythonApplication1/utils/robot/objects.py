import time
import numpy as np
import os

from utils.arg.model import ArgsModel
from utils.robot.model import RobotModel

class RobotObjects():

    prev_obj_positions = []
    obj_positions = []

    def __init__(self):
        pass

    def seed_test_objects(self, m: RobotModel):
        print('is_testing',m.num_obj)
        file = open(m.test_preset_file, 'r')
        file_content = file.readlines()
        m.test_obj_mesh_files = []
        m.test_obj_mesh_colors = []
        m.test_obj_positions = []
        m.test_obj_orientations = []
        for object_idx in range(m.num_obj):
            file_content_curr_object = file_content[object_idx].split()
            print('append')
            m.test_obj_mesh_files.append(os.path.join(m.obj_mesh_dir,file_content_curr_object[0]))
            m.test_obj_mesh_colors.append([float(file_content_curr_object[1]),float(file_content_curr_object[2]),float(file_content_curr_object[3])])
            m.test_obj_positions.append([float(file_content_curr_object[4]),float(file_content_curr_object[5]),float(file_content_curr_object[6])])
            m.test_obj_orientations.append([float(file_content_curr_object[7]),float(file_content_curr_object[8]),float(file_content_curr_object[9])])
        file.close()
        m.obj_mesh_color = np.asarray(m.test_obj_mesh_colors)


    #def instantiate_cubes(self, m: RobotModel):

    #    # Add objects to simulation environment
    #    self.add_objects(m)

    
    #def add_objects(self, sim: RobotSim, m: RobotModel):
    def add_objects(self, m: RobotModel):
        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        m.object_handles = []
        sim_obj_handles = []
        if m.stage == 'grasp_only': # true at 1.2
            obj_number = 1
            m.obj_mesh_ind = np.random.randint(0, len(m.mesh_list), size=m.num_obj)
            if m.goal_conditioned or m.grasp_goal_conditioned:
                obj_number = len(m.obj_mesh_ind)
        else:
            obj_number = len(m.obj_mesh_ind)

        # load each file.obj from folder "objects"
        for object_idx in range(obj_number): # in 1.2: 0,1,2,3,4 (obj_number == 5)
            # 1) get path

            #print('mesh list', len(m.mesh_list))
            #print('obj_mesh_ind ', len(m.obj_mesh_ind), m.obj_mesh_ind)
            #print('object_idx', object_idx)
            #print('m.obj_mesh_ind[object_idx', m.obj_mesh_ind[object_idx])

            curr_mesh_file = os.path.join(m.obj_mesh_dir, m.mesh_list[m.obj_mesh_ind[object_idx]])
            # curr_mesh_file => E:\notes\mipt4\repos\xukechun\Recreate\PythonApplication1\objects\blocks\4.obj
            if m.is_testing and m.test_preset_cases: # false and false in 1.2
                curr_mesh_file = m.test_obj_mesh_files[object_idx]
            
            # 2) add iteraion index to name
            curr_shape_name = 'shape_%02d' % object_idx # => shape_00

            # 3) find place where to drop cube (at height 0.15)
            drop_x = (m.workspace_limits[0][1] - m.workspace_limits[0][0] - 0.2) * np.random.random_sample() + m.workspace_limits[0][0] + 0.1
            drop_y = (m.workspace_limits[1][1] - m.workspace_limits[1][0] - 0.2) * np.random.random_sample() + m.workspace_limits[1][0] + 0.1  # + 0.1
            object_position = [drop_x, drop_y, 0.15] # [-0.5563970338899049, -0.05543686472451201, 0.15]
            
            # 4) set random rotation
            object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
            # ignored in 1.2:
            if m.is_testing and m.test_preset_cases: # false and false in 1.2
                object_position = [m.test_obj_positions[object_idx][0], m.test_obj_positions[object_idx][1] + 0.1, m.test_obj_positions[object_idx][2]]
                object_orientation = [m.test_obj_orientations[object_idx][0], m.test_obj_orientations[object_idx][1], m.test_obj_orientations[object_idx][2]]
            
            # 5) set color from constant list in robot model, by index
            object_color = [m.obj_mesh_color[object_idx][0], m.obj_mesh_color[object_idx][1], m.obj_mesh_color[object_idx][2]]
            
            for i in range(3): # give him 3 attempts to call function in scene
                ret_resp, ret_ints, ret_floats, ret_strings, ret_buffer = m.engine.getcomponent_and_run(
                    _ObjName = 'remoteApiCommandServer', 
                    _FunName = 'importShape',
                    _Input = ([0,0,255,0], #int
                             object_position + object_orientation + object_color, #float
                             [curr_mesh_file, curr_shape_name], #string
                             bytearray()) # buffer
                )
                if ret_resp == 8: 
                    print('Failed to add new objects to simulation. Trying again.')
                    #time.sleep(1)
                else:
                    break

            if ret_resp == 8:
                print('Failed to add new objects to simulation. Please restart.')
                exit()

            curr_shape_handle = ret_ints[0]
            m.object_handles.append(curr_shape_handle)
            if not (m.is_testing and m.test_preset_cases):
                time.sleep(0.5)
        self.prev_obj_positions = []
        self.obj_positions = []

    #def get_obj_positions(self, sim: RobotSim, m: RobotModel):
    def get_obj_positions(self, m: RobotModel):

        obj_positions = []
        for object_handle in m.object_handles:
            sim_ret, object_position = m.engine.global_position_get(_ObjID = object_handle)
            #sim_ret, object_position = vrep.simxGetObjectPosition(self.sim_client, object_handle, -1, vrep.simx_opmode_blocking)
            obj_positions.append(object_position)

        return obj_positions


#args=['Hello World!',[1,2,3],59]
#ret=vrep.simxCallScriptFunction('myFunctionName@Dummy2',vrep.sim_scripttype_childscript,args,sim.sim_client)
#print('ret=',ret)

                    
#res,retInts,retFloats,retStrings,retBuffer=vrep.simxCallScriptFunction(
#    sim.sim_client,
#    'DefaultCamera',
#    vrep.sim_scripttype_childscript,
#    'myPrinter',
#    [],
#    [],
#    ['Hello world!'],
#    bytearray(),
#    vrep.simx_opmode_blocking)
            

#https://github.com/mrabbah/vrep-python-bridge/blob/master/complexCommandTest.py
#my_ret = vrep.simxCallScriptFunction(
#    clientID = sim.sim_client, 
#    scriptDescription = 'DefaultCamera',
#    options = vrep.sim_scripttype_childscript,
#    functionName = 'myPrinter',
#    inputInts = [0,0,255,0], 
#    inputFloats = object_position + object_orientation + object_color, 
#    inputStrings = [curr_mesh_file, curr_shape_name],
#    inputBuffer = bytearray(), 
#    operationMode = vrep.simx_opmode_blocking
#)

# 6) 
# https://www.coppeliarobotics.com/helpFiles/en/b0RemoteApiExtension.htm
# https://www.coppeliarobotics.com/helpFiles/en/remoteApiFunctions.htm#simxCallScriptFunction
# Remotely calls a CoppeliaSim script function. 
# When calling simulation scripts, then simulation must be running. 
# error => ret_resp != 8
# return my scene => 8 [] [] [] bytearray(b'')
# return orig scene => 0 [96] [] [] bytearray(b'')
# fix:
# to be able to call your script function, your script needs to be initialized. A child script is a simulation script, and only gets initialized when simulation is running.
# Try maybe using a customization script instead.


#ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(
#    clientID = sim.sim_client, 
#    scriptDescription = 'remoteApiCommandServer',
#    options = vrep.sim_scripttype_childscript,
#    functionName = 'importShape',
#    inputInts = [0,0,255,0], 
#    inputFloats = object_position + object_orientation + object_color, 
#    inputStrings = [curr_mesh_file, curr_shape_name],
#    inputBuffer = bytearray(), 
#    operationMode = vrep.simx_opmode_blocking
#)

#ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(
#    clientID = sim.sim_client, 
#    scriptDescription = 'Dummy2',
#    options = vrep.sim_scripttype_childscript,
#    functionName = 'myFunctionName',
#    inputInts = [0,0,255,0], 
#    inputFloats = object_position + object_orientation + object_color, 
#    inputStrings = [curr_mesh_file, curr_shape_name],
#    inputBuffer = bytearray(), 
#    operationMode = vrep.simx_opmode_blocking
#)


