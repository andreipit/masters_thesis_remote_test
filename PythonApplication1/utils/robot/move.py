import time
import numpy as np

import utils.utils as utils
from utils.arg.model import ArgsModel
from utils.robot.model import RobotModel
from utils.robot.sim import RobotSim
from simulator import Simulator
from utils.custom_types import NDArray

class RobotMove():

    def __init__(self):
        pass

    
    def move_to(self, sim: RobotSim, m: RobotModel, tool_position, tool_orientation):

         #sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        #sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
        sim_ret, UR5_target_handle = m.engine.gameobject_find(_Name = 'UR5_target')
        sim_ret, UR5_target_position = m.engine.global_position_get(_ObjID = UR5_target_handle)

        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.02*move_direction/move_magnitude
        num_move_steps = int(np.floor(move_magnitude/0.02))

        for step_iter in range(num_move_steps):
            m.engine.global_position_set(_ObjID = UR5_target_handle, _NewPos3D = (UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1], UR5_target_position[2] + move_step[2]))
            #vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0], UR5_target_position[1] + move_step[1], UR5_target_position[2] + move_step[2]),vrep.simx_opmode_blocking)
            #sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client,self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = m.engine.global_position_get(_ObjID = UR5_target_handle)
        
        #vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
        m.engine.global_position_set(_ObjID = UR5_target_handle, _NewPos3D = (tool_position[0],tool_position[1],tool_position[2]))

