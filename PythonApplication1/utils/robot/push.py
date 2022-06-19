import time
import numpy as np

import utils.utils as utils
from utils.arg.model import ArgsModel
from utils.robot.model import RobotModel
from simulator import Simulator
from utils.custom_types import NDArray

#from utils.arg.model import ArgsModel
#from utils.robot.model import RobotModel
from utils.robot.sim import RobotSim
#from utils.robot.objects import RobotObjects

#from utils.robot.mask import RobotMask
#from utils.robot.push import RobotPush
from utils.robot.move import RobotMove
from utils.robot.gripper import RobotGripper
#from utils.robot.camera import RobotCamera
#from utils.robot.grasp import RobotGrasp


class RobotPush():

    def __init__(self):
        pass
    
    def push(self, sim: RobotSim, m: RobotModel, gripper: RobotGripper, mover: RobotMove, 
        position, heightmap_rotation_angle, workspace_limits
    ):
        print('Executing: push at (%f, %f, %f)' % (position[0], position[1], position[2]))
        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

        # Adjust pushing point to be on tip of finger
        position[2] = position[2] + 0.026

        # Compute pushing direction
        push_orientation = [1.0,0.0]
        push_direction = np.asarray([push_orientation[0]*np.cos(heightmap_rotation_angle) - push_orientation[1]*np.sin(heightmap_rotation_angle), push_orientation[0]*np.sin(heightmap_rotation_angle) + push_orientation[1]*np.cos(heightmap_rotation_angle)])

        # Move gripper to location above pushing point
        pushing_point_margin = 0.1
        location_above_pushing_point = (position[0], position[1], position[2] + pushing_point_margin)

        # Compute gripper position and linear movement increments
        tool_position = location_above_pushing_point
        sim_ret, UR5_target_position = sim.engine.global_position_get(_ObjID = m.UR5_target_handle)
        #sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.05*move_direction/move_magnitude
        num_move_steps = int(np.floor(move_direction[0]/move_step[0]))

        # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = sim.engine.global_rotation_get(_ObjID = m.UR5_target_handle)
        #sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))

        # Simultaneously move and rotate gripper
        #for step_iter in range(max(num_move_steps, num_rotation_steps)):
        #    vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
        #    vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
        #vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
        #vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            sim.engine.global_position_set(_ObjID = m.UR5_target_handle, _NewPos3D = (UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)))
            sim.engine.global_rotation_set(_ObjID = m.UR5_target_handle, _NewRot3D = (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2))
        sim.engine.global_position_set(_ObjID = m.UR5_target_handle, _NewPos3D = (tool_position[0],tool_position[1],tool_position[2]))
        sim.engine.global_rotation_set(_ObjID = m.UR5_target_handle, _NewRot3D = (np.pi/2, tool_rotation_angle, np.pi/2))

        # Ensure gripper is closed
        gripper.close_gripper(sim, m)

        # Approach pushing point
        mover.move_to(sim, m, position, None)

        # Compute target location (push to the right)
        push_length = 0.13
        target_x = min(max(position[0] + push_direction[0]*push_length, workspace_limits[0][0]), workspace_limits[0][1])
        target_y = min(max(position[1] + push_direction[1]*push_length, workspace_limits[1][0]), workspace_limits[1][1])
        push_length = np.sqrt(np.power(target_x-position[0],2)+np.power(target_y-position[1],2))

        # Move in pushing direction towards target location
        mover.move_to(sim, m, [target_x, target_y, position[2]], None)

        # Move gripper to location above grasp target
        mover.move_to(sim, [target_x, target_y, location_above_pushing_point[2]], None)

        push_success = True

        return push_success
