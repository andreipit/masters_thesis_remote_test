import time
import numpy as np
from utils.robot.camera import RobotCamera

import utils.utils as utils
from utils.arg.model import ArgsModel
from utils.robot.model import RobotModel
from utils.robot.camera import RobotCamera
from utils.robot.sim import RobotSim
from utils.robot.gripper import RobotGripper
from utils.robot.move import RobotMove
from utils.robot.objects import RobotObjects
from simulator import Simulator
from utils.custom_types import NDArray

class RobotGrasp():

    def __init__(self):
        pass

     # Primitives ----------------------------------------------------------
    def grasp(self, a: ArgsModel, sim: RobotSim, m: RobotModel, cam: RobotCamera, gripper: RobotGripper, mover: RobotMove, obj: RobotObjects,
              position, 
              heightmap_rotation_angle, 
              workspace_limits):
        print('Executing: grasp at (%f, %f, %f)' % (position[0], position[1], position[2]))
        # Compute tool orientation from heightmap rotation angle
        tool_rotation_angle = (heightmap_rotation_angle % np.pi) - np.pi/2

        # Avoid collision with floor
        position = np.asarray(position).copy()
        position[2] = max(position[2] - 0.04, workspace_limits[2][0] + 0.02)

        # Move gripper to location above grasp target
        grasp_location_margin = 0.15
        # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
        location_above_grasp_target = (position[0], position[1], position[2] + grasp_location_margin)

        # Compute gripper position and linear movement increments
        tool_position = location_above_grasp_target

        sim_ret, UR5_target_position = m.engine.global_position_get(_ObjID = m.UR5_target_handle)
        #sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
        move_direction = np.asarray([tool_position[0] - UR5_target_position[0], tool_position[1] - UR5_target_position[1], tool_position[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.05*move_direction/move_magnitude

        #if move_direction == None or  move_direction[0] == None or move_step == None or move_step[0] == None or move_direction[0]/move_step[0] == None or np.floor(move_direction[0]/move_step[0]) == None:
        if move_direction[0] == None or move_step[0] == None or move_direction[0]/move_step[0] == None or np.floor(move_direction[0]/move_step[0]) == None:
            num_move_steps = 0
        else:
            num_move_steps = int(np.floor(move_direction[0]/move_step[0]))

        # Compute gripper orientation and rotation increments
        sim_ret, gripper_orientation = m.engine.global_rotation_get(_ObjID = m.UR5_target_handle)
        #sim_ret, gripper_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)
        rotation_step = 0.3 if (tool_rotation_angle - gripper_orientation[1] > 0) else -0.3
        num_rotation_steps = int(np.floor((tool_rotation_angle - gripper_orientation[1])/rotation_step))

        # Simultaneously move and rotate gripper
        for step_iter in range(max(num_move_steps, num_rotation_steps)):
            m.engine.global_position_set(
                _ObjID = m.UR5_target_handle, 
                _NewPos3D = (
                    UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), 
                    UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), 
                    UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)
                )
            )
            m.engine.global_rotation_set(
                _ObjID = m.UR5_target_handle, 
                _NewRot3D = (
                    np.pi/2, 
                    gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), 
                    np.pi/2
                )
            )
        m.engine.global_position_set(
            _ObjID = m.UR5_target_handle, 
            _NewPos3D = (tool_position[0], tool_position[1], tool_position[2])
        )
        m.engine.global_rotation_set(
            _ObjID = m.UR5_target_handle, 
            _NewRot3D = (np.pi/2, tool_rotation_angle, np.pi/2)
        )
        # # Simultaneously move and rotate gripper
        #for step_iter in range(max(num_move_steps, num_rotation_steps)):
        #    vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
        #    vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, gripper_orientation[1] + rotation_step*min(step_iter,num_rotation_steps), np.pi/2), vrep.simx_opmode_blocking)
        #vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(tool_position[0],tool_position[1],tool_position[2]),vrep.simx_opmode_blocking)
        #vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (np.pi/2, tool_rotation_angle, np.pi/2), vrep.simx_opmode_blocking)

        # Ensure gripper is open
        gripper.open_gripper(sim, m)

        # Approach grasp target
        mover.move_to(sim, m, position, None)

        # Get images before grasping
        color_img, depth_img = cam.get_camera_data(sim, m)
        depth_img = depth_img * m.cam_depth_scale # Apply depth scale from calibration

        # Get heightmaps beforew grasping
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, m.cam_intrinsics,
                                                                m.cam_pose, a.workspace_limits,
                                                                0.002)  # heightmap resolution from args
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

        # Close gripper to grasp target
        gripper_full_closed = gripper.close_gripper(sim, m)

        # Move gripper to location above grasp target
        mover.move_to(sim, m, location_above_grasp_target, None)

        # Check if grasp is successful
        gripper_full_closed = gripper.close_gripper(sim, m)
        grasp_success = not gripper_full_closed

        # Move the grasped object elsewhere
        if grasp_success:
            object_positions = np.asarray(obj.get_obj_positions(m))
            object_positions = object_positions[:,2]
            grasped_object_ind = np.argmax(object_positions)
            print('grasp obj z position', max(object_positions))
            grasped_object_handle = m.object_handles[grasped_object_ind]
            m.engine.global_position_set(_ObjID = grasped_object_handle, _NewPos3D = (-0.5, 0.5 + 0.05*float(grasped_object_ind), 0.1))
            #vrep.simxSetObjectPosition(self.sim_client,grasped_object_handle,-1,(-0.5, 0.5 + 0.05*float(grasped_object_ind), 0.1),vrep.simx_opmode_blocking)
            return grasp_success, color_img, depth_img, color_heightmap, valid_depth_heightmap, grasped_object_ind
        else:
            return grasp_success, None, None, None, None, None

