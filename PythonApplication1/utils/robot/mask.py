import time
import numpy as np

import utils.utils as utils
from utils.arg.model import ArgsModel
from utils.robot.model import RobotModel
from simulator import Simulator
from utils.custom_types import NDArray

import numpy as np
import os
import trimesh

# 3rd party
from utils.robot.model import RobotModel
from utils.robot.sim import RobotSim
import utils.utils as utils

class RobotMask():

    def get_obj_mask(self, obj_ind: int, sim: RobotSim, m: RobotModel) -> trimesh.caching.TrackedArray: # 2D array with different len # obj_ind=4 at start 1.2
        debugMode = False
        if (debugMode): print('obj_ind------------------=',obj_ind)

        if obj_ind >= len(m.object_handles):
            print('Error: index =', obj_ind, ' list len = ', len(m.object_handles))

        # Get object pose in simulation
        sim_ret, obj_position = m.engine.global_position_get(_ObjID = m.object_handles[obj_ind])
        sim_ret, obj_orientation = m.engine.global_rotation_get(_ObjID = m.object_handles[obj_ind])
        if (debugMode): print('obj_position=',obj_position)
        if (debugMode): print('obj_orientation=',obj_orientation)

        # Convert them to matrices (create linear transformation 'obj_pose')
        obj_trans = np.eye(4,4)
        obj_trans[0:3,3] = np.asarray(obj_position)
        obj_orientation = [obj_orientation[0], obj_orientation[1], obj_orientation[2]]

        obj_rotm = np.eye(4,4)
        obj_rotm[0:3,0:3] = utils.obj_euler2rotm(obj_orientation)
        obj_pose = np.dot(obj_trans, obj_rotm) # Compute rigid transformation representating camera pose

        # load .obj files
        obj_mesh_file = os.path.join(m.obj_mesh_dir, m.mesh_list[m.obj_mesh_ind[obj_ind]]) # 1.obj
        mesh = trimesh.load_mesh(obj_mesh_file)
        if (debugMode): print('obj_mesh_file=',obj_mesh_file, mesh)

        # transform the mesh to world frame
        if obj_mesh_file.split('/')[-1] == '2.obj' or obj_mesh_file.split('/')[-1] == '6.obj':
            mesh.apply_transform(obj_pose)
        else:
            # rest
            transformation = np.array([[0,0,1,0],[0,-1,0,0],[1,0,0,0],[0,0,0,1]])
            mesh.apply_transform(transformation)
            mesh.apply_transform(obj_pose)

        obj_contour = mesh.vertices[:, 0:2]
        if (debugMode): print('obj_contour.shape',obj_contour.shape)
        return obj_contour

    def get_obj_masks(self, sim: RobotSim, m: RobotModel) -> list:
        # from scipy.spatial.transform import Rotation as R
        obj_contours = []
        obj_number = len(m.obj_mesh_ind)
        # scene = trimesh.Scene()
        for object_idx in range(obj_number):
            # Get object pose in simulation
            sim_ret, obj_position = m.engine.global_position_get(_ObjID = m.object_handles[object_idx])
            sim_ret, obj_orientation = m.engine.global_rotation_get(_ObjID = m.object_handles[object_idx])
            #sim_ret, obj_position = vrep.simxGetObjectPosition(self.sim_client, self.object_handles[object_idx], -1, vrep.simx_opmode_blocking)
            #sim_ret, obj_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.object_handles[object_idx], -1, vrep.simx_opmode_blocking)
            
            # convert pos/rot to linear transformation
            obj_trans = np.eye(4,4)
            obj_trans[0:3,3] = np.asarray(obj_position)
            obj_orientation = [obj_orientation[0], obj_orientation[1], obj_orientation[2]]

            obj_rotm = np.eye(4,4)
            obj_rotm[0:3,0:3] = utils.obj_euler2rotm(obj_orientation)
            obj_pose = np.dot(obj_trans, obj_rotm) # Compute rigid transformation representating camera pose
            # load .obj files
            obj_mesh_file = os.path.join(m.obj_mesh_dir, m.mesh_list[m.obj_mesh_ind[object_idx]])
            # print(obj_mesh_file)

            mesh = trimesh.load_mesh(obj_mesh_file)

            if obj_mesh_file.split('/')[-1] == '2.obj' or obj_mesh_file.split('/')[-1] == '6.obj':
                mesh.apply_transform(obj_pose)
            else:
                # rest
                transformation = np.array([[0,0,1,0],[0,-1,0,0],[1,0,0,0],[0,0,0,1]])
                mesh.apply_transform(transformation)
                mesh.apply_transform(obj_pose)

            # scene.add_geometry(mesh)
            obj_contours.append(mesh.vertices[:, 0:2])
        # scene.show()
        return obj_contours        


    def get_test_obj_mask(self, obj_ind: int, sim: RobotSim, m: RobotModel):
        """
        1) by obj_ind: find obj in scene, get pos/rot
        convert pos/rot to linear transformaion
        2) by obj_ind: find asset.obj from which scene obj was created
        apply linear tran-n (align mesh with obj)
        3) return vertices (called 'contour')
        """

        # Get obj pos and rot: Get object pose in simulation (earlier we've instantiated them in robot_objects and saved into object_handles)
        sim_ret, obj_position = m.engine.global_position_get(_ObjID = m.object_handles[obj_ind])
        sim_ret, obj_orientation = m.engine.global_rotation_get(_ObjID = m.object_handles[obj_ind])

        # Convert them to matrices (create linear transformation 'obj_pose')
        obj_trans = np.eye(4,4)
        obj_trans[0:3,3] = np.asarray(obj_position)
        obj_orientation = [obj_orientation[0], obj_orientation[1], obj_orientation[2]]

        obj_rotm = np.eye(4,4)
        obj_rotm[0:3,0:3] = utils.obj_euler2rotm(obj_orientation)
        obj_pose = np.dot(obj_trans, obj_rotm) # Compute rigid transformation representating camera pose

        # load .obj files
        print('mask.py', m.test_obj_mesh_files)
        
        obj_mesh_file = m.test_obj_mesh_files[obj_ind] # get obj file path
        mesh = trimesh.load_mesh(obj_mesh_file) # trimesh obj to mesh

        # transform the mesh to world frame
        if obj_mesh_file.split('/')[-1] == '2.obj' or obj_mesh_file.split('/')[-1] == '6.obj':
            mesh.apply_transform(obj_pose)
        else:
            # rest
            transformation = np.array([[0,0,1,0],[0,-1,0,0],[1,0,0,0],[0,0,0,1]])
            mesh.apply_transform(transformation)
            mesh.apply_transform(obj_pose) # put loaded mesh to object pos and rot

        obj_contour = mesh.vertices[:, 0:2] # contour is just vertices

        return obj_contour
    
    def get_test_obj_masks(self, sim: RobotSim, m: RobotModel):
        """
        make get_test_obj_mask for all cubes (ie return contours)
        """
        obj_contours = []
        obj_number = len(m.test_obj_mesh_files)
        for object_idx in range(obj_number):
            # Get object pose in simulation
            sim_ret, obj_position = m.engine.global_position_get(_ObjID = m.object_handles[object_idx])
            sim_ret, obj_orientation = m.engine.global_rotation_get(_ObjID = m.object_handles[object_idx])
            #sim_ret, obj_position = vrep.simxGetObjectPosition(self.sim_client, self.object_handles[object_idx], -1, vrep.simx_opmode_blocking)
            #sim_ret, obj_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.object_handles[object_idx], -1, vrep.simx_opmode_blocking)
            
            # convert it to transformation matrix
            obj_trans = np.eye(4,4)
            obj_trans[0:3,3] = np.asarray(obj_position)
            obj_orientation = [obj_orientation[0], obj_orientation[1], obj_orientation[2]]

            obj_rotm = np.eye(4,4)
            obj_rotm[0:3,0:3] = utils.obj_euler2rotm(obj_orientation)
            obj_pose = np.dot(obj_trans, obj_rotm) # Compute rigid transformation representating camera pose
            # load .obj files
            obj_mesh_file = m.test_obj_mesh_files[object_idx]
            # print(obj_mesh_file)

            mesh = trimesh.load_mesh(obj_mesh_file)

            if obj_mesh_file.split('/')[-1] == '2.obj' or obj_mesh_file.split('/')[-1] == '6.obj':
                mesh.apply_transform(obj_pose)
            else:
                # rest
                transformation = np.array([[0,0,1,0],[0,-1,0,0],[1,0,0,0],[0,0,0,1]])
                mesh.apply_transform(transformation)
                mesh.apply_transform(obj_pose)

            obj_contours.append(mesh.vertices[:, 0:2])
        return obj_contours        
