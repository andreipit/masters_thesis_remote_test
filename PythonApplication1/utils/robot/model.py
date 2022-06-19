import os
import numpy as np
from utils.custom_types import NDArray
from utils.arg.model import ArgsModel
from simulator import Simulator
from typing import TextIO

class RobotModel():

    _workspace_limits: NDArray["3,2", float] = NDArray(shape=(3,2), dtype=float)# [[-7.24e-01 -2.76e-01] [-2.24e-01  2.24e-01] [-1.00e-04  4.00e-01]] = np.zeros((3, 3), dtype=np.complex64)
    workspace_limits = property(_workspace_limits.getx, _workspace_limits.setx, _workspace_limits.delx, "_workspace_limits")
    _color_space: NDArray["10,3", float] = NDArray(shape=(10,3), dtype=float)
    color_space = property(_color_space.getx, _color_space.setx, _color_space.delx, "_color_space")
    num_obj: int = 0 # 5
    stage: str = '' # grasp_only
    goal_conditioned: bool = False # True
    grasp_goal_conditioned: bool = False # False
    obj_mesh_dir: str = '' # C:\Users\user\Desktop\notes\mipt4\repos\xukechun\Recreate_v2\PythonApplication1\PythonApplication1\objects\blocks
    mesh_list: list # ['0.obj', '1.obj', '2.obj', '3.obj', '4.obj', '6.obj', '7.obj', '8.obj']
    _obj_mesh_ind: NDArray["5", float] = NDArray(shape=(5), dtype=int) # numpy.ndarray [7 3 6 5 4]
    obj_mesh_ind = property(_obj_mesh_ind.getx, _obj_mesh_ind.setx, _obj_mesh_ind.delx, "_obj_mesh_ind")
    _obj_mesh_color: NDArray["5,3", float] = NDArray(shape=(5,3), dtype=float) # [[4.67296746e-307 1.69121096e-306 9.34609111e-307],[],[],[],[]]
    obj_mesh_color = property(_obj_mesh_color.getx, _obj_mesh_color.setx, _obj_mesh_color.delx, "_obj_mesh_color")
    is_testing: bool = False # False
    test_preset_cases: bool = False # False
    test_preset_file: None = None # None => type=<class 'NoneType'>; value=None
    
    # created later at robot_objects.py:
    test_obj_mesh_files: list = []
    test_obj_mesh_colors: list = []
    test_obj_positions: list = []
    test_obj_orientations: list = []
    object_handles: list = []

    # created later at robot_sim.py
    engine:Simulator = None
    UR5_target_handle: int = 0 # 84
    RG2_tip_handle: int = 0 # 83
    cam_handle: int = 0 # 85
    _cam_pose: NDArray["4,4", float] = NDArray(shape=(4,4), dtype=float) #[[ 1.37678730e-07 -7.07106555e-01  7.07107007e-01 -1.00000000e+00][][][]]
    cam_pose = property(_cam_pose.getx, _cam_pose.setx, _cam_pose.delx, "")
    _cam_intrinsics: NDArray["3,3", float] = NDArray(shape=(3,3), dtype=float) #[[618.62   0.   320.  ]# [  0.   618.62 240.  ]# [  0.     0.     1.  ]]
    cam_intrinsics = property(_cam_intrinsics.getx, _cam_intrinsics.setx, _cam_intrinsics.delx, "")
    _bg_color_img: NDArray["480,640,3", np.uint8] = NDArray(shape=(480, 640, 3), dtype=np.uint8) # [ [[ 80   1 221] [ 49 124   1][  0   0 144] ... [ 18 101   5][106  18 101][  5 160  19]]...[...]...[]..]
    bg_color_img = property(_bg_color_img.getx, _bg_color_img.setx, _bg_color_img.delx, "")
    _bg_depth_img: NDArray["480,640", float] = NDArray(shape=(480, 640), dtype=float) #[ [0. 0. 0. ... 0. 0. 0.] [] ... [] ]
    bg_depth_img = property(_bg_depth_img.getx, _bg_depth_img.setx, _bg_depth_img.delx, "") # [[1.15526764 1.15526764 1.15526764 ... 1.15527561 1.15527561 1.15528357] ... ]
    cam_depth_scale: int = 0 # 1

    def __init__(self):
        pass

    def copy_args_create_consts(self, a: ArgsModel):
        self.workspace_limits = a.workspace_limits
        self.num_obj = a.num_obj
        self.stage = a.stage
        self.goal_conditioned = a.goal_conditioned
        self.grasp_goal_conditioned = a.grasp_goal_conditioned

        # Define colors for object meshes (Tableau palette)
        self.color_space = np.asarray([[78.0, 121.0, 167.0], # blue
                                        [89.0, 161.0, 79.0], # green
                                        [156, 117, 95], # brown
                                        [242, 142, 43], # orange
                                        [237.0, 201.0, 72.0], # yellow
                                        [186, 176, 172], # gray
                                        [255.0, 87.0, 89.0], # red
                                        [176, 122, 161], # purpl                                           
                                        [118, 183, 178], # cyan
                                        [255, 157, 167]])/255.0 #pink
        # Read files in object mesh directory
        self.obj_mesh_dir = a.obj_mesh_dir
        self.mesh_list = os.listdir(self.obj_mesh_dir)

        # Randomly choose objects to add to scene
        # self.obj_mesh_ind = 5 * np.ones(len(self.mesh_list), dtype=np.int32)
        self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
        
        self.obj_mesh_color = self.color_space[np.asarray(range(self.num_obj)) % 10, :]

        # moved them here from simulation file:
        self.is_testing = a.is_testing
        self.test_preset_cases = a.test_preset_cases
        self.test_preset_file = a.test_preset_file


    def debug_types_values(self):
        # parameters
        self.print_type_value(self.num_obj, 'num_obj')
        self.print_type_value(self.stage, 'stage')
        self.print_type_value(self.goal_conditioned, 'goal_conditioned')
        self.print_type_value(self.grasp_goal_conditioned, 'grasp_goal_conditioned')
        self.print_type_value(self.obj_mesh_dir, 'obj_mesh_dir')
        self.print_type_value(self.mesh_list, 'mesh_list')
        self.print_type_value(self.obj_mesh_ind, 'obj_mesh_ind')
        self.print_type_value(self.obj_mesh_color, 'obj_mesh_color')
        self.print_type_value(self.is_testing, 'is_testing')
        self.print_type_value(self.test_preset_cases, 'test_preset_cases')
        self.print_type_value(self.test_preset_file, 'test_preset_file')

        # later robot_objects
        self.print_type_value(self.test_obj_mesh_files, 'test_obj_mesh_files')
        self.print_type_value(self.test_obj_mesh_colors, 'test_obj_mesh_colors')
        self.print_type_value(self.test_obj_positions, 'test_obj_positions')
        self.print_type_value(self.test_obj_orientations, 'test_obj_orientations')
        self.print_type_value(self.object_handles, 'object_handles')
        self.print_type_value(self.engine, 'engine')
        self.print_type_value(self.UR5_target_handle, 'UR5_target_handle')
        self.print_type_value(self.RG2_tip_handle, 'RG2_tip_handle')
        self.print_type_value(self.cam_handle, 'cam_handle')
        self.print_type_value(self.cam_pose, 'cam_pose')
        self.print_type_value(self.cam_intrinsics, 'cam_intrinsics')
        self.print_type_value(self.bg_color_img, 'bg_color_img')
        self.print_type_value(self.bg_depth_img, 'bg_depth_img')
        self.print_type_value(self.cam_depth_scale, 'cam_depth_scale')
        
        # later sim:
        self.print_type_value(self.engine, 'engine')
        self.print_type_value(self.UR5_target_handle, 'UR5_target_handle')
        self.print_type_value(self.RG2_tip_handle, 'RG2_tip_handle')
        self.print_type_value(self.cam_handle, 'cam_handle')
        self.print_type_value(self.cam_pose, 'cam_pose')
        self.print_type_value(self.cam_intrinsics, 'cam_intrinsics')
        self.print_type_value(self.bg_color_img, 'bg_color_img')
        self.print_type_value(self.bg_depth_img, 'bg_depth_img')
        self.print_type_value(self.cam_depth_scale, 'cam_depth_scale')

    def print_type_value(self, _X, _Name):
        if type(_X) is np.ndarray or type(_X) is NDArray:
            print('name=', _Name, '; shape=', _X.shape, '; dtype=', _X.dtype)
        print('name=', _Name, '; type=', type(_X), '; value=', _X, sep='')
