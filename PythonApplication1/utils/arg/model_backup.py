##from __future__ import annotations

#import os
#import numpy as np
#import argparse


#from utils.custom_types import NDArray

#class ArgsModel():

#    # all comments are given for the case 1.2) --stage grasp_only --num_obj 5 --goal_conditioned --goal_obj_idx 4 --experience_replay --explore_rate_decay --save_visualizations
    
#    # --------------- General setup options ---------------
#    obj_mesh_dir: str = '' # C:\Users\user\Desktop\notes\mipt4\repos\xukechun\Recreate_v2\PythonApplication1\PythonApplication1\objects\blocks
#    num_obj: int = 0 # 5
#    heightmap_resolution: float = .0 # 0.002
#    random_seed: int = 0 # 1234
#    force_cpu: bool = False # False

#    ## --------------- Workspace setting -----------------
#    _workspace_limits: NDArray["3,2", float] = NDArray(shape=(3,2), dtype=float)# [[-7.24e-01 -2.76e-01] [-2.24e-01  2.24e-01] [-1.00e-04  4.00e-01]] = np.zeros((3, 3), dtype=np.complex64)
#    workspace_limits = property(_workspace_limits.getx, _workspace_limits.setx, _workspace_limits.delx, "_workspace_limits")


#    # ------------- Training stage options -------------
#    stage: str = '' # grasp_only
#    max_push_episode_length: int = 0 # 5
#    grasp_reward_threshold: float = .0 # 1.8
#    alternating_training: bool = False
#    cooperative_training: bool = False
    
#    # -------------- Q-learning tricks ----------------
#    future_reward_discount: float = .0 # 0.5
#    experience_replay: bool = False # True
#    heuristic_bootstrap: bool = False # False
#    explore_rate_decay: bool = False # True

#    # -------------- Testing options --------------
#    is_testing: bool = False # False
#    max_test_trials: int = 0 # 30
#    test_preset_cases: bool = False # False
#    test_preset_file: None = None # None => type=<class 'NoneType'>; value=None
#    random_scene_testing: bool = False # False

#    # ------ Pre-loading and logging options ------
#    load_snapshot: bool = False # False
#    snapshot_file: None = None # None => type=<class 'NoneType'>; value=None
#    continue_logging: bool = False # False
#    logging_directory: str = '' # C:\Users\user\Desktop\notes\mipt4\repos\xukechun\Recreate_v2\PythonApplication1\PythonApplication1\logs
#    save_visualizations: bool = False # True
    
#    # ------- Goal-conditioned grasp net explore options ---------
#    load_explore_snapshot: bool = False # False
#    explore_snapshot_file: None = None # None => type=<class 'NoneType'>; value=None

#    # ------- Goal-conditioned option ------------
#    goal_conditioned: bool = False # True
#    grasp_goal_conditioned: bool = False # False

#    # moved here from hyperparameters
#    grasp_explore: bool = False # False
#    goal_obj_idx: int = 0 # 4
#    tensor_logging_directory: str = '' # ./tensorlog

#    # --------------- Workspace setting -----------------


#    def __init__(self):
#        pass

#    def convert_args_to_vars(self, args: argparse.Namespace):
#        # --------------- General setup options ---------------
#        self.obj_mesh_dir = os.path.abspath(args.obj_mesh_dir) # Directory containing 3D mesh files (.obj) of objects to be added to simulation
#        self.num_obj = args.num_obj # Number of objects to add to simulation
#        self.heightmap_resolution = args.heightmap_resolution # Meters per pixel of heightmap
#        self.random_seed = args.random_seed
#        self.force_cpu = args.force_cpu

#        # --------------- Workspace setting -----------------
#        self.workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]], dtype = float) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
#        #self.workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]], dtype = np.complex64) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
#        #self.workspace_limits = np.asarray([[-0.224, 0.224], [-0.0001, 0.4]], dtype = float) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)

#        # ------------- Training stage options -------------
#        self.stage = args.stage
#        self.max_push_episode_length = args.max_push_episode_length
#        self.grasp_reward_threshold = args.grasp_reward_threshold
#        self.alternating_training = args.alternating_training
#        self.cooperative_training = args.cooperative_training
    
#        # -------------- Q-learning tricks ----------------
#        self.future_reward_discount = args.future_reward_discount
#        self.experience_replay = args.experience_replay # Use prioritized experience replay?
#        self.heuristic_bootstrap = args.heuristic_bootstrap # Use handcrafted grasping algorithm when grasping fails too many times in a row?
#        self.explore_rate_decay = args.explore_rate_decay

#        # -------------- Testing options --------------
#        self.is_testing = args.is_testing
#        self.max_test_trials = args.max_test_trials # Maximum number of test runs per case/scenario
#        self.test_preset_cases = args.test_preset_cases
#        self.test_preset_file = os.path.abspath(args.test_preset_file) if self.test_preset_cases else None
#        self.random_scene_testing = args.random_scene_testing  

#        # ------ Pre-loading and logging options ------
#        self.load_snapshot = args.load_snapshot # Load pre-trained snapshot of model?
#        self.snapshot_file = os.path.abspath(args.snapshot_file) if self.load_snapshot else None
#        self.continue_logging = args.continue_logging # Continue logging from previous session
#        self.logging_directory = os.path.abspath(args.logging_directory) if self.continue_logging else os.path.abspath('logs')
#        self.save_visualizations = args.save_visualizations # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True
    
#        # ------- Goal-conditioned grasp net explore options ---------
#        self.load_explore_snapshot = args.load_explore_snapshot # Load pre-trained snapshot of model?
#        self.explore_snapshot_file = os.path.abspath(args.explore_snapshot_file) if self.load_explore_snapshot else None

#        # Set random seed
#        np.random.seed(self.random_seed)

#        # ------- Goal-conditioned option ------------
#        self.goal_conditioned = args.goal_conditioned
#        self.grasp_goal_conditioned = args.grasp_goal_conditioned

#        # moved here from hyperparameters
#        self.grasp_explore = args.grasp_explore
#        self.goal_obj_idx = args.goal_obj_idx
#        self.tensor_logging_directory = args.tensor_logging_directory


#    #def check_types_values(self):
#    #    self.print_type_value(self.obj_mesh_dir, 'obj_mesh_dir')
#    #    self.print_type_value(self.num_obj, 'num_obj')
#    #    self.print_type_value(self.heightmap_resolution, 'heightmap_resolution')
#    #    self.print_type_value(self.random_seed, 'random_seed')
#    #    self.print_type_value(self.force_cpu, 'force_cpu')
#    #    self.print_type_value(self.workspace_limits, 'workspace_limits')
#    #    self.print_type_value(self.stage, 'stage')
#    #    self.print_type_value(self.max_push_episode_length, 'max_push_episode_length')
#    #    self.print_type_value(self.grasp_reward_threshold, 'grasp_reward_threshold')
#    #    self.print_type_value(self.alternating_training, 'alternating_training')
#    #    self.print_type_value(self.cooperative_training, 'cooperative_training')
#    #    self.print_type_value(self.future_reward_discount, 'future_reward_discount')
#    #    self.print_type_value(self.experience_replay, 'experience_replay')
#    #    self.print_type_value(self.heuristic_bootstrap, 'heuristic_bootstrap')
#    #    self.print_type_value(self.explore_rate_decay, 'explore_rate_decay')
#    #    self.print_type_value(self.is_testing, 'is_testing')
#    #    self.print_type_value(self.max_test_trials, 'max_test_trials')
#    #    self.print_type_value(self.test_preset_cases, 'test_preset_cases')
#    #    self.print_type_value(self.test_preset_file, 'test_preset_file')
#    #    self.print_type_value(self.random_scene_testing, 'random_scene_testing')
#    #    self.print_type_value(self.load_snapshot, 'load_snapshot')
#    #    self.print_type_value(self.snapshot_file, 'snapshot_file')
#    #    self.print_type_value(self.continue_logging, 'continue_logging')
#    #    self.print_type_value(self.logging_directory, 'logging_directory')
#    #    self.print_type_value(self.save_visualizations, 'save_visualizations')
#    #    self.print_type_value(self.load_explore_snapshot, 'load_explore_snapshot')
#    #    self.print_type_value(self.explore_snapshot_file, 'explore_snapshot_file')
#    #    self.print_type_value(self.goal_conditioned, 'goal_conditioned')
#    #    self.print_type_value(self.grasp_goal_conditioned, 'grasp_goal_conditioned')
#    #    self.print_type_value(self.grasp_explore, 'grasp_explore')
#    #    self.print_type_value(self.goal_obj_idx, 'goal_obj_idx')
#    #    self.print_type_value(self.tensor_logging_directory, 'tensor_logging_directory')

#    #def print_type_value(self, _X, _Name):
#    #    print('name=', _Name, '; type=', type(_X), '; value=', _X, sep='')

## Example of 1.2:
##--stage grasp_only 
##--num_obj 5 
##--goal_conditioned 
##--goal_obj_idx 4 
##--experience_replay 
##--explore_rate_decay 
##--save_visualizations

## for 1.2 case:
##name=obj_mesh_dir; type=<class 'str'>; value=C:\Users\user\Desktop\notes\mipt4\repos\xukechun\Recreate_v2\PythonApplication1\PythonApplication1\objects\blocks
##name=num_obj; type=<class 'int'>; value=5
##name=heightmap_resolution; type=<class 'float'>; value=0.002
##name=random_seed; type=<class 'int'>; value=1234
##name=force_cpu; type=<class 'bool'>; value=False
##name=workspace_limits; type=<class 'numpy.ndarray'>; value=[[-7.24e-01 -2.76e-01]
##[-2.24e-01  2.24e-01]
##[-1.00e-04  4.00e-01]]
##name=stage; type=<class 'str'>; value=grasp_only
##name=max_push_episode_length; type=<class 'int'>; value=5
##name=grasp_reward_threshold; type=<class 'float'>; value=1.8
##name=alternating_training; type=<class 'bool'>; value=False
##name=cooperative_training; type=<class 'bool'>; value=False
##name=future_reward_discount; type=<class 'float'>; value=0.5
##name=experience_replay; type=<class 'bool'>; value=True
##name=heuristic_bootstrap; type=<class 'bool'>; value=False
##name=explore_rate_decay; type=<class 'bool'>; value=True
##name=is_testing; type=<class 'bool'>; value=False
##name=max_test_trials; type=<class 'int'>; value=30
##name=test_preset_cases; type=<class 'bool'>; value=False
##name=test_preset_file; type=<class 'NoneType'>; value=None
##name=random_scene_testing; type=<class 'bool'>; value=False
##name=load_snapshot; type=<class 'bool'>; value=False
##name=snapshot_file; type=<class 'NoneType'>; value=None
##name=continue_logging; type=<class 'bool'>; value=False
##name=logging_directory; type=<class 'str'>; value=C:\Users\user\Desktop\notes\mipt4\repos\xukechun\Recreate_v2\PythonApplication1\PythonApplication1\logs
##name=save_visualizations; type=<class 'bool'>; value=True
##name=load_explore_snapshot; type=<class 'bool'>; value=False
##name=explore_snapshot_file; type=<class 'NoneType'>; value=None
##name=goal_conditioned; type=<class 'bool'>; value=True
##name=grasp_goal_conditioned; type=<class 'bool'>; value=False
##name=grasp_explore; type=<class 'bool'>; value=False
##name=goal_obj_idx; type=<class 'int'>; value=4
##name=tensor_logging_directory; type=<class 'str'>; value=./tensorlog