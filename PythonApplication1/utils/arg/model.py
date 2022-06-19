#from __future__ import annotations

import os
import numpy as np
import argparse

from utils.custom_types import NDArray


class ArgsModel():

    # all comments are given for the case 1.2) --stage grasp_only --num_obj 5 --goal_conditioned --goal_obj_idx 4 --experience_replay --explore_rate_decay --save_visualizations
    
    # --------------- General setup options ---------------
    obj_mesh_dir: str = '' # C:\Users\user\Desktop\notes\mipt4\repos\xukechun\Recreate_v2\PythonApplication1\PythonApplication1\objects\blocks
    num_obj: int = 0 # 5
    heightmap_resolution: float = .0 # 0.002
    random_seed: int = 0 # 1234
    force_cpu: bool = False # False

    ## --------------- Workspace setting -----------------
    _workspace_limits: NDArray["3,2", float] = NDArray(shape=(3,2), dtype=float)# [[-7.24e-01 -2.76e-01] [-2.24e-01  2.24e-01] [-1.00e-04  4.00e-01]] = np.zeros((3, 3), dtype=np.complex64)
    workspace_limits = property(_workspace_limits.getx, _workspace_limits.setx, _workspace_limits.delx, "_workspace_limits")

    # ------------- Training stage options -------------
    stage: str = '' # grasp_only
    max_push_episode_length: int = 0 # 5
    grasp_reward_threshold: float = .0 # 1.8
    alternating_training: bool = False
    cooperative_training: bool = False
    
    # -------------- Q-learning tricks ----------------
    future_reward_discount: float = .0 # 0.5
    experience_replay: bool = False # True
    heuristic_bootstrap: bool = False # False
    explore_rate_decay: bool = False # True

    # -------------- Testing options --------------
    is_testing: bool = False # False
    max_test_trials: int = 0 # 30
    test_preset_cases: bool = False # False
    test_preset_file: None = None # None => type=<class 'NoneType'>; value=None
    random_scene_testing: bool = False # False

    # ------ Pre-loading and logging options ------
    load_snapshot: bool = False # False
    snapshot_file: None = None # None => type=<class 'NoneType'>; value=None
    continue_logging: bool = False # False
    logging_directory: str = '' # C:\Users\user\Desktop\notes\mipt4\repos\xukechun\Recreate_v2\PythonApplication1\PythonApplication1\logs
    save_visualizations: bool = False # True
    
    # ------- Goal-conditioned grasp net explore options ---------
    load_explore_snapshot: bool = False # False
    explore_snapshot_file: None = None # None => type=<class 'NoneType'>; value=None

    # ------- Goal-conditioned option ------------
    goal_conditioned: bool = False # True
    grasp_goal_conditioned: bool = False # False

    # moved here from hyperparameters
    grasp_explore: bool = False # False
    goal_obj_idx: int = 0 # 4
    tensor_logging_directory: str = '' # ./tensorlog


    def __init__(self):
        pass
