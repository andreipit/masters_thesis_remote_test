import json
from typing import Any
import io
import os
import numpy as np

from utils.arg.model import ArgsModel

class ParserJson():
    def __init__(self):
        pass

    @staticmethod
    def load_config(debug = False) -> dict:
        conf_file: io.TextIOWrapper = open('utils/config/config.json')
        conf: dict = json.load(conf_file)
        if debug:
            print('future_reward_discount=',type(conf["future_reward_discount"]))
            for x in conf:
                print(x, '=', conf[x])
            #[print(x, conf[x]) for x in conf]
        return conf

    @staticmethod
    def convert_dict_to_vars(c: dict) ->ArgsModel:
        """Just copy and convert: path to abs, list to np.array"""

        res: ArgsModel = ArgsModel()

        # --------------- General setup options ---------------
        res.obj_mesh_dir = os.path.abspath(c['obj_mesh_dir']) # Directory containing 3D mesh files (.obj) of objects to be added to simulation
        res.num_obj = c['num_obj'] # Number of objects to add to simulation
        res.heightmap_resolution = c['heightmap_resolution'] # Meters per pixel of heightmap
        res.random_seed = c['random_seed']
        res.force_cpu = c['cpu'] # force_cpu

        # --------------- Workspace setting -----------------
        res.workspace_limits = np.asarray(c['workspace_limits'], dtype = float) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        #res.workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]], dtype = float) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        #self.workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]], dtype = np.complex64) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
        #self.workspace_limits = np.asarray([[-0.224, 0.224], [-0.0001, 0.4]], dtype = float) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)

        # ------------- Training stage options -------------
        res.stage = c['stage']
        res.max_push_episode_length = c['max_push_episode_length']
        res.grasp_reward_threshold = c['grasp_reward_threshold']
        res.alternating_training = c['alternating_training']
        res.cooperative_training = c['cooperative_training']
    
        # -------------- Q-learning tricks ----------------
        res.future_reward_discount = c['future_reward_discount']
        res.experience_replay = c['experience_replay'] # Use prioritized experience replay?
        res.heuristic_bootstrap = c['heuristic_bootstrap'] # Use handcrafted grasping algorithm when grasping fails too many times in a row?
        res.explore_rate_decay = c['explore_rate_decay']

        # -------------- Testing options --------------
        res.is_testing = c['is_testing']
        res.max_test_trials = c['max_test_trials'] # Maximum number of test runs per case/scenario
        res.test_preset_cases = c['test_preset_cases']
        res.test_preset_file = os.path.abspath(c['test_preset_file']) if res.test_preset_cases else None
        res.random_scene_testing = c['random_scene_testing']  

        # ------ Pre-loading and logging options ------
        res.load_snapshot = c['load_snapshot'] # Load pre-trained snapshot of model?
        res.snapshot_file = os.path.abspath(c['snapshot_file']) if res.load_snapshot else None
        res.continue_logging = c['continue_logging'] # Continue logging from previous session
        res.logging_directory = os.path.abspath(c['logging_directory']) if res.continue_logging else os.path.abspath('logs')
        res.save_visualizations = c['save_visualizations'] # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True
    
        # ------- Goal-conditioned grasp net explore options ---------
        res.load_explore_snapshot = c['load_explore_snapshot'] # Load pre-trained snapshot of model?
        res.explore_snapshot_file = os.path.abspath(c['explore_snapshot_file']) if res.load_explore_snapshot else None

        # Set random seed
        #np.random.seed(res.random_seed)

        # ------- Goal-conditioned option ------------
        res.goal_conditioned = c['goal_conditioned']
        res.grasp_goal_conditioned = c['grasp_goal_conditioned']

        # moved here from hyperparameters
        res.grasp_explore = c['grasp_explore']
        res.goal_obj_idx = c['goal_obj_idx']
        res.tensor_logging_directory = c['tensor_logging_directory']

        return res


