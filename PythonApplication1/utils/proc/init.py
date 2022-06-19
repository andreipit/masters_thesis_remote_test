import numpy as np
import os
import time
import datetime

from tensorboardX import SummaryWriter

from utils.arg.model import ArgsModel
from utils.proc.model import ProcModel
from utils.trainer.model import TrainerModel

class ProcInit():
    
    def __init__(self):
        pass

    def copy_args_create_nonlocal(self, a: ArgsModel, m:ProcModel, trainer_m: TrainerModel):
        # Initialize episode loss
        m.episode_loss = 0

        # Initialize variables for heuristic bootstrapping and exploration probability
        # The bootstrap method is a resampling technique used to estimate statistics on a population by sampling a dataset with replacement.
        m.no_change_count = [2, 2] if not a.is_testing else [0, 0]
        m.explore_prob = 0.5 if not a.is_testing else 0.0
        m.grasp_explore_prob = 0.8 if not a.is_testing else 0.0
        m.grasp_explore = a.grasp_explore #was args.grasp_explore, but I've copied it to a

        # Quick hack for nonlocal memory between threads in Python 2
        m.nonlocal_variables = {'executing_action' : False,
                              'primitive_action' : None,
                              'best_pix_ind' : None,
                              'push_success' : False,
                              'grasp_success' : False,
                              'grasp_reward' : 0,
                              'improved_grasp_reward' : 0,
                              'push_step' : 0, # plus one after pushing
                              'goal_obj_idx' : 0,
                              'goal_catched' : 0,
                              'border_occupy_ratio' : 1,
                              'decreased_occupy_ratio' : 0,
                              'restart_scene' : 0,
                              'episode' : 0, # episode number
                              'new_episode_flag' : 0, # flag to begin a new episode
                              'episode_grasp_reward' : 0, # grasp reward at the end of a episode
                              'episode_ratio_of_grasp_to_push' : 0, # ratio of grasp to push at the end of a episode
                              'episode_improved_grasp_reward' : 0,
                              'push_predictions': np.zeros((16, 224, 224), dtype=float),
                              'grasp_predictions' : np.zeros((16,224,224),dtype=float)} # average of improved grasp reward of a episode

        
        # --------- Initialize nonlocal variables -----------
        m.nonlocal_variables['goal_obj_idx'] = a.goal_obj_idx # args.goal_obj_idx

        if a.continue_logging:
            if not a.is_testing:
                m.nonlocal_variables['episode'] = trainer_m.episode_log[len(trainer_m.episode_log) - 1][0]
            if a.stage == 'push_only':
                # Initialize nonlocal memory
                m.nonlocal_variables['push_step'] = trainer_m.push_step_log[trainer_m.iteration - 1][0]
                m.nonlocal_variables['episode_improved_grasp_reward'] = trainer_m.episode_improved_grasp_reward_log[len(trainer_m.episode_improved_grasp_reward_log) - 1][0]

        # ------ Tensorboard setting --------
        m.timestamp = time.time()
        m.timestamp_value = datetime.datetime.fromtimestamp(m.timestamp)
        m.tensor_logging_directory = a.tensor_logging_directory
        if a.continue_logging:
            m.writer = SummaryWriter(os.path.join(a.tensor_logging_directory, a.logging_directory.split('/')[-1]))
        else:
            # writer = SummaryWriter(os.path.join(tensor_logging_directory, timestamp_value.strftime('%Y-%m-%d.%H:%M:%S')))
            m.writer = SummaryWriter(os.path.join(a.tensor_logging_directory, m.timestamp_value.strftime('%Y-%m-%d.%H_%M_%S')))





