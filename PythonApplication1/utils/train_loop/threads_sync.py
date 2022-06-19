# MARLI P
from utils.train_loop.model import MainloopModel
from utils.arg.model import ArgsModel

from robot import Robot
from logger import Logger
from trainer import Trainer
from proc import Proc

import numpy as np
import time

class MainloopThreadsSync(object):
    def save(self, 
        m: MainloopModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc,
    ): 
        # Save information for next training step
        prev_color_img = m.color_img.copy() # used only for return
        prev_depth_img = m.depth_img.copy() # useless var
        m.prev_color_heightmap = p.m.color_heightmap.copy()
        m.prev_depth_heightmap = p.m.depth_heightmap.copy()
        m.prev_valid_depth_heightmap = p.m.valid_depth_heightmap.copy()
        
        # useless var:
        m.prev_push_success = p.m.nonlocal_variables['push_success'] # usesell var
        
        m.prev_grasp_success = p.m.nonlocal_variables['grasp_success']
        m.prev_primitive_action = p.m.nonlocal_variables['primitive_action']
        
        # useless var:
        m.prev_push_predictions = m.push_predictions.copy() # useless var
        # useless var:
        m.prev_grasp_predictions = m.grasp_predictions.copy() # useless var
        
        m.prev_best_pix_ind = p.m.nonlocal_variables['best_pix_ind']
        m.prev_grasp_reward = p.m.nonlocal_variables['grasp_reward']
        if a.grasp_goal_conditioned or a.goal_conditioned:
            m.prev_goal_mask_heightmap = p.m.goal_mask_heightmap.copy()
        if a.stage == 'push_only':
            m.prev_improved_grasp_reward = p.m.nonlocal_variables['improved_grasp_reward']
            m.prev_grasp_reward = p.m.nonlocal_variables['grasp_reward']
        else:
            m.prev_improved_grasp_reward = 0.0

        t.m.iteration += 1
        iteration_time_1 = time.time() # used only here and on the next line
        print('Time elapsed: %f' % (iteration_time_1-m.iteration_time_0))
        return prev_color_img
