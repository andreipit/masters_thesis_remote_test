# MARLI P
from utils.train_loop.model import MainloopModel
from utils.arg.model import ArgsModel

from robot import Robot
from logger import Logger
from trainer import Trainer
from proc import Proc

import numpy as np
import os
import cv2
import utils.utils as utils
import torch

class MainloopExpReplay(object):
    def run(self,
        m: MainloopModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc,
    ): 
        m.sample_primitive_action = m.prev_primitive_action
        if a.grasp_goal_conditioned:
            m.sample_goal_obj_idx = p.m.nonlocal_variables['goal_obj_idx']
            print('sample_goal_obj_idx', m.sample_goal_obj_idx)
        if m.sample_primitive_action == 'push':
            m.sample_primitive_action_id = 0
            m.sample_reward_value = 0 if m.prev_reward_value == 0.5 else 0.5
        elif m.sample_primitive_action == 'grasp':
            m.sample_primitive_action_id = 1
            m.sample_reward_value = 0 if m.prev_reward_value == 1 else 1


    def get_samples(self,
        m: MainloopModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc,
    ): 
        # Get samples of the same primitive but with different results
        if not a.grasp_goal_conditioned or m.sample_primitive_action == 'push':
            m.sample_ind = np.argwhere(np.logical_and(np.asarray(t.m.reward_value_log)[0:t.m.iteration,0] == m.sample_reward_value, np.asarray(t.m.executed_action_log)[0:t.m.iteration,0] == m.sample_primitive_action_id))
        else:
            m.sample_ind = np.argwhere(np.logical_and(np.asarray(t.m.reward_value_log)[0:t.m.iteration,0] == m.sample_reward_value, 
            np.asarray(t.m.grasp_obj_log)[0:t.m.iteration,0] == m.sample_goal_obj_idx))
                   

    def get_highest_sample(self,
        m: MainloopModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc,
    ): 
        print('reward_value_log:', np.asarray(t.m.reward_value_log)[m.sample_ind[:,0], 0])
        # Find sample with highest surprise value
        # sample_surprise_values used only in current fun
        sample_surprise_values = np.abs(np.asarray(t.m.predicted_value_log)[m.sample_ind[:,0]] - np.asarray(t.m.label_value_log)[m.sample_ind[:,0]])
        sorted_surprise_ind = np.argsort(sample_surprise_values[:,0]) # used only in current fun
        sorted_sample_ind = m.sample_ind[sorted_surprise_ind,0] # used only in current fun
        pow_law_exp = 2  # used only in current fun
        rand_sample_ind = int(np.round(np.random.power(pow_law_exp, 1)*(m.sample_ind.size-1))) # used only in current fun
        m.sample_iteration = sorted_sample_ind[rand_sample_ind]
        print('Experience replay: iteration %d (surprise value: %f)' % (m.sample_iteration, sample_surprise_values[sorted_surprise_ind[rand_sample_ind]]))

        # Load sample RGB-D heightmap
        m.sample_color_heightmap = cv2.imread(os.path.join(l.m.color_heightmaps_directory, '%06d.0.color.png' % (m.sample_iteration)))
        m.sample_color_heightmap = cv2.cvtColor(m.sample_color_heightmap, cv2.COLOR_BGR2RGB)
        m.sample_depth_heightmap = cv2.imread(os.path.join(l.m.depth_heightmaps_directory, '%06d.0.depth.png' % (m.sample_iteration)), -1)
        m.sample_depth_heightmap = m.sample_depth_heightmap.astype(np.float32)/100000
                    
        if a.grasp_goal_conditioned or a.goal_conditioned:
            if a.is_testing and not a.random_scene_testing:
                # contour is local var
                obj_contour = r.get_test_obj_mask(p.m.nonlocal_variables['goal_obj_idx'])
            else:
                obj_contour = r.get_obj_mask(p.m.nonlocal_variables['goal_obj_idx'])
            m.sample_goal_mask_heightmap = np.zeros(p.m.color_heightmap.shape[:2], np.uint8)
            m.sample_goal_mask_heightmap = utils.get_goal_mask(obj_contour, m.sample_goal_mask_heightmap, a.workspace_limits, a.heightmap_resolution)
            p.m.writer.add_image('goal_mask_heightmap', cv2.cvtColor(m.sample_goal_mask_heightmap, cv2.COLOR_BGR2RGB), global_step=t.m.iteration, walltime=None, dataformats='HWC')


    def fwd_pass(self,
        m: MainloopModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc,
    ): 
        # Compute forward pass with sample
        with torch.no_grad():
            if not a.grasp_goal_conditioned:
                m.sample_push_predictions, m.sample_grasp_predictions, m.sample_state_feat = t.forward(m.sample_color_heightmap, m.sample_depth_heightmap, is_volatile=True)
            else:
                m.sample_push_predictions, m.sample_grasp_predictions, m.sample_state_feat = t.goal_forward(m.sample_color_heightmap, m.sample_depth_heightmap, m.sample_goal_mask_heightmap, is_volatile=True)

        sample_grasp_success = m.sample_reward_value == 1 # useless line
        # Get labels for sample and backpropagate
        # sample_best_pix_ind used only in current fun
        sample_best_pix_ind = (np.asarray(t.m.executed_action_log)[m.sample_iteration,1:4]).astype(int)
        if not a.grasp_goal_conditioned:  
            t.backprop(m.sample_color_heightmap, m.sample_depth_heightmap, m.sample_primitive_action, sample_best_pix_ind, t.m.label_value_log[m.sample_iteration])
        else:
            t.backprop(m.sample_color_heightmap, m.sample_depth_heightmap, m.sample_primitive_action, sample_best_pix_ind, t.m.label_value_log[m.sample_iteration], m.sample_goal_mask_heightmap)

        # Recompute prediction value and label for replay buffer
        if m.sample_primitive_action == 'push':
            t.m.predicted_value_log[m.sample_iteration] = [np.max(m.sample_push_predictions)]
        elif m.sample_primitive_action == 'grasp':
            t.m.predicted_value_log[m.sample_iteration] = [np.max(m.sample_grasp_predictions)]






