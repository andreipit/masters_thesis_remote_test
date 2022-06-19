# MARLI P
from utils.train_loop.model import MainloopModel
from utils.arg.model import ArgsModel

from robot import Robot
from logger import Logger
from trainer import Trainer
from proc import Proc

import utils.utils as utils
from scipy.ndimage import binary_dilation
from skimage.morphology.convex_hull import convex_hull_image

class MainloopRunner(object):
    def detect_changes(self, m: MainloopModel,a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc):
        # Detect changes
        if not a.goal_conditioned:
            depth_diff = abs(p.m.depth_heightmap - m.prev_depth_heightmap) # used only in that fun
            change_threshold = 300 # used only in that fun
            change_value = utils.get_change_value(depth_diff) # used only in that fun
            change_detected = change_value > change_threshold or prev_grasp_success # used only in next fun
            print('Change detected: %r (value: %d)' % (change_detected, change_value))
        else:
            prev_mask_hull = binary_dilation(convex_hull_image(m.prev_goal_mask_heightmap), iterations=5)# used only in that fun
            depth_diff = prev_mask_hull*(m.prev_depth_heightmap-p.m.depth_heightmap)
            change_threshold = 50
            change_value = utils.get_change_value(depth_diff)
            change_detected = change_value > change_threshold
            print('Goal change detected: %r (value: %d)' % (change_detected, change_value)) 

        if change_detected:
            if m.prev_primitive_action == 'push':
                p.m.no_change_count[0] = 0
            elif m.prev_primitive_action == 'grasp':
                p.m.no_change_count[1] = 0
        else:
            if m.prev_primitive_action == 'push':
                p.m.no_change_count[0] += 1
            elif m.prev_primitive_action == 'grasp':
                p.m.no_change_count[1] += 1
        
        return change_detected

    def compute_labels(
        self, m: MainloopModel,a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc,
        change_detected
    ):
        # Compute training labels
        if not a.grasp_goal_conditioned:
            m.label_value, m.prev_reward_value = t.get_label_value(
                m.prev_primitive_action, m.prev_grasp_success, 
                m.prev_grasp_reward, m.prev_improved_grasp_reward, 
                change_detected, p.m.color_heightmap, p.m.valid_depth_heightmap
            )
        else:
            m.label_value, m.prev_reward_value = t.get_label_value(
                m.prev_primitive_action, m.prev_grasp_success, 
                m.prev_grasp_reward, m.prev_improved_grasp_reward, 
                change_detected, p.m.color_heightmap, p.m.valid_depth_heightmap, 
                p.m.goal_mask_heightmap, p.m.nonlocal_variables['goal_catched'], 
                p.m.nonlocal_variables['decreased_occupy_ratio']
            )

        t.m.label_value_log.append([m.label_value])
        l.write_to_log('label-value', t.m.label_value_log)
        t.m.reward_value_log.append([m.prev_reward_value])
        l.write_to_log('reward-value', t.m.reward_value_log)



