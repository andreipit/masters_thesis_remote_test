# MARLI P
from utils.train_loop.model import MainloopModel
from utils.arg.model import ArgsModel

from robot import Robot
from logger import Logger
from trainer import Trainer
from proc import Proc

import numpy as np
import utils.utils as utils

class MainloopAction(object):
    def execute(self, m: MainloopModel,a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc):
        t.m.push_step_log.append([p.m.nonlocal_variables['push_step']])
        l.write_to_log('push-step', t.m.push_step_log)              

        if not m.exit_called:
            if a.stage == 'grasp_only' and a.grasp_explore:
                m.grasp_explore_actions = np.random.uniform() < p.m.grasp_explore_prob
                print('Strategy: explore (exploration probability: %f)' % (p.m.grasp_explore_prob))
                if m.grasp_explore_actions:
                    # Run forward pass with network to get affordances
                    m.push_predictions, m.grasp_predictions, m.state_feat = t.forward(p.m.color_heightmap, p.m.valid_depth_heightmap, is_volatile=True, grasp_explore_actions=True)
                    obj_contour = r.get_obj_mask(p.m.nonlocal_variables['goal_obj_idx'])
                    m.mask = np.zeros(p.m.color_heightmap.shape[:2], np.uint8)
                    m.mask = utils.get_goal_mask(obj_contour, m.mask, a.workspace_limits, a.heightmap_resolution)
                    obj_grasp_prediction = np.multiply(m.grasp_predictions, m.mask)
                    m.grasp_predictions = obj_grasp_prediction / 255 # obj_grasp_prediction never used later
                else:
                    m.push_predictions, m.grasp_predictions, m.state_feat = t.goal_forward(p.m.color_heightmap, p.m.valid_depth_heightmap, p.m.goal_mask_heightmap, is_volatile=True)

            else:
                if not a.grasp_goal_conditioned:
                    m.push_predictions, m.grasp_predictions, m.state_feat = t.forward(p.m.color_heightmap, p.m.valid_depth_heightmap, is_volatile=True)
                else:
                    m.push_predictions, m.grasp_predictions, m.state_feat = t.goal_forward(p.m.color_heightmap, p.m.valid_depth_heightmap, p.m.goal_mask_heightmap, is_volatile=True)
            
            p.m.nonlocal_variables['push_predictions'] = m.push_predictions
            p.m.nonlocal_variables['grasp_predictions'] = m.grasp_predictions

            # Execute best primitive action on robot in another thread
            p.m.nonlocal_variables['executing_action'] = True
            print('----------------executing_action--------------')




