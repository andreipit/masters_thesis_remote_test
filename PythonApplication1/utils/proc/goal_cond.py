from utils.arg.model import ArgsModel
from utils.proc.model import ProcModel

from robot import Robot
from logger import Logger
from trainer import Trainer

import utils.utils as utils
import torch
import numpy as np

class ProcGoalCond(object):
    
    def save_goal_contour(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        # find concrete cube in scene. Load it's mesh and put in same place/rot.
        # get vertices from mesh and connect them (fill)
        # convert result to pixels
        # get intersection of pixels vs our camera
        # save it as m.grasp_predictions
        # if not big enough -> restart

        m.push_predictions = m.nonlocal_variables['push_predictions']
        m.grasp_predictions = m.nonlocal_variables['grasp_predictions']

        # For goal-conditioned case, cut grasp predictions with goal mask
        if a.goal_conditioned: # true at 1.2
            if self._get_object_contour(m,a,r,l,t) == False:
                return False
        m.best_push_conf = np.max(m.push_predictions)
        m.best_grasp_conf = np.max(m.grasp_predictions)
        m.nonlocal_variables['grasp_reward'] = m.best_grasp_conf
        print('Primitive confidence scores: %f (push), %f (grasp)' % (m.best_push_conf, m.best_grasp_conf))
        return True

       
    def _get_object_contour(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        if a.is_testing and not a.random_scene_testing:
            obj_contour = r.mask.get_test_obj_mask(m.nonlocal_variables['goal_obj_idx'], r.sim, r.m)
        else: # this condition work in 1.2
            obj_contour = r.mask.get_obj_mask(m.nonlocal_variables['goal_obj_idx'], r.sim, r.m)
        
        #print('m.color_heightmap=',m.color_heightmap)
        mask = np.zeros(m.color_heightmap.shape[:2], np.uint8)
        mask = utils.get_goal_mask(obj_contour, mask, a.workspace_limits, a.heightmap_resolution)
        obj_grasp_prediction = np.multiply(m.grasp_predictions, mask)
        m.grasp_predictions = obj_grasp_prediction / 255
        # if goal object is pushed completely out of scene, restart scene
        if np.max(obj_contour[:, 0]) < 0 or np.max(obj_contour[:, 1]) < 0 or np.min(obj_contour[:, 0]) > 224 or np.min(obj_contour[:, 1]) > 224:
            self._restart_scene(m, a, r, l, t)
            return False
        return True

             
    def _restart_scene(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        m.nonlocal_variables['new_episode_flag'] = 1
        r.sim.restart_sim(r.m)
        r.add_objects()
        if a.is_testing: # If at end of test run, re-load original weights (before test run)
            t.m.model.load_state_dict(torch.load(a.snapshot_file))
        t.m.clearance_log.append([t.m.iteration])
        l.write_to_log('clearance', t.m.clearance_log)
        if a.is_testing and len(t.m.clearance_log) >= a.max_test_trials:
            m.exit_called = True # Exit after training thread (backprop and saving labels)
        #continue



