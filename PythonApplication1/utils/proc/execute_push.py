import numpy as np
import utils.utils as utils

from utils.arg.model import ArgsModel
from utils.proc.model import ProcModel

from robot import Robot
from logger import Logger
from trainer import Trainer

class ProcExecutePush(object):

    def execute(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        m.nonlocal_variables['push_success'] = r.push(m.primitive_position, m.best_rotation_angle, a.workspace_limits)
        print('Push successful: %r' % (m.nonlocal_variables['push_success']))
        t.m.grasp_obj_log.append([-1])
        l.write_to_log('grasp-obj', t.grasp_obj_log) 
        if a.stage == 'push_only':
            if m.best_grasp_conf <= a.grasp_reward_threshold and m.nonlocal_variables['push_step'] < a.max_push_episode_length:
                self._push(m, a, r, l, t)
                
            # update push step
            print('step %d in episode (at most five pushes correspond one episode)' % m.nonlocal_variables['push_step'])
            m.nonlocal_variables['push_step'] += 1
            
    def _push(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):

        latest_color_img, latest_depth_img, latest_color_heightmap, latest_depth_heightmap, latest_valid_depth_heightmap = self._get_rgbd_and_h_maps(m, a, r, l, t)

        if a.grasp_goal_conditioned or a.goal_conditioned: # Get goal mask heightmap 
            latest_goal_mask_heightmap = self._get_goal_mask(m, a, r, l, t, latest_color_heightmap)

        if not a.grasp_goal_conditioned:
            latest_push_predictions, latest_grasp_predictions, latest_state_feat = t.forward(latest_color_heightmap, latest_valid_depth_heightmap, is_volatile=True)
        else:
            latest_push_predictions, latest_grasp_predictions, latest_state_feat = t.goal_forward(latest_color_heightmap, latest_valid_depth_heightmap, latest_goal_mask_heightmap, is_volatile=True)
                   
        if a.goal_conditioned: # Get grasp reward after pushing
            latest_grasp_predictions = self._get_grasp_reward(m, a, r, l, t, latest_grasp_predictions, latest_color_heightmap)
                     
        img = latest_color_heightmap
        obj_contours = r.get_obj_masks() # get latest contours of objects
        obj_number = len(r.m.obj_mesh_ind) # get mask image and predictions of each object
        mask_all = np.zeros(img.shape[:2], np.uint8)
        obj_grasp_predictions, mask_all = utils.get_obj_grasp_predictions(latest_grasp_predictions, obj_contours, mask_all, img, obj_number, a.workspace_limits, a.heightmap_resolution)

        single_predictions = [np.max(obj_grasp_predictions[i]) for i in range(len(obj_grasp_predictions))]
        print('reward of grasping after pushing: ', single_predictions)

        self._get_reward_improved(m,a,r,l,t, single_predictions)

        if a.goal_conditioned:
            self._get_occupy_ratio(m,a,r,l,t, latest_goal_mask_heightmap, latest_depth_heightmap, m.prev_occupy_ratio)
        
    def _get_reward_improved(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer, single_predictions):
        # Get improved grasp reward by pushing 
        improved_grasp_reward = [single_predictions[i] - m.prev_single_predictions[i] for i in range(len(single_predictions))]
        print('expected reward of pushing(improved grasp reward)', improved_grasp_reward)
        if not a.grasp_goal_conditioned:
            m.nonlocal_variables['improved_grasp_reward'] = np.max(improved_grasp_reward)
        else:
            m.nonlocal_variables['improved_grasp_reward'] = improved_grasp_reward[m.nonlocal_variables['goal_obj_idx']]
        print('improved grasp reward in thread:', m.nonlocal_variables['improved_grasp_reward'])
        m.writer.add_scalar('improved grasp reward', m.nonlocal_variables['improved_grasp_reward'], t.m.iteration)

        m.nonlocal_variables['episode_improved_grasp_reward'] += m.nonlocal_variables['improved_grasp_reward']
        t.m.episode_improved_grasp_reward_log.append([m.nonlocal_variables['episode_improved_grasp_reward']])
        l.write_to_log('episode-improved-grasp-reward', t.m.episode_improved_grasp_reward_log)
    
            
    def _get_occupy_ratio(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer,
        latest_goal_mask_heightmap, latest_depth_heightmap, prev_occupy_ratio):
        # Get occupy ratio after pushing
        occupy_ratio =  utils.get_occupy_ratio(latest_goal_mask_heightmap, latest_depth_heightmap)
        m.nonlocal_variables['decreased_occupy_ratio'] = prev_occupy_ratio - occupy_ratio
        print('decreased_occupy_ratio', m.nonlocal_variables['decreased_occupy_ratio'])
        m.writer.add_scalar('decreased_occupy_ratio', m.nonlocal_variables['decreased_occupy_ratio'], t.m.iteration)

    def _get_grasp_reward(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer, latest_grasp_predictions, latest_color_heightmap):
        if a.is_testing and not a.random_scene_testing:
            obj_contour = r.get_test_obj_mask(m.nonlocal_variables['goal_obj_idx'])
        else:
            obj_contour = r.get_obj_mask(m.nonlocal_variables['goal_obj_idx'])
        mask = np.zeros(latest_color_heightmap.shape[:2], np.uint8)
        mask = utils.get_goal_mask(obj_contour, mask, a.workspace_limits, a.heightmap_resolution)
        latest_obj_grasp_prediction = np.multiply(latest_grasp_predictions, mask)
        latest_grasp_predictions = latest_obj_grasp_prediction / 255
        return latest_grasp_predictions

    def _get_goal_mask(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer, latest_color_heightmap):
        if a.is_testing and not a.random_scene_testing:
            obj_contour = r.get_test_obj_mask(m.nonlocal_variables['goal_obj_idx'])
        else:
            obj_contour = r.get_obj_mask(m.nonlocal_variables['goal_obj_idx'])
        latest_goal_mask_heightmap = np.zeros(latest_color_heightmap.shape[:2], np.uint8)
        latest_goal_mask_heightmap = utils.get_goal_mask(obj_contour, latest_goal_mask_heightmap, a.workspace_limits, a.heightmap_resolution)
        return latest_goal_mask_heightmap

    def _get_rgbd_and_h_maps(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        # rgb
        latest_color_img, latest_depth_img = r.get_camera_data()
        # d
        latest_depth_img = latest_depth_img * r.m.cam_depth_scale # Apply depth scale from calibration
        # h
        latest_color_heightmap, latest_depth_heightmap = utils.get_heightmap(latest_color_img, latest_depth_img, r.cam_intrinsics, r.cam_pose, a.workspace_limits, a.heightmap_resolution)
        latest_valid_depth_heightmap = latest_depth_heightmap.copy()
        latest_valid_depth_heightmap[np.isnan(latest_valid_depth_heightmap)] = 0
        return latest_color_img, latest_depth_img, latest_color_heightmap, latest_depth_heightmap, latest_valid_depth_heightmap





    #def _get_heightmap(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
    #    # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
    #    latest_color_heightmap, latest_depth_heightmap = utils.get_heightmap(latest_color_img, latest_depth_img, r.cam_intrinsics, r.cam_pose, a.workspace_limits, a.heightmap_resolution)
    #    latest_valid_depth_heightmap = latest_depth_heightmap.copy()
    #    latest_valid_depth_heightmap[np.isnan(latest_valid_depth_heightmap)] = 0

    #def _get_rgbd(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
    #    # Get latest RGB-D image
    #    latest_color_img, latest_depth_img = r.get_camera_data()
    #    latest_depth_img = latest_depth_img * r.sim.cam_depth_scale # Apply depth scale from calibration
    #    return latest_color_img, latest_depth_img
