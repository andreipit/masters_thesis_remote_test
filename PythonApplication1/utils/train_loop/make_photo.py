# MARLI P
from utils.train_loop.model import MainloopModel
from utils.arg.model import ArgsModel

from robot import Robot
from logger import Logger
from trainer import Trainer
from proc import Proc

import utils.utils as utils
import cv2
import torch
import time
import numpy as np



class MainloopMakePhoto(object):
    def get_rgb(self, m: MainloopModel,a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc):
        print('\n%s iteration: %d' % ('Testing' if a.is_testing else 'Training', t.m.iteration))
        m.iteration_time_0 = time.time()

        # Make sure simulation is still stable (if not, reset simulation)
        r.check_sim()

        # Get latest RGB-D image
        m.color_img, depth_img = r.get_camera_data()
        m.depth_img = depth_img * r.m.cam_depth_scale # Apply depth scale from calibration


    def get_heightmap(self, m: MainloopModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc):
        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        p.m.color_heightmap, p.m.depth_heightmap = utils.get_heightmap(m.color_img, m.depth_img, r.m.cam_intrinsics, r.m.cam_pose, a.workspace_limits, a.heightmap_resolution)
        p.m.valid_depth_heightmap = p.m.depth_heightmap.copy()
        p.m.valid_depth_heightmap[np.isnan(p.m.valid_depth_heightmap)] = 0
        if a.goal_conditioned:
            if a.is_testing and not a.random_scene_testing:
                obj_contour = r.get_test_obj_mask(p.m.nonlocal_variables['goal_obj_idx'])
            else: # works in 1.2 start
                obj_contour = r.get_obj_mask(p.m.nonlocal_variables['goal_obj_idx'])
            
            p.m.goal_mask_heightmap = np.zeros(p.m.color_heightmap.shape[:2], np.uint8)
            p.m.goal_mask_heightmap = utils.get_goal_mask(obj_contour, p.m.goal_mask_heightmap, a.workspace_limits, a.heightmap_resolution)
            #kernel = np.ones((3,3))
            p.m.nonlocal_variables['border_occupy_ratio'] = utils.get_occupy_ratio(p.m.goal_mask_heightmap, p.m.depth_heightmap)
            p.m.writer.add_scalar('border_occupy_ratio', p.m.nonlocal_variables['border_occupy_ratio'], t.m.iteration)

    def save_rgb_heightmap(self, m: MainloopModel,a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc):
        # Save RGB-D images and RGB-D heightmaps
        l.save_images(t.m.iteration, m.color_img, m.depth_img, '0')
        l.save_heightmaps(t.m.iteration, p.m.color_heightmap, p.m.valid_depth_heightmap, '0')
                
        p.m.writer.add_image('goal_mask_heightmap', cv2.cvtColor(p.m.goal_mask_heightmap, cv2.COLOR_BGR2RGB), global_step=t.m.iteration, walltime=None, dataformats='HWC')
        l.save_visualizations(t.m.iteration, p.m.goal_mask_heightmap, 'mask')
        cv2.imwrite('visualization.mask.png', p.m.goal_mask_heightmap)
        # Reset simulation or pause real-world training if table is empty
        stuff_count = np.zeros(p.m.valid_depth_heightmap.shape)
        stuff_count[p.m.valid_depth_heightmap > 0.02] = 1
        empty_threshold = 300
        if a.is_testing:
            empty_threshold = 10
        if np.sum(stuff_count) < empty_threshold or (p.m.no_change_count[0] + p.m.no_change_count[1] > 10):
            no_change_count = [0, 0]
            if np.sum(stuff_count) < empty_threshold:
                print('Not enough objects in view (value: %d)! Repositioning objects.' % (np.sum(stuff_count)))
            elif m.p.no_change_count[0] + m.p.no_change_count[1] > 10:
                print('Too many no change counts (value: %d)! Repositioning objects.' % (m.p.no_change_count[0] + m.p.no_change_count[1]))

            r.sim.restart_sim(r.m)
            r.add_objects()
            if a.is_testing: # If at end of test run, re-load original weights (before test run)
                t.m.model.load_state_dict(torch.load(a.snapshot_file))

            t.m.clearance_log.append([t.iteration])
            l.write_to_log('clearance', t.m.clearance_log)
            if a.is_testing and len(t.m.clearance_log) >= a.max_test_trials:
                m.exit_called = True # Exit after training thread (backprop and saving labels)
            #continue
            return False # false breaks loop, replaces continue
        return True

            




#print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration))
#iteration_time_0 = time.time()

## Make sure simulation is still stable (if not, reset simulation)
#robot.check_sim()

## Get latest RGB-D image
#color_img, depth_img = robot.get_camera_data()
#depth_img = depth_img * robot.cam_depth_scale # Apply depth scale from calibration
