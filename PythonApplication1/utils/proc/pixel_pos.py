import numpy as np

from utils.arg.model import ArgsModel
from utils.proc.model import ProcModel

from robot import Robot
from logger import Logger
from trainer import Trainer

import utils.utils as utils

class ProcPixelPos(object):
    def compute_pixel_3d_position(self, m: ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        # Compute 3D position of pixel
        print('Action: %s at (%d, %d, %d)' % (m.nonlocal_variables['primitive_action'], m.nonlocal_variables['best_pix_ind'][0], m.nonlocal_variables['best_pix_ind'][1], m.nonlocal_variables['best_pix_ind'][2]))
        m.best_rotation_angle = np.deg2rad(m.nonlocal_variables['best_pix_ind'][0]*(360.0/t.m.model.num_rotations))
        best_pix_x = m.nonlocal_variables['best_pix_ind'][2]
        best_pix_y = m.nonlocal_variables['best_pix_ind'][1]
        # 3D position
        m.primitive_position = [best_pix_x * a.heightmap_resolution + a.workspace_limits[0][0], best_pix_y * a.heightmap_resolution + a.workspace_limits[1][0], m.valid_depth_heightmap[best_pix_y][best_pix_x] + a.workspace_limits[2][0]]
        # If pushing, adjust start position, and make sure z value is safe and not too low
        if m.nonlocal_variables['primitive_action'] == 'push': # or m.nonlocal_variables['primitive_action'] == 'place':
            # simulation parameter
            finger_width = 0.02
            safe_kernel_width = int(np.round((finger_width/2)/a.heightmap_resolution))
            local_region = m.valid_depth_heightmap[max(best_pix_y - safe_kernel_width, 0):min(best_pix_y + safe_kernel_width + 1, m.valid_depth_heightmap.shape[0]), max(best_pix_x - safe_kernel_width, 0):min(best_pix_x + safe_kernel_width + 1, m.valid_depth_heightmap.shape[1])]
            if local_region.size == 0:
                safe_z_position = a.workspace_limits[2][0] - 0.02
            else:
                safe_z_position = np.max_z_position = np.max(local_region) + a.workspace_limits[2][0] - 0.02
            m.primitive_position[2] = safe_z_position
            print('3D z position:', m.primitive_position[2])

            # Before pushing
            if a.stage == 'push_only':
                if m.best_grasp_conf <= a.grasp_reward_threshold and m.nonlocal_variables['push_step'] < a.max_push_episode_length:
                    # Get grasp reward before pushing
                    prev_img = m.color_heightmap
                    obj_contours = r.get_obj_masks()
                    obj_number = len(r.m.obj_mesh_ind)
                    mask_all = np.zeros(prev_img.shape[:2], np.uint8)
                    obj_grasp_predictions, mask_all = utils.get_obj_grasp_predictions(m.grasp_predictions, obj_contours, mask_all, prev_img, obj_number, a.workspace_limits, a.heightmap_resolution)

                    m.prev_single_predictions = [np.max(obj_grasp_predictions[i]) for i in range(len(obj_grasp_predictions))]
                    print('reward of grasping before pushing: ', m.prev_single_predictions) 

                    # Get occupy ratio before pushing
                    if a.goal_conditioned:
                        m.prev_occupy_ratio = utils.get_occupy_ratio(m.goal_mask_heightmap, m.depth_heightmap)      





