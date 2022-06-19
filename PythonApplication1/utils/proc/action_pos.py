import numpy as np

from utils.arg.model import ArgsModel
from utils.proc.model import ProcModel

from robot import Robot
from logger import Logger
from trainer import Trainer
import utils.utils as utils

class ProcActionPos(object):
    
    # --------- Generate position of selected action ----------
    # If heuristic bootstrapping is enabled: if change has not been detected more than 2 times, execute heuristic algorithm to detect grasps/pushes
    # NOTE: typically not necessary and can reduce final performance.
    def generate_selected_act_pos(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        if a.heuristic_bootstrap and m.nonlocal_variables['primitive_action'] == 'push' and m.no_change_count[0] >= 2:
            predicted_value = self._push()
            use_heuristic = True
        elif a.heuristic_bootstrap and m.nonlocal_variables['primitive_action'] == 'grasp' and m.no_change_count[1] >= 2:
            predicted_value = self._grasp()
            use_heuristic = True
        else:
            use_heuristic = False
            # Get pixel location and rotation with highest affordance prediction from heuristic algorithms (rotation, y, x)
            if m.nonlocal_variables['primitive_action'] == 'push':
                predicted_value = self._other_push(m,a,r,l,t)
            elif m.nonlocal_variables['primitive_action'] == 'grasp':
                predicted_value = self._other_grasp(m,a,r,l,t)

        t.m.use_heuristic_log.append([1 if use_heuristic else 0])
        l.write_to_log('use-heuristic', t.m.use_heuristic_log)

        # Save predicted confidence value
        t.m.predicted_value_log.append([predicted_value])
        l.write_to_log('predicted-value', t.m.predicted_value_log)


    def _other_grasp(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        if a.goal_conditioned:
            if a.is_testing and not a.random_scene_testing:
                obj_contour = r.get_test_obj_mask(m.nonlocal_variables['goal_obj_idx'])
            else:
                obj_contour = r.get_obj_mask(m.nonlocal_variables['goal_obj_idx'])
            obj_contour[:, 0] = (obj_contour[:, 0] - a.workspace_limits[0][0]) / a.heightmap_resolution  # drop_x to pixel_dimension2
            obj_contour[:, 1] = (obj_contour[:, 1] - a.workspace_limits[1][0]) / a.heightmap_resolution  # drop_y to pixel_dimension1
            obj_contour = np.array(obj_contour).astype(int)
            # if goal object is pushed completely out of scene, restart episode
            if np.max(obj_contour[:, 0]) < 0 or np.max(obj_contour[:, 1]) < 0 or np.min(obj_contour[:, 0]) > 224 or np.min(obj_contour[:, 1]) > 224:
                m.nonlocal_variables['new_episode_flag'] = 1
                m.nonlocal_variables['restart_scene'] = r.m.num_obj / 2

        m.nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(m.grasp_predictions), m.grasp_predictions.shape)
        predicted_value = np.max(m.grasp_predictions)
        return predicted_value


    def _other_push(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        if a.is_testing and not a.random_scene_testing:
            obj_contours = r.get_test_obj_masks()
            obj_number = len(r.m.test_obj_mesh_files)
        else:
            obj_contours = r.get_obj_masks()
            obj_number = len(r.m.obj_mesh_ind)
            mask = 255 * np.ones(m.color_heightmap.shape[:2], np.uint8)
            mask = utils.get_all_mask(obj_contours, mask, obj_number, a.workspace_limits, a.heightmap_resolution)
            push_predictions = np.multiply(m.push_predictions, mask) / 255

        m.nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
        predicted_value = np.max(push_predictions)
        return predicted_value


    def _grasp(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        print('Change not detected for more than two grasps. Running heuristic grasping.')
        m.nonlocal_variables['best_pix_ind'] = t.grasp_heuristic(m.valid_depth_heightmap)
        m.no_change_count[1] = 0
        predicted_value = m.grasp_predictions[m.nonlocal_variables['best_pix_ind']]
        #use_heuristic = True
        return predicted_value

    def _push(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        print('Change not detected for more than two pushes. Running heuristic pushing.')
        m.nonlocal_variables['best_pix_ind'] = t.push_heuristic(m.valid_depth_heightmap)
        m.no_change_count[0] = 0
        predicted_value = m.push_predictions[m.nonlocal_variables['best_pix_ind']]
        #use_heuristic = True
        return predicted_value




