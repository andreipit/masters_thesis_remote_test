import cv2

from utils.arg.model import ArgsModel
from utils.proc.model import ProcModel

from robot import Robot
from logger import Logger
from trainer import Trainer

class ProcPrimitiveSave(object):
    def save(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        """
        Just save proc.m.nonlocal_variables to trainer.m.executed_action_log
        """
        # Save executed primitive
        if m.nonlocal_variables['primitive_action'] == 'push':
            t.m.executed_action_log.append([0, m.nonlocal_variables['best_pix_ind'][0], m.nonlocal_variables['best_pix_ind'][1], m.nonlocal_variables['best_pix_ind'][2]]) # 0 - push
        elif m.nonlocal_variables['primitive_action'] == 'grasp':
            t.m.executed_action_log.append([1, m.nonlocal_variables['best_pix_ind'][0], m.nonlocal_variables['best_pix_ind'][1], m.nonlocal_variables['best_pix_ind'][2]]) # 1 - grasp
        l.write_to_log('executed-action', t.m.executed_action_log)

    def visualize(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        """
        just create 3 images.png
        """
        # Visualize executed primitive, and affordances
        if a.save_visualizations:
            push_pred_vis = t.get_prediction_vis(m.push_predictions, m.color_heightmap, m.nonlocal_variables['best_pix_ind'])
            l.save_visualizations(t.m.iteration, push_pred_vis, 'push')
            cv2.imwrite('visualization.push.png', push_pred_vis)
            grasp_pred_vis = t.get_prediction_vis(m.grasp_predictions, m.color_heightmap, m.nonlocal_variables['best_pix_ind'])
            l.save_visualizations(t.m.iteration, grasp_pred_vis, 'grasp')
            cv2.imwrite('visualization.grasp.png', grasp_pred_vis)
            push_direction_vis = t.get_best_push_direction_vis(m.nonlocal_variables['best_pix_ind'], m.color_heightmap)
            l.save_visualizations(t.m.iteration, push_direction_vis, 'best_push_direction')
            cv2.imwrite('visualization.best_push_direction.png', push_direction_vis)



