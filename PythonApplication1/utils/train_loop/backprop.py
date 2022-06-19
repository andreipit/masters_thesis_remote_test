# MARLI P
from utils.train_loop.model import MainloopModel
from utils.arg.model import ArgsModel

from robot import Robot
from logger import Logger
from trainer import Trainer
from proc import Proc

class MainloopBackprop(object):
    def run(
        self, m: MainloopModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc,
    ):
        # Backpropagate
        if not a.grasp_goal_conditioned:
            # loss used only in that fun
            loss = t.backprop(m.prev_color_heightmap, m.prev_valid_depth_heightmap, m.prev_primitive_action, m.prev_best_pix_ind, m.label_value)
        else:
            loss = t.backprop(m.prev_color_heightmap, m.prev_valid_depth_heightmap, m.prev_primitive_action, m.prev_best_pix_ind, m.label_value, m.prev_goal_mask_heightmap)
        p.m.writer.add_scalar('loss', loss, t.m.iteration)

        p.m.episode_loss += loss
        if p.m.nonlocal_variables['push_step'] == a.max_push_episode_length or p.m.nonlocal_variables['new_episode_flag'] == 1:
            p.m.writer.add_scalar('episode loss', p.m.episode_loss, p.m.nonlocal_variables['episode'])
            p.m.episode_loss = 0
            




