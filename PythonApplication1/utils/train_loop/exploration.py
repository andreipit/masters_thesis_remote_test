# MARLI P
from utils.train_loop.model import MainloopModel
from utils.arg.model import ArgsModel

from robot import Robot
from logger import Logger
from trainer import Trainer
from proc import Proc

import numpy as np

class MainloopExploration(object):
    def run(
        self, m: MainloopModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc,
    ):
        # Adjust exploration probability
        if not a.is_testing:
            p.m.explore_prob = max(0.5 * np.power(0.9998, t.m.iteration),0.1) if a.explore_rate_decay else 0.5
            p.m.grasp_explore_prob = max(0.8 * np.power(0.998, t.m.iteration),0.1) if a.explore_rate_decay else 0.8





