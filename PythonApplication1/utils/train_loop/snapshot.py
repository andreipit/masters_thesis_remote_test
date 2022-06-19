# MARLI P
from utils.train_loop.model import MainloopModel
from utils.arg.model import ArgsModel

from robot import Robot
from logger import Logger
from trainer import Trainer
from proc import Proc

import numpy as np


class MainloopSnapshot(object):
    def run(self, m: MainloopModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc): 
        # Save model snapshot
        #print('----------MainloopSnapshot: a.is_testing=',a.is_testing,'t.m.iteration % 2 ==', t.m.iteration % 2)
        if not a.is_testing:
            l.save_backup_model(t.m.model, a.stage)
            if t.m.iteration % 2 == 0: # use 2 for test
                l.save_model(t.m.iteration, t.m.model, a.stage)
                if t.m.use_cuda:
                    t.m.model = t.m.model.cuda()





