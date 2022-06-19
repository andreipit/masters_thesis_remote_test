# MARLI P
from utils.train_loop.model import MainloopModel
from utils.arg.model import ArgsModel

from robot import Robot
from logger import Logger
from trainer import Trainer
from proc import Proc

import torch

class MainloopRestart(object):
    def restart(self, m: MainloopModel,a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc):
        # Restart for push_only stage and goal-conditioned case
        if p.m.nonlocal_variables['push_step'] == a.max_push_episode_length + 1 or p.m.nonlocal_variables['new_episode_flag'] == 1 or p.m.nonlocal_variables['restart_scene'] == r.m.num_obj / 2:
            p.m.nonlocal_variables['push_step'] = 0  # reset push step
            p.m.nonlocal_variables['new_episode_flag'] = 0
            # save episode_improved_grasp_reward
            print('episode %d begins' % p.m.nonlocal_variables['episode'])
            if p.m.nonlocal_variables['restart_scene'] == r.m.num_obj / 2: # If at end of test run, re-load original weights (before test run)
                p.m.nonlocal_variables['restart_scene'] = 0
                r.sim.restart_sim()
                r.add_objects()
                if a.is_testing: # If at end of test run, re-load original weights (before test run)
                    t.m.model.load_state_dict(torch.load(a.snapshot_file))

            t.m.clearance_log.append([t.m.iteration])
            l.write_to_log('clearance', t.m.clearance_log)
            if a.is_testing and len(t.m.clearance_log) >= a.max_test_trials:
                m.exit_called = True # Exit after training thread (backprop and saving labels)
            #continue
            return False # false breaks loop, replaces continue
        return True

            

