import numpy as np

from utils.arg.model import ArgsModel
from utils.proc.model import ProcModel

from robot import Robot
from logger import Logger
from trainer import Trainer


class ProcActionSelect(object):

    def select_action_push_or_grasp(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        """
        Just set 3 vars:
        - m.nonlocal_variables['primitive_action']
        - t.episode_log
        - l.write_to_log
        """

         # ------- Action selection --------
        explore_actions = False
        if a.stage == 'grasp_only':  
            self._grasp_only(m, a, r, l, t)
        elif a.stage == 'push_only':
            self._push_only(m, a, r, l, t)
        elif a.stage == 'push_grasp':
            self._push_grasp(m, a, r, l, t)
        t.m.is_exploit_log.append([0 if explore_actions else 1])
        l.write_to_log('is-exploit', t.m.is_exploit_log)


    def _grasp_only(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        m.nonlocal_variables['primitive_action'] = 'grasp'
        t.m.episode_log.append([m.nonlocal_variables['episode']])
        l.write_to_log('episode', t.m.episode_log)


    def _push_only(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        m.nonlocal_variables['primitive_action'] = 'push'
        t.m.episode_log.append([m.nonlocal_variables['episode']])
        l.write_to_log('episode', t.m.episode_log)
        # executing grasp if grasp reward exceeds reward threshold or push length exceeds max_push_episode_length
        if m.best_grasp_conf > a.grasp_reward_threshold or m.nonlocal_variables['push_step'] == a.max_push_episode_length:
            m.nonlocal_variables['primitive_action'] = 'grasp'
            m.nonlocal_variables['episode_grasp_reward'] = m.best_grasp_conf
            m.nonlocal_variables['episode_ratio_of_grasp_to_push'] = m.best_grasp_conf / m.best_push_conf

    def _push_grasp(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        # testing is more conservative for pushing, somewhat reduce ratio of grasping and pushing
        m.nonlocal_variables['primitive_action'] = 'grasp'
        if a.is_testing:
            if not a.goal_conditioned and not a.grasp_goal_conditioned:
                if m.best_push_conf > 1.5 * m.best_grasp_conf:
                    m.nonlocal_variables['primitive_action'] = 'push'
            else:
                print('border_occupy_ratio', m.nonlocal_variables['border_occupy_ratio'])
                if a.random_scene_testing:
                    if m.best_grasp_conf < 1.5:
                        m.nonlocal_variables['primitive_action'] = 'push'
                else:
                    if m.nonlocal_variables['border_occupy_ratio'] > 0.3 or m.best_grasp_conf < 1.5:
                        m.nonlocal_variables['primitive_action'] = 'push'
        else:
            if m.best_push_conf > m.best_grasp_conf:
                m.nonlocal_variables['primitive_action'] = 'push'

        explore_actions = np.random.uniform() < m.explore_prob

        if explore_actions: # Exploitation (do best action) vs exploration (do other action)
            print('Strategy: explore (exploration probability: %f)' % (m.explore_prob))
            m.nonlocal_variables['primitive_action'] = 'push' if np.random.randint(0,2) == 0 else 'grasp'
        else:
            print('Strategy: exploit (exploration probability: %f)' % (m.explore_prob))
