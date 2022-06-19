from utils.arg.model import ArgsModel
from utils.proc.model import ProcModel

from robot import Robot
from logger import Logger
from trainer import Trainer

class ProcExecuteGrasp(object):

    def execute(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        m.nonlocal_variables['grasp_success'], color_image, depth_image, color_height_map, depth_height_map, grasped_object_ind \
           = r.grasp(m.primitive_position, m.best_rotation_angle, a.workspace_limits)                  
        
        print('Grasp successful: %r' % (m.nonlocal_variables['grasp_success']))
        m.writer.add_scalar('grasp success', m.nonlocal_variables['grasp_success'], m.nonlocal_variables['episode'])   
                    
        if m.nonlocal_variables['grasp_success']:
            print('Grasp object: %d' % grasped_object_ind)
            t.m.grasp_obj_log.append([grasped_object_ind])
            l.write_to_log('grasp-obj', t.m.grasp_obj_log) 
        else:
            t.m.grasp_obj_log.append([-1])
            l.write_to_log('grasp-obj', t.m.grasp_obj_log) 

        self._update_episode(m,a,r,l,t)

        if a.goal_conditioned:
            self._handle_goal_cond(m,a,r,l,t, grasped_object_ind)

    def _update_episode(self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        # update episode
        m.nonlocal_variables['episode'] += 1
        if a.stage == 'push_only':
            m.writer.add_scalar('episode improved grasp reward', m.nonlocal_variables['episode_improved_grasp_reward'], m.nonlocal_variables['episode'])
            m.nonlocal_variables['episode_improved_grasp_reward'] = 0                        
            # update push step
            print('step %d in episode (five pushes correspond one episode)' % m.nonlocal_variables['push_step'])
            m.writer.add_scalar('episode push step', m.nonlocal_variables['push_step'], m.nonlocal_variables['episode'])
            m.nonlocal_variables['push_step'] += 1
            m.nonlocal_variables['new_episode_flag'] = 1
            
    def _handle_goal_cond(
        self, m:ProcModel, a: ArgsModel, r: Robot, l: Logger, t: Trainer,
        grasped_object_ind
    ):
        if m.nonlocal_variables['grasp_success']:
            if grasped_object_ind == m.nonlocal_variables['goal_obj_idx']:
                m.nonlocal_variables['goal_catched'] = 1
                print('Goal object catched!')
                m.nonlocal_variables['new_episode_flag'] = 1
                if a.is_testing:
                    m.nonlocal_variables['restart_scene'] = r.m.num_obj / 2
            else:
                m.nonlocal_variables['goal_catched'] = 0.5
                print('A different goal catched! Change the goal object index to', grasped_object_ind)
                if not a.is_testing:
                    m.nonlocal_variables['goal_obj_idx'] = grasped_object_ind
                    m.nonlocal_variables['new_episode_flag'] = 1
        else:
            m.nonlocal_variables['goal_catched'] = 0
        m.writer.add_scalar('episode goal catched', m.nonlocal_variables['goal_catched'], m.nonlocal_variables['episode'])

