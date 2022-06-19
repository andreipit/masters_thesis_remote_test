from utils.trainer.model import TrainerModel


class TrainerPreload():
    def __init__(self):
        pass

    # Pre-load execution info and RL variables
    def preload(self, transitions_directory, m: TrainerModel):
        m.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'), delimiter=' ')
        m.iteration = m.executed_action_log.shape[0] - 2
        m.executed_action_log = m.executed_action_log[0:m.iteration,:]
        m.executed_action_log = m.executed_action_log.tolist()
        m.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), delimiter=' ')
        m.label_value_log = m.label_value_log[0:m.iteration]
        m.label_value_log.shape = (m.iteration,1)
        m.label_value_log = m.label_value_log.tolist()
        m.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'), delimiter=' ')
        m.predicted_value_log = m.predicted_value_log[0:m.iteration]
        m.predicted_value_log.shape = (m.iteration,1)
        m.predicted_value_log = m.predicted_value_log.tolist()
        m.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
        m.reward_value_log = m.reward_value_log[0:m.iteration]
        m.reward_value_log.shape = (m.iteration,1)
        m.reward_value_log = m.reward_value_log.tolist()
        m.use_heuristic_log = np.loadtxt(os.path.join(transitions_directory, 'use-heuristic.log.txt'), delimiter=' ')
        m.use_heuristic_log = m.use_heuristic_log[0:m.iteration]
        m.use_heuristic_log.shape = (m.iteration,1)
        m.use_heuristic_log = m.use_heuristic_log.tolist()
        m.is_exploit_log = np.loadtxt(os.path.join(transitions_directory, 'is-exploit.log.txt'), delimiter=' ')
        m.is_exploit_log = m.is_exploit_log[0:m.iteration]
        m.is_exploit_log.shape = (m.iteration,1)
        m.is_exploit_log = m.is_exploit_log.tolist()
        m.clearance_log = np.loadtxt(os.path.join(transitions_directory, 'clearance.log.txt'), delimiter=' ')
        m.clearance_log.shape = (m.clearance_log.shape[0],1)
        m.clearance_log = m.clearance_log.tolist()
        m.grasp_obj_log = np.loadtxt(os.path.join(transitions_directory, 'grasp-obj.log.txt'), delimiter=' ')
        m.grasp_obj_log = m.grasp_obj_log[0:m.iteration]
        m.grasp_obj_log.shape = (m.iteration,1)
        m.grasp_obj_log = m.grasp_obj_log.tolist()
        
        if not m.is_testing:
            m.episode_log = np.loadtxt(os.path.join(transitions_directory, 'episode.log.txt'), delimiter=' ')
            m.episode_log = m.episode_log[0:m.iteration]
            m.episode_log.shape = (m.iteration,1)
            m.episode_log = m.episode_log.tolist()
        
        if m.stage == 'push_only':
            m.push_step_log = np.loadtxt(os.path.join(transitions_directory, 'push-step.log.txt'), delimiter=' ')
            m.push_step_log = m.push_step_log[0:len(m.push_step_log)]
            m.push_step_log.shape = (len(m.push_step_log),1)
            m.push_step_log = m.push_step_log.tolist()
            m.episode_improved_grasp_reward_log = np.loadtxt(os.path.join(transitions_directory, 'episode-improved-grasp-reward.log.txt'), delimiter=' ')
            m.episode_improved_grasp_reward_log = m.episode_improved_grasp_reward_log[0:len(m.episode_improved_grasp_reward_log)]
            m.episode_improved_grasp_reward_log.shape = (len(m.episode_improved_grasp_reward_log),1)
            m.episode_improved_grasp_reward_log = m.episode_improved_grasp_reward_log.tolist()
