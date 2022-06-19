import torch.nn as nn

class TrainerModel(object):

    debug_mode:bool = False

    model: nn.Module = None

    stage = None
    grasp_goal_conditioned = None
    is_testing = None
    alternating_training = None
    use_cuda = None
    explore_model = None
    future_reward_discount = None
    criterion = None
    optimizer = None
    iteration = None

    # buffers lists:
    executed_action_log = []
    label_value_log = []
    reward_value_log = []
    predicted_value_log = []
    use_heuristic_log = []
    is_exploit_log = []
    clearance_log = []
    push_step_log = []
    grasp_obj_log = [] # grasp object index (if push or grasp fail then index is -1)
    episode_log = []
    episode_improved_grasp_reward_log = []

    def __init__(self):

        pass




