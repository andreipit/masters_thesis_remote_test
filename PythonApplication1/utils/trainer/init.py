import torch

from utils.arg.model import ArgsModel
from utils.trainer.model import TrainerModel

from utils.models.push_grasp_net import push_grasp_net
from utils.models.goal_conditioned_net import goal_conditioned_net


class TrainerInit(object):
    def __init__(self):
        pass

    def create_network(self, m: TrainerModel, a: ArgsModel):
        self._copy_args(m, a)
        self._check_cuda(m, a)
        # 2 parallell nets: 
        #- push or grasp (3 conv layers) => output_prob
        #- pretrained torchvision => interm_feat
        self._create_conv_qnet(m, a)
        self._set_reward(m, a) # trivial args copy
        self._set_loss(m, a) # huber (less sensitive to outliers then squared)
        self._load_pretrained_model(m, a) # just load_state_dict of model or explore_model
        self._froze_some_nets(m, a) # atGrasp->freezePush, atPush->sometimesFreezeGrasp...
        self._convert_model_cpu_to_gpu(m, a) # small conversion
        self._set_optimizer(m, a) # adam
        self._create_log_lists(m, a) # just buffers
        return m.model

    def _copy_args(self, m: TrainerModel, a: ArgsModel):
        m.stage = a.stage
        m.grasp_goal_conditioned = a.grasp_goal_conditioned
        m.is_testing = a.is_testing
        m.alternating_training = a.alternating_training

    def _check_cuda(self, m: TrainerModel, a: ArgsModel):
        # Check if CUDA can be used
        if torch.cuda.is_available() and not a.force_cpu:
            if m.debug_mode: print("CUDA detected. Running with GPU acceleration.")
            m.use_cuda = True
        elif a.force_cpu:
            if m.debug_mode: print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            m.use_cuda = False
        else:
            if m.debug_mode: print("CUDA is *NOT* detected. Running with only CPU.")
            m.use_cuda = False

    def _create_conv_qnet(self, m: TrainerModel, a: ArgsModel):
        
        # Fully convolutional Q network for deep reinforcement learning
        #this models looks the same
        #2 parallell nets: 
        #- push or grasp (3 conv layers) => output_prob
        #- pretrained torchvision => interm_feat
        #* also augmentaion (rotate)
        #* volatile means "freeze weights" (no_grad)

        if not m.grasp_goal_conditioned:
            m.model = push_grasp_net(m.use_cuda)
        else:
            m.model = goal_conditioned_net(m.use_cuda)
            if a.load_explore_snapshot:
                m.explore_model = push_grasp_net(m.use_cuda)

    def _set_reward(self, m: TrainerModel, a: ArgsModel):
        m.future_reward_discount = a.future_reward_discount

    def _set_loss(self, m: TrainerModel, a: ArgsModel):
        # Initialize Huber loss (less sensitive to outliers in data than the squared error loss)
        #m.criterion = torch.nn.SmoothL1Loss(reduce=False) # Huber loss
        m.criterion = torch.nn.SmoothL1Loss(reduction='none') # Huber loss
        if m.use_cuda:
            m.criterion = m.criterion.cuda()

    def _load_pretrained_model(self, m: TrainerModel, a: ArgsModel):
        # Load pre-trained model
        if a.load_explore_snapshot:
            m.explore_model.load_state_dict(torch.load(a.explore_snapshot_file))
            if m.debug_mode: print('Explore model snapshot loaded from: %s' % (a.explore_snapshot_file))

        if a.load_snapshot:
            if m.stage == 'grasp_only':
                m.model.load_state_dict(torch.load(a.snapshot_file))
                if m.debug_mode: print('Pre-trained model snapshot loaded from: %s' % (a.snapshot_file))
            elif m.stage == 'push_only':
                # only load grasp then initialize grasp
                m.model.load_state_dict(torch.load(a.snapshot_file))
                if m.debug_mode: print('Pre-trained model snapshot loaded from: %s' % (a.snapshot_file))
            elif m.stage == 'push_grasp':
                m.model.load_state_dict(torch.load(a.snapshot_file))
                if m.debug_mode: print('Pre-trained model snapshot loaded from: %s' % (a.snapshot_file))

    def _froze_some_nets(self, m: TrainerModel, a: ArgsModel):
        """
        step1 - Grasp
            hardway:
                1.1) just train C-G 
            easyway:
                1.2) train A-G
                1.3) then use it to train C-G
        step2 - Push (grasp is inside it)
            2) use step1 result to train push (fix grasp net weights)
            3) do same but in cluttered scene (don't fix, ie alternating)

        number = install_run.txt command type

        (1) stage == 'grasp_only': 
            froze push
        (2,3) stage == 'push_only':
            if not cooperative_training: froze grasp     # (ie no_grad)
            if alternating_training: froze push, unfroze grasp
        (4.real world) stage == 'push_grasp': 
            froze push
        """

        # For push_only stage, grasp net is fixed at the beginning, for grasp_only stage, push net will not be trained
        if m.stage == 'push_only':
            if not a.cooperative_training:
                # co-training
                for k,v in m.model.named_parameters():
                    if 'grasp-'in k:
                        v.requires_grad=False # fix parameters
            if m.alternating_training:
                ########################################
                # change me to update different policies
                ########################################
                for k,v in m.model.named_parameters():
                    if 'push-'in k:
                        v.requires_grad=False # fix parameters
                for k,v in m.model.named_parameters():
                    if 'grasp-'in k:
                        v.requires_grad=True # fix parameters
                                                  
            # Print
            for k,v in m.model.named_parameters():
                if 'push-'in k:
                    if m.debug_mode: print(v.requires_grad) # supposed to be false 
                if 'grasp-'in k:
                    if m.debug_mode: print(v.requires_grad) # supposed to be false 

        elif m.stage == 'grasp_only':
            for k,v in m.model.named_parameters():
                if 'push-'in k:
                    v.requires_grad=False # fix parameters
            # Print
            for k,v in m.model.named_parameters():
                if 'push-'in k:
                    if m.debug_mode: print(v.requires_grad) # supposed to be false 
        # for real world experiments
        elif m.stage == 'push_grasp':
            for k,v in m.model.named_parameters():
                if 'push-'in k:
                    v.requires_grad=False # fix parameters 
            # Print
            for k,v in m.model.named_parameters():
                if 'push-'in k:
                    if m.debug_mode: print(v.requires_grad) # supposed to be false 

    def _convert_model_cpu_to_gpu(self, m: TrainerModel, a: ArgsModel):
        # Convert model from CPU to GPU
        if m.use_cuda:
            m.model = m.model.cuda()
            #if m.debug_mode:  print('change to cuda!')
            if a.load_explore_snapshot:
                m.explore_model = m.explore_model.cuda()

    def _set_optimizer(self, m: TrainerModel, a: ArgsModel):
        # Set model to training mode
        m.model.train()

        # Initialize optimizer Adam
        m.optimizer = torch.optim.Adam(m.model.parameters(), lr=1e-4, weight_decay=2e-5, betas=(0.9,0.99))
        m.iteration = 0

    def _create_log_lists(self, m: TrainerModel, a: ArgsModel):
        # Initialize lists to save execution info and RL variables
        m.executed_action_log = []
        m.label_value_log = []
        m.reward_value_log = []
        m.predicted_value_log = []
        m.use_heuristic_log = []
        m.is_exploit_log = []
        m.clearance_log = []
        m.push_step_log = []
        m.grasp_obj_log = [] # grasp object index (if push or grasp fail then index is -1)
        m.episode_log = []
        m.episode_improved_grasp_reward_log = []





