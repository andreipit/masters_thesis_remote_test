import time

from utils.arg.model import ArgsModel
from robot import Robot
from logger import Logger
from trainer import Trainer

from utils.proc.model import ProcModel
from utils.proc.init import ProcInit

from utils.proc.goal_cond import ProcGoalCond
from utils.proc.action_select import ProcActionSelect
from utils.proc.action_pos import ProcActionPos
from utils.proc.pixel_pos import ProcPixelPos
from utils.proc.primitive_save import ProcPrimitiveSave
from utils.proc.execute_grasp import ProcExecuteGrasp
from utils.proc.execute_push import ProcExecutePush  


class Proc():

    m: ProcModel = None
    init: ProcInit = None
    goal_cond: ProcGoalCond = None
    act_select: ProcActionSelect = None
    act_pos: ProcActionPos =None
    pix_pos: ProcPixelPos = None
    primitive: ProcPrimitiveSave = None
    exec_push: ProcExecutePush = None
    exec_grasp: ProcExecuteGrasp = None


    def __init__(self):
        pass

    def create_empty_helpers(self, a: ArgsModel, r: Robot, l: Logger, t: Trainer):
        self.a = a
        self.r = r
        self.l = l
        self.t = t

        # empty init:
        self.m = ProcModel() # just table
        self.init = ProcInit()
        self.goal_cond = ProcGoalCond()
        self.act_select = ProcActionSelect()
        self.act_pos = ProcActionPos()
        self.pix_pos = ProcPixelPos()
        self.primitive = ProcPrimitiveSave()
        self.exec_push = ProcExecutePush()
        self.exec_grasp = ProcExecuteGrasp()


    def process_actions(self):
        """
        Summary: each frame:
        - find best pixel (received from sensor) by masking goal object
        - convert it to 3d coords 
        - move robot closer to it
        """
        # 1) get mask img around vertices of mesh of current obj # don't forget to fill m.color_heightmap in main_train_loop
        if self.goal_cond.save_goal_contour(self.m, self.a, self.r, self.l, self.t) == False:
            return # continue

        # 2) just set m.nonlocal_variables['primitive_action'] = push or grasp, using stage
        self.act_select.select_action_push_or_grasp(self.m, self.a, self.r, self.l, self.t)

        # 3) find best pixel ->  get "predicted_value": trainer finds best_pixel on mask => m.push_predictions gives value for that pixel
        # then just save it to t.m.predicted_value_log # #orig: Save predicted confidence value
        self.act_pos.generate_selected_act_pos(self.m, self.a, self.r, self.l, self.t)

        # 4) convert 2d to 3d
        self.pix_pos.compute_pixel_3d_position(self.m, self.a, self.r, self.l, self.t)

        # 5) save proc.m.nonlocal_variables to trainer.m.executed_action_log
        self.primitive.save(self.m, self.a, self.r, self.l, self.t)

        # 6) just change 3 variables orig: # ------- Executing Actions --------- 
        self.m.nonlocal_variables['push_success'] = False
        self.m.nonlocal_variables['grasp_success'] = False
        self.m.change_detected = False # used later in maint train loop

        # 7) # Large method. Calls robot.push/grasp. Execute primitive (primitive is "small move/rot"???)
        if self.m.nonlocal_variables['primitive_action'] == 'push':
            self.exec_push.execute(self.m, self.a, self.r, self.l, self.t)
        elif self.m.nonlocal_variables['primitive_action'] == 'grasp':
            self.exec_grasp.execute(self.m, self.a, self.r, self.l, self.t)
        self.m.nonlocal_variables['executing_action'] = False
        
