# default
import numpy as np
#import numpy.typing as npt
import time 
from matplotlib import pyplot as plt
import threading
import asyncio


# 3rd party
import argparse

# local
from utils.arg.parser_json import ParserJson
from utils.arg.model import ArgsModel
from robot import Robot
from logger import Logger
from trainer import Trainer
from proc import Proc
from monitor import Monitor
from train_loop import TrainLoop
from utils.utils import euler2rotm

class EntryPoint():
    @staticmethod
    def r_connect_cammatrix_shot(a: ArgsModel, r: Robot):
        if not r.sim.connect(r.m):
            r.sim.restart_sim(r.m)
        r.sim.stop_start_game_fix(r.m)
        r.m.cam_pose = r.sim.create_perspcamera_trans_matrix4x4(r.m)
        r.sim.create_constants(r.m)
        r.m.bg_color_img, r.m.bg_depth_img = r.sim.get_2_perspcamera_photos_480x640(r.m)
        # If testing, read object meshes and poses from test case file
        # in 1.2 we don't test, so ignore whole block
        if r.m.is_testing and r.m.test_preset_cases: # false and false at 1.2
            r.obj.seed_test_objects(r.m)

    @staticmethod
    def start(a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc, tl: TrainLoop):
        # 1) ArgsModel
        #a = ParserJson.convert_dict_to_vars(ParserJson.load_config(debug = False))
        np.random.seed(a.random_seed)
    
        # 2) robot simulation and generate cubes
        r.create_empty_helpers(a)
        r.m.copy_args_create_consts(a)
        EntryPoint.r_connect_cammatrix_shot(a, r)
        r.add_objects()

        # 3) logger
        l.create_empty_helpers(a)
        l.init.copy_args(l.m, a)
        l.save_heightmap_info(a.workspace_limits, a.heightmap_resolution) # Save heightmap parameters

        # 4) trainer (networks holder)
        t.create_empty_helpers(a) # just init empty helpers
        t.m.model = t.init.create_network(t.m, a)
        if a.continue_logging: # 'resume training' if was paused
            t.copy_args_from_log(l.m.transitions_directory) # just fill vars in model

        # 5) processor (complex actions in simulation)
        p.create_empty_helpers(a, r, l, t) # just init empty helpers
        p.init.copy_args_create_nonlocal(a, p.m, t.m) # args->m & uses trainer 

        # 6) train loop (main loop of learning)
        tl.create_empty_helpers()
        tl.m.exit_called = False

        # 7) save vars to txt
        Monitor.show(tl, a, r, l, t, p)


    @staticmethod
    def update(a: ArgsModel, r: Robot, l: Logger, t: Trainer, p: Proc, tl: TrainLoop)->bool:
        if not tl.iterate(tl.m, a, r, l, t, p):
            return False

        Monitor.show(tl, a, r, l, t, p)
        time.sleep(0.1)
        return True

    @staticmethod
    def update_proc(p: Proc) -> bool:
        # MonoBehaviour does smth on update. Action type set by his variables.
        if p.m.nonlocal_variables['executing_action']:
            print('executing_action', time.time())
            p.process_actions()

        time.sleep(0.1)
        return True

#if __name__ == '__main__222':
#    print('hi main')
#    a: ArgsModel = ParserJson.convert_dict_to_vars(ParserJson.load_config(debug = False))
#    r: Robot = Robot()
#    r.create_empty_helpers(a)
#    r.m.copy_args_create_consts(a)
#    EntryPoint.r_connect_cammatrix_shot(a, r)
#    r.add_objects()


async def thread2(a, r, l, t, p, tl):
    ## update 2 - main training
    #while True:
    #    if not EntryPoint.update(a, r, l, t, p, tl):
    #        break

    while True:
        await asyncio.sleep(0.01)
        if not EntryPoint.update(a, r, l, t, p, tl):
            break
        #print(2)


async def thread1(a, r, l, t, p, tl):
    ## update 1 - simulator
    #def coroutine():
    #    while True:
    #        EntryPoint.update_proc(p)
    #action_thread = threading.Thread(target=coroutine); action_thread.daemon = True; action_thread.start()  # daemon - thread will stop when main process exits # actually run coroutine, starts right now (not in next frame!)
    

    while True:
        await asyncio.sleep(0.01)
        EntryPoint.update_proc(p)
        print(1)


async def two_loops(a, r, l, t, p, tl):
    task1 = asyncio.create_task(thread1(a, r, l, t, p, tl))
    task2 = asyncio.create_task(thread2(a, r, l, t, p, tl))

    print(f"started at {time.strftime('%X')}")

    # Wait until both tasks are completed (should take
    # around 2 seconds.)
    await task1
    await task2



if __name__ == '__main__':

    print('begin')
    
    #a: ArgsModel = ArgsModel()
    a: ArgsModel = ParserJson.convert_dict_to_vars(ParserJson.load_config(debug = False))

    r: Robot = Robot()
    l: Logger = Logger()
    t: Trainer = Trainer()
    tl: TrainLoop = TrainLoop()
    p: Proc = Proc()

    EntryPoint.start(a, r, l, t, p, tl)

    #asyncio.run(two_loops(a, r, l, t, p, tl))
    
    # update 1 - simulator
    def coroutine():
        while True:
            EntryPoint.update_proc(p)
    action_thread = threading.Thread(target=coroutine); action_thread.daemon = True; action_thread.start()  # daemon - thread will stop when main process exits # actually run coroutine, starts right now (not in next frame!)
    
    # update 2 - main training
    while True:
        if not EntryPoint.update(a, r, l, t, p, tl):
            break





## 5) run processor
## CPU thread 1 - move loop - find pixel & move to it (like update in unity wokrs even if we do nothing)
##action_thread = threading.Thread(target=processor.process_actions); action_thread.daemon = True; action_thread.start()  # daemon - thread will stop when main process exits # actually run coroutine, starts right now (not in next frame!)
##p.m.nonlocal_variables['executing_action'] = True
#i = 0
#while True:
#    i += 1
#    if i > 100: 
#        break
#    print(i, ' ', end='')
#    if p.m.nonlocal_variables['executing_action']:
#        p.process_actions()
#    time.sleep(0.01)


#mon: Monitor = Monitor()
#print('start monitor')
#while True:
#    #mon.show({"One" : 7, "Two" : 10, "Three" : 45, "Four" : 23, "Five" : 77 })
#    mon.show(p.m.nonlocal_variables)
#    time.sleep(0.5)
#    i+=1
#    if i>110: break
#print('end monitor')


## 6) prepare main train loop
## CPU thread 2 - train loop - backprop, exp_replay ... (just fills data for loop1)
#tl: TrainLoop = TrainLoop()
#tl.create_empty_helpers()
    
## 7) run main train loop
#tl.m.exit_called = False
#i = 0
#print('\n main loop:')
#while True:
#    i += 1
#    if i > 10: break
#    print(i, ' ', end='')
#    if not tl.iterate(tl.m, a, r, l, t, p):
#        break




#print('end2')