# default
import numpy as np
#import numpy.typing as npt

# 3rd party
import argparse

# local
#from utils.arg.parser import ArgsParser
from utils.arg.parser_json import ParserJson
from utils.arg.model import ArgsModel
from logger import Logger
from robot import Robot
import time 
from matplotlib import pyplot as plt
from utils.custom_types import NDArray
from entry_point import EntryPoint

if __name__ == '__main__':
    a: ArgsModel = ParserJson.convert_dict_to_vars(ParserJson.load_config(debug = False))
    np.random.seed(a.random_seed)
    
    r: Robot = Robot(a)
    EntryPoint.r_connect_cammatrix_shot(a, r)
    r.obj.instantiate_cubes(r.m)

    while True:
        print('type command: 1 - test, q - quit, r - restart, obj - instantiate, photo - shot')
        time.sleep(.1)
        x = input()
       
        if x == '1':
            print('1 was pressed')
        elif x == 'q':
            break 
        elif x == 'r':
            r.sim.restart_sim(r.m)
            r.sim.stop_start_game_fix(r.m)
        elif x == 'photo':
            bg_color_img: NDArray["480,640,3", np.uint8] = None
            bg_depth_img: NDArray["480,640", float] = None
            bg_color_img, bg_depth_img = r.sim.get_2_perspcamera_photos_480x640(r.m)

            plt.imshow(bg_color_img)
            plt.show(block=True)

            plt.imshow(bg_depth_img)
            plt.show(block=True)

            #plt.plot([1, 2, 3], [1, 2, 3], '-.', c='red', label = 'bubble')
            #plt.legend(loc="upper left")      
            ##plt.savefig('output.png')
            #plt.show(block=True)

        elif x == 'obj':
            r.obj.instantiate_cubes(r.m)






#x = list(map(int, input().split()))
#N = len(x)
#if x[0] == 1:
#print(' '.join(map(str, x)))
