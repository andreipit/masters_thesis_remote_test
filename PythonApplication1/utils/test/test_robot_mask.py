# default
import numpy as np
import time

# local
from utils.arg.parser_json import ParserJson
from utils.arg.model import ArgsModel
from robot import Robot
from matplotlib import pyplot as plt

from entry_point import EntryPoint


def _plot_image(image):

    plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    im = image
    fig, ax = plt.subplots()
    im = ax.imshow(im, extent=[0, 300, 0, 300])
    x = np.array(range(300))
    ax.plot(x, x, ls='dotted', linewidth=2, color='red')
    plt.show()

if __name__ == '__main__':
    a: ArgsModel = ParserJson.convert_dict_to_vars(ParserJson.load_config(debug = False))
    np.random.seed(a.random_seed)
    
    r: Robot = Robot(a)
    EntryPoint.r_connect_cammatrix_shot(a, r)
    r.obj.instantiate_cubes(r.m)

    while True:
        print('type command: \
        1 - get_test_obj_mask, \
        2 - get_test_obj_masks, \
        3 - get_obj_mask, \
        4 - get_obj_masks, \
        q - quit, \
        r - restart')

        time.sleep(.1)
        x = input()
       
        if x == '1': # not working
            print('not working')
            #r.obj.seed_test_objects(r.m)
            #r.mask.get_test_obj_mask(1, r.sim, r.m)
        elif x == '2': # not working
            print('not working')
            #r.obj.seed_test_objects(r.m)
            #r.mask.get_test_obj_masks(r.sim, r.m)
        elif x == '3':
            obj_contour = r.mask.get_obj_mask(1, r.sim, r.m)
            print('obj_contour =', type(obj_contour), obj_contour.shape, obj_contour)

            _plot_image(obj_contour)

            #m.mask = np.zeros(p.m.color_heightmap.shape[:2], np.uint8)
            #m.mask = utils.get_goal_mask(obj_contour, m.mask, a.workspace_limits, a.heightmap_resolution)

        elif x == '4':
            obj_contours = r.mask.get_obj_masks(r.sim, r.m)
            print('obj_contours = ', type(obj_contours), len(obj_contours))
            for obj_contour in obj_contours:
                print('obj_contour =', type(obj_contour), obj_contour.shape)
                _plot_image(obj_contour)

        elif x == 'q':
            break 
        elif x == 'r':
            r.sim.restart_sim(r.m)
            r.sim.stop_start_game_fix(r.m)

