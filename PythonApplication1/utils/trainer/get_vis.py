from scipy import ndimage
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.arg.model import ArgsModel
from utils.trainer.model import TrainerModel

class TrainerGetVis(object):

    def get_prediction_vis(self, m: TrainerModel, predictions, color_heightmap, best_pix_ind):

        canvas = None
        num_rotations = predictions.shape[0]
        for canvas_row in range(int(num_rotations/4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row*4+canvas_col
                prediction_vis = predictions[rotate_idx,:,:].copy()
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0,0,255), 2)
                prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                background_image = ndimage.rotate(color_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

        return canvas


    def get_push_direction_vis(self, m: TrainerModel, push_predictions, color_heightmap):
        push_direction_canvas = color_heightmap
        x = 0
        while x < push_predictions.shape[2]:
            y = 0
            while y < push_predictions.shape[1]:
                angle_idx = np.argmax(push_predictions[:, y, x])
                angel = np.deg2rad(angle_idx*(360.0/m.model.num_rotations))
                start_point = (x, y)
                end_point = (int(x + 10*np.cos(angel)), int(y + 10*np.sin(angel)))
                quality = np.max(push_predictions[:, y, x])
                
                color = (0, 0, (quality*255).astype(np.uint8))
                cv2.arrowedLine(push_direction_canvas,start_point, end_point, (0,0,255), 1, 0, 0, 0.3)
                y+=10
            x+=10

        plt.figure()
        plt.imshow(push_direction_canvas)
        plt.show()
        return push_direction_canvas


    def get_best_push_direction_vis(self, m: TrainerModel, best_pix_ind, color_heightmap):
        push_direction_canvas = color_heightmap
        angle_idx = best_pix_ind[0]
        angel = np.deg2rad(angle_idx*(360.0/m.model.num_rotations))
        start_point = (best_pix_ind[2], best_pix_ind[1])
        end_point = (int(best_pix_ind[2] + 20*np.cos(angel)), int(best_pix_ind[1] + 20*np.sin(angel)))
        cv2.arrowedLine(push_direction_canvas,start_point, end_point, (0,0,255), 1, 0, 0, 0.3)
        cv2.circle(push_direction_canvas, (int(best_pix_ind[2]), int(best_pix_ind[1])), 4, (0,0,255), 1)

        return push_direction_canvas

