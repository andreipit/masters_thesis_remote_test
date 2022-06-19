class TrainerGetPushDirectionVis(object):


    def get_push_direction_vis(self, push_predictions, color_heightmap):
        push_direction_canvas = color_heightmap
        x = 0
        while x < push_predictions.shape[2]:
            y = 0
            while y < push_predictions.shape[1]:
                angle_idx = np.argmax(push_predictions[:, y, x])
                angel = np.deg2rad(angle_idx*(360.0/self.model.num_rotations))
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


    def get_best_push_direction_vis(self, best_pix_ind, color_heightmap):
        push_direction_canvas = color_heightmap
        angle_idx = best_pix_ind[0]
        angel = np.deg2rad(angle_idx*(360.0/self.model.num_rotations))
        start_point = (best_pix_ind[2], best_pix_ind[1])
        end_point = (int(best_pix_ind[2] + 20*np.cos(angel)), int(best_pix_ind[1] + 20*np.sin(angel)))
        cv2.arrowedLine(push_direction_canvas,start_point, end_point, (0,0,255), 1, 0, 0, 0.3)
        cv2.circle(push_direction_canvas, (int(best_pix_ind[2]), int(best_pix_ind[1])), 4, (0,0,255), 1)

        return push_direction_canvas
