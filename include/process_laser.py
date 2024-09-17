from colorsys import yiq_to_rgb

import cv2
import numpy as np


class ProcessLaser:
    def __init__(self, img_resolution):
        self.img_resolution = img_resolution
        self.detected_laser_pos = []

    def mask2detect(self, frame):
        mask = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 180, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        return mask

    def detect_laser(self, frame, debug=False, dt=1):

        mask = self.mask2detect(frame)
        cnt = None
        contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(colored, contours, -1, (255, 0, 0), 3)
        if len(contours) > 0:
            for contour in contours:
                # Compute moments
                Mid = cv2.moments(contour)

                # Calculate the centroid
                cXid = int(Mid["m10"] / np.maximum(Mid["m00"], 1e-10))
                cYid = int(Mid["m01"] / np.maximum(Mid["m00"], 1e-10))
                print(cYid, cXid)
                if cYid > 100 and cXid > 100:
                    cnt = (cXid, cYid)
                    self.detected_laser_pos.append((cXid, cYid))

        if debug and len(contours) > 0:
            debug_img = np.ones_like(frame) * 255
            if cnt is not None:
                cv2.circle(debug_img, cnt, 5, (0, 0, 255), 3)
            cv2.namedWindow('debug detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('debug detection', 800, 600)
            cv2.imshow('debug detection', debug_img)
            if dt < 0.5:
                cv2.destroyWindow('debug detection')

    def write_detected_laser_pos(self, frame):
        image_out = np.zeros_like(frame)
        for k in range(len(self.detected_laser_pos)):
            cv2.circle(image_out, self.detected_laser_pos[k], 2, (0, 0, 255), 2)
        return image_out

    def reset_laser_pos(self):
        self.detected_laser_pos = []

    def detect_laser_menu(self, frame, menu_squares_position, rectangle_size=50, mouse=None):

        xy_start, xy_help, xy_quit = menu_squares_position
        cXid, cYid = 0, 0

        if mouse is None:
            mask = self.mask2detect(frame)
            contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                for contour in contours:
                    # Compute moments
                    Mid = cv2.moments(contour)

                    # Calculate the centroid
                    cXid = int(Mid["m10"] / np.maximum(Mid["m00"], 1e-10))
                    cYid = int(Mid["m01"] / np.maximum(Mid["m00"], 1e-10))
        else:
            cXid, cYid = mouse

        if  xy_start[0] - rectangle_size <= cXid <= xy_start[0] and xy_start[1] - rectangle_size <= cYid <= xy_start[1]:
            print( 'start')
        if xy_help[0] - rectangle_size <= cXid <= xy_help[0] and xy_help[1] - rectangle_size <= cYid <= xy_help[1]:
            print('help')
        if xy_quit[0] - rectangle_size <= cXid <= xy_quit[0] and xy_quit[1] - rectangle_size <= cYid <= xy_quit[1]:
            print('quit')
