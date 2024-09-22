from colorsys import yiq_to_rgb

import cv2
import numpy as np
from matplotlib.testing.widgets import click_and_drag


class ProcessLaser:
    def __init__(self, img_resolution):
        self.img_resolution = img_resolution
        self.test_1_points = []
        self.test_2_points = []

    def mask2detect(self, frame, debug=False):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert to HSV
        lower_red = np.array([0, 120, 70])  # Lower bound for the laser color (adjust as needed)
        upper_red = np.array([10, 255, 255])  # Upper bound for the laser color

        mask = cv2.inRange(hsv_frame, lower_red, upper_red)  # Create a mask for laser color
        mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))  # Remove noise
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))  # Expand the laser area
        if debug:
            cv2.imshow('debug mask', mask)
            cv2.waitKey(1)

        return mask

    def detect_laser(self, frame, debug=False, dt=1):

        cnt = None
        contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(colored, contours, -1, (255, 0, 0), 3)
        if len(contours) > 0:

            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 30]  # Filter small contours

            for contour in contours:
                # Compute moments
                Mid = cv2.moments(contour)

                # Calculate the centroid
                cXid = int(Mid["m10"] / np.maximum(Mid["m00"], 1e-10))
                cYid = int(Mid["m01"] / np.maximum(Mid["m00"], 1e-10))
                print(cYid, cXid)
                if cYid > 5 and cXid > 5:
                    cnt = (cXid, cYid)
                    return cXid, cYid

        if debug and len(contours) > 0:
            debug_img = np.ones_like(frame) * 255
            if cnt is not None:
                cv2.circle(debug_img, cnt, 5, (0, 0, 255), 3)
            cv2.namedWindow('debug detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('debug detection', 800, 600)
            cv2.imshow('debug detection', debug_img)
            if dt < 0.5:
                cv2.destroyWindow('debug detection')
        return 0, 0

    def reset_laser_pos(self):
        self.test_1_points = []

    def write_detected_laser_pos(self, frame, test_number):
        image_out = np.zeros_like(frame)
        if test_number == 1:
            for k in range(len(self.test_1_points)):
                cv2.circle(image_out, self.test_1_points[k], 2, (0, 0, 255), 2)
        if test_number == 2:
            for k in range(len(self.test_2_points)):
                cv2.circle(image_out, self.test_2_points[k], 2, (0, 0, 255), 2)
        return image_out

    def detect_laser_menu(self, frame, menu_squares_position, click=(0, 0), debug=False):

        if click == (0, 0):
            mask = self.mask2detect(frame, debug=False)
            cXid, cYid = self.detect_laser(mask, debug=False)
        else:
            cXid, cYid = click
        if menu_squares_position[0][0][0] <= cXid <= menu_squares_position[0][1][0] \
                and menu_squares_position[0][0][1] <= cYid <= menu_squares_position[0][1][1]:
            print('Menu start')
            return 'start'
        if menu_squares_position[1][0][0] <= cXid <= menu_squares_position[1][1][0] \
                and menu_squares_position[1][0][1] <= cYid <= menu_squares_position[1][1][1]:
            print('Menu help')
            return 'help'
        if menu_squares_position[2][0][0] <= cXid <= menu_squares_position[2][1][0] \
                and menu_squares_position[2][0][1] <= cYid <= menu_squares_position[2][1][1]:
            print('Menu quit')
            return 'quit'
        if debug:
            cv2.namedWindow('debug laser menu', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('debug laser menu', 800, 600)
            cv2.circle(frame, (cXid, cYid), 5, (0, 255, 0), 2)
            cv2.imshow('debug laser menu', frame)
            print(cXid, cYid)
            cv2.waitKey(1)

    def detect_laser_help(self, frame, help_box, click=(0, 0)):
        if click == (0, 0):
            mask = self.mask2detect(frame, debug=False)
            cXid, cYid = self.detect_laser(mask, debug=False)
        else:
            cXid, cYid = click
        if help_box[0][0] <= cXid <= help_box[1][0] \
                and help_box[0][1] <= cYid <= help_box[1][1]:
            print('Help quit')
            return True
        return False

    def detect_test_start(self, frame, test_number, laser_boxes, click=(0, 0)):
        if click == (0, 0):
            mask = self.mask2detect(frame, debug=False)
            cXid, cYid = self.detect_laser(mask, debug=False)
        else:
            cXid, cYid = click
        if test_number == 1:
            if laser_boxes[0][0][0] <= cXid <= laser_boxes[0][1][0] \
                    and laser_boxes[0][0][1] <= cYid <= laser_boxes[0][1][1]:
                # print('Laser start')
                return True

        if test_number == 2:
            if laser_boxes[2][0][0] <= cXid <= laser_boxes[2][1][0] \
                    and laser_boxes[2][0][1] <= cYid <= laser_boxes[2][1][1]:
                # print('Laser start')
                return True

        return False

    def detect_test_end(self, frame, test_number, laser_boxes, click=(0, 0)):
        if click == (0, 0):
            mask = self.mask2detect(frame, debug=False)
            cXid, cYid = self.detect_laser(mask, debug=False)
        else:
            cXid, cYid = click
        if test_number == 1:
            if laser_boxes[1][0][0] <= cXid <= laser_boxes[1][1][0] \
                    and laser_boxes[1][0][1] <= cYid <= laser_boxes[1][1][1]:
                print('Laser End')
                return True

        if test_number == 2:
            if laser_boxes[3][0][0] <= cXid <= laser_boxes[3][1][0] \
                    and laser_boxes[3][0][1] <= cYid <= laser_boxes[3][1][1]:
                print('Laser End')
                return True

        return False

    def detect_results_img(self, frame, boxes, click=(0, 0)):
        if click == (0, 0):
            mask = self.mask2detect(frame, debug=False)
            cXid, cYid = self.detect_laser(mask, debug=False)
        else:
            cXid, cYid = click

        if boxes[0][0] <= cXid <= boxes[1][0] \
                and boxes[0][1] <= cYid <= boxes[1][1]:
            print('Laser End')
            return True

        return False

    def detect_laser_on_test(self, test_number, click=(0, 0), ):
        # Check if user is using mouse or laser
        if click == (0, 0):
            mask = self.mask2detect(test_number, debug=False)
            cXid, cYid = self.detect_laser(mask, debug=False)
        else:
            cXid, cYid = click
        # check the test_number
        if test_number == 1:
            self.test_1_points.append((cXid, cYid))
        if test_number == 2:
            self.test_2_points.append((cXid, cYid))
