import cv2
import numpy as np


class ProcessLaser:
    def __init__(self, img_resolution):
        self.img_resolution = img_resolution
        self.test_1_points = []
        self.test_2_points = []

    def mask2detect(self, frame, debug=False):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(frame, lower_red, upper_red)  # Create a mask for laser color
        # lower_hsv = np.asarray([0,0,230]) # normal
        # upper_hsv = np.asarray([180,30,255])
        lower_hsv = np.asarray([100, 0, 230])
        upper_hsv = np.asarray([180, 30, 255])
        mask = cv2.inRange(frame, lower_hsv, upper_hsv)
        # mask = cv2.threshold(frame[:, :, 2], 210, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))  # Remove noise
        mask = cv2.dilate(mask, np.ones((7, 7), np.uint8))  # Expand the laser area
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))  # Remove noise
        if debug:
            cv2.namedWindow('debug mask', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('debug mask', 800, 600)
            cv2.imshow('debug mask', mask)

        return mask

    def detect_laser(self, frame, debug=False):
        mask = self.mask2detect(frame, debug=debug)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cXid, cYid = 0, 0

        if len(contours) > 0:
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50]  # Filter small contours

            if len(contours) > 0:  # Ensure there are valid contours after filtering
                contour = contours[0]  # Assuming you want the first contour
                Mid = cv2.moments(contour)

                # Calculate the centroid
                cXid = int(Mid["m10"] / np.maximum(Mid["m00"], 1e-10))
                cYid = int(Mid["m01"] / np.maximum(Mid["m00"], 1e-10))
                # print('Detected point: ', cXid, ' x ', cYid)
                # print('Size: ', cv2.contourArea(contour))

            if debug and len(contours) > 0:
                debug_img = np.ones_like(frame) * 255
                cv2.circle(debug_img, (cXid, cYid), 5, (0, 0, 255), 3)
                cv2.namedWindow('debug detection', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('debug detection', 800, 600)
                cv2.imshow('debug detection', debug_img)
                # Return the detected point
        return cXid, cYid

    def reset_laser_pos(self):
        self.test_1_points = []
        self.test_2_points = []

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
        # if click == (0, 0):
        #     mask = self.mask2detect(frame, debug=False)
        #     cXid, cYid = self.detect_laser(mask, debug=False)
        # print('got it')
        # else:
        cXid, cYid = click
        if menu_squares_position[0][0][0] <= cXid <= menu_squares_position[0][1][0] \
                and menu_squares_position[0][0][1] <= cYid <= menu_squares_position[0][1][1]:
            # print('Menu start')
            return 'start'
        if menu_squares_position[1][0][0] <= cXid <= menu_squares_position[1][1][0] \
                and menu_squares_position[1][0][1] <= cYid <= menu_squares_position[1][1][1]:
            # print('Menu help')
            return 'help'
        if menu_squares_position[2][0][0] <= cXid <= menu_squares_position[2][1][0] \
                and menu_squares_position[2][0][1] <= cYid <= menu_squares_position[2][1][1]:
            # print('Menu quit')
            return 'quit'
        if debug:
            cv2.namedWindow('debug laser menu', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('debug laser menu', 800, 600)
            cv2.circle(frame, (cXid, cYid), 5, (0, 255, 0), 2)
            cv2.imshow('debug laser menu', frame)
            # print(cXid, cYid)
            cv2.waitKey(1)
        return 'continue'

    def detect_laser_help(self, frame, help_box, click=(0, 0)):
        if click == (0, 0):
            mask = self.mask2detect(frame, debug=False)
            cXid, cYid = self.detect_laser(mask, debug=False)
        else:
            cXid, cYid = click
        if help_box[0][0] <= cXid <= help_box[1][0] \
                and help_box[0][1] <= cYid <= help_box[1][1]:
            # print('Help quit')
            return True
        return False

    def detect_test_start(self, frame, test_number, laser_boxes, click=(0, 0)):
        # if click == (0, 0):
        #     mask = self.mask2detect(frame, debug=False)
        #     cXid, cYid = self.detect_laser(mask, debug=False)
        # else:
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
        # if click == (0, 0):
        #     mask = self.mask2detect(frame, debug=False)
        #     cXid, cYid = self.detect_laser(mask, debug=False)
        # else:
        cXid, cYid = click
        if test_number == 1:
            if laser_boxes[1][0][0] <= cXid <= laser_boxes[1][1][0] \
                    and laser_boxes[1][0][1] <= cYid <= laser_boxes[1][1][1]:
                # print('Laser End')
                return True

        if test_number == 2:
            if laser_boxes[3][0][0] <= cXid <= laser_boxes[3][1][0] \
                    and laser_boxes[3][0][1] <= cYid <= laser_boxes[3][1][1]:
                # print('Laser End')
                return True

        return False

    def detect_results_img(self, frame, boxes, click=(0, 0)):
        # if click == (0, 0):
        #     mask = self.mask2detect(frame, debug=False)
        #     cXid, cYid = self.detect_laser(mask, debug=False)
        # else:
        cXid, cYid = click

        if boxes[0][0] <= cXid <= boxes[1][0] \
                and boxes[0][1] <= cYid <= boxes[1][1]:
            # print('Laser End')
            return True

        return False

    def detect_laser_on_test(self, test_number, click=(0, 0), ):
        # Check if user is using mouse or laser
        # if click == (0, 0):
        #     mask = self.mask2detect(test_number, debug=False)
        #     cXid, cYid = self.detect_laser(mask, debug=False)
        # else:
        cXid, cYid = click
        # check the test_number
        if test_number == 1:
            self.test_1_points.append((cXid, cYid))
        if test_number == 2:
            self.test_2_points.append((cXid, cYid))
