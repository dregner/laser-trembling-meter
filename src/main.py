import os
from logging import lastResort
from math import trunc

import cv2
import numpy as np
import time

from include.image_projector import ImageProjector
from include.ProcessLaser import ProcessLaser


def opencv_menu(k):
    if k == 32:
        print('Correcting prespective')
        return 'perspective'

    if k == ord('r'):
        print("Reseting transform")
        return 'reset'

    if k == 13:
        return 'laser'

    if k == ord('s'):
        return 'save_img'
    return None



def main(projector_output):
    projector = ImageProjector(display=projector_output, marker_size=400)

    width, height = projector.get_screen_info()
    projector.show_image(image=projector.marked_image)

    laser_process = ProcessLaser(img_resolution=(height, width))
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # projector.show_image()
    print('Opened Camera')
    transform_perspective = False
    start_test = False

    cv2.namedWindow('Webcam Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Webcam Feed', 800, 600)
    counter = 0
    elapsed_time = 10
    while cap.isOpened():
        counter += 1
        ret, frame = cap.read()
        if transform_perspective:
            frame = projector.correct_perspective(frame)
        if not ret:
            break

        cv2.imshow('Webcam Feed', frame)
        k = cv2.waitKey(1)

        # check what key was pressed
        cv_menu = opencv_menu(k)
        if start_test:

            if round((time.time() - t0), 2) > 1:
                if round((time.time() - t0), 2) < elapsed_time:
                    dt = round(30 + (t0 - time.time()), 2)
                    print("\rTime: {} s".format(dt), end="")
                    laser_process.detect_laser(frame, debug=True, dt=dt)
                else:
                    laser_detected = laser_process.write_detected_laser_pos(frame)
                    projector.show_image(image=laser_detected)
                    laser_process.reset_laser_pos()
                    start_test = False

        if cv_menu == 'save_img':
            os.makedirs('../images', exist_ok=True)
            cv2.imwrite('../images/frame{}.jpg'.format(counter), frame)
            print('saved image: {}'.format(counter))

        # if 'space' is pressed: wrap perspective
        if cv_menu == 'perspective':
            transform_perspective = True
            # print('Correct prespective')
            if projector.process_frame(frame):
                print("Homography matrix found")
                projector.show_image(image=np.ones_like(frame) * 255)

        # If menu reset pressed, reset perspective
        if cv_menu == 'reset':
            # print('Reseting Perspective')
            projector.show_image(projector.marked_image)
            transform_perspective = False
            start_test = False

        # If menu laser is pressed
        if cv_menu == 'laser':
            print("start laser test")
            projector.show_image(image=np.zeros_like(frame))
            start_test = True
            t0 = time.time()
        # Started test and process detection


        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    projector_output_source = '\\\\.\\DISPLAY4'
    main(projector_output_source)
