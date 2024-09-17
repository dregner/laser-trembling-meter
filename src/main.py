import os
from logging import lastResort
from math import trunc

import cv2
import numpy as np
import time

from include.image_projector import ImageProjector
from include.process_laser import ProcessLaser
from include.camera_control import CameraClass
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('X: {}, Y: {}'.format(x, y))
        param.append((x, y))


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


def main(projector_output, debug=True):
    frame_counter = 0
    elapsed_time = 10
    fps = 0
    projector = ImageProjector(display=projector_output, marker_size=400)
    width, height = projector.get_screen_info()
    camera_process = CameraClass(monitor_resoltuion=(width, height), camera_input=0, camera_resolution=(width, height),
                                 marker_size=400)

    laser_process = ProcessLaser(img_resolution=(height, width))
    click = []
    cv2.setMouseCallback(projector.window_name, mouse_callback, click)
    # projector.show_image()
    print('Opened Camera')
    transform_perspective = False
    start_test = False

    projector.show_image(image=projector.marked_image)
    cv2.namedWindow('Webcam Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Webcam Feed', 800, 600)
    while camera_process.camera.isOpened():
        ret, frame = camera_process.camera.read()
        frame_counter += 1
        # Put the FPS text on the frame
        if transform_perspective:
            frame = camera_process.correct_perspective(frame)

            if menu_screen and len(click) > 0:
                    laser_menu = laser_process.detect_laser_menu(frame=frame, menu_squares_position=projector.menu_boxes, mouse=click[-1])
                    if laser_menu == 'Start':
                        start_test = True
                    if laser_menu == 'Help':
                        help_menu, menu_screen = True, False
                    if laser_menu == 'Quit':
                        k = ord('q')
        if not ret:
            break

        # check what key was pressed
        k = cv2.waitKey(1)
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
            cv2.imwrite('../images/frame{}.jpg'.format(frame_counter), frame)
            print('saved image: {}'.format(frame_counter))

        if help_menu:
            projector.show_image(image=projector.help_menu_image())
            laser_process.detect_laser_help(frame=frame, help_box=projector.help_boxes, mouse=click[-1])

        # if 'space' is pressed: wrap perspective
        if cv_menu == 'perspective':
            transform_perspective = True
            # if camera_process.process_frame(frame):
            #     print("Homography matrix found")
            menu_screen = True
            projector.show_image(image=projector.menu_image(rectangle_size=20))
            print(projector.menu_boxes)
            cv2.waitKey(int(1e3/fps))

        # If menu reset pressed, reset perspective
        if cv_menu == 'reset':
            # print('Reseting Perspective')
            projector.show_image(projector.marked_image)
            transform_perspective, start_test = False, False


        fps, frame_counter = camera_process.frame_counter(frame_counter=frame_counter, fps=fps)

        if k == ord('q'):
            break

        if debug:
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Webcam Feed', frame)

    camera_process.release_camera()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    global mouseX, mouseY

    projector_output_source = '\\\\.\\DISPLAY1'  # 4' numero do proj sala de aula
    main(projector_output_source)
