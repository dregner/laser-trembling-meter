import os
import time
import cv2
import numpy as np

from include.image_projector import ImageProjector
from include.process_laser import ProcessLaser
from include.camera_control import CameraClass


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        param.append((x, y))
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f'X: {x}, Y: {y}')



def opencv_menu(k):
    if k == 32:  # Space key
        print('Correcting perspective')
        return 'perspective'
    elif k == ord('r'):
        print("Resetting transform")
        return 'reset'
    elif k == 13:  # Enter key
        return 'laser'
    elif k == ord('s'):
        return 'save_img'
    return None


def main(projector_output, debug=True):
    frame_counter, fps = 0, 0
    elapsed_time, transform_perspective, menu_help, start_test = 10, False, False, False
    click = [(0,0)]

    projector = ImageProjector(display=projector_output, marker_size=400)
    width, height = projector.get_screen_info()
    camera_process = CameraClass(
        monitor_resoltuion=(width, height),
        camera_input=1,
        camera_resolution=(width, height),
        marker_size=400
    )

    laser_process = ProcessLaser(img_resolution=(height, width))
    cv2.setMouseCallback(projector.window_name, mouse_callback, click)
    projector.show_image(image=projector.marked_image)

    print('Opened Camera')
    if debug:
        cv2.namedWindow('Webcam Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Webcam Feed', 800, 600)

    while camera_process.camera.isOpened():
        ret, frame = camera_process.camera.read()
        if not ret:
            break

        frame_counter += 1

        if transform_perspective:
            frame = camera_process.correct_perspective(frame)

            if menu_screen:
                laser_menu = laser_process.detect_laser_menu(frame, projector.menu_boxes, click=click[-1], debug=False)
                if laser_menu == 'start':
                    test_number = 1
                    start_test, menu_screen = True, False
                    if test_number == 1:
                        test_started = False
                        projector.show_image(image=projector.h_test_img)
                    if test_number == 2:
                        projector.show_image(image=projector.v_test_img)
                    t0 = time.time()

                elif laser_menu == 'help':
                    projector.show_image(image=projector.help_img)
                    menu_help, menu_screen = True, False
                elif laser_menu == 'quit':
                    break

        k = cv2.waitKey(1)
        cv_menu = opencv_menu(k)

        if start_test:
            if laser_process.detect_laser_start(frame, test_number=test_number, laser_boxes=projector.test_boxes, click=click[-1],rect_size=50) or test_started:
                print('Detect start')
                if round(time.time()-t0,2) > 3:
                    test_started = True
                    if round(time.time() - t0, 2) < elapsed_time+3:
                        dt = round(elapsed_time + (t0 - time.time()), 2)
                        print(f"\rTime: {dt} s", end="")
                        # laser_process.detect_laser(frame, debug=True, dt=dt)
                        # Calculate the percentage of time passed (from 0 to 1)
                        t_percentage = (time.time() - t0+3) / elapsed_time
                        start_point = projector.line_test[0]  # Start point from line
                        end_point = projector.line_test[1] # End point from line

                        # Interpolate the position of the moving circle
                        current_pos = (
                            int(start_point[0] + t_percentage * (end_point[0] - start_point[0])),
                            int(start_point[1] + t_percentage * (end_point[1] - start_point[1]))
                        )

                        # Clear the image and draw the moving circle
                        frame_with_circle = projector.h_test_img.copy()
                        cv2.circle(frame_with_circle, current_pos, radius=10, color=(0, 255, 0), thickness=-1)
                        projector.show_image(frame_with_circle)
                    else:
                        laser_detected = laser_process.write_detected_laser_pos(frame)
                        projector.show_image(image=laser_detected)
                        cv2.waitKey(1000)
                        laser_process.reset_laser_pos()
                        start_test, test_started, menu_screen = False, False, True

        if cv_menu == 'save_img':
            os.makedirs('../images', exist_ok=True)
            cv2.imwrite(f'../images/frame{frame_counter}.jpg', frame)
            print(f'Saved image: {frame_counter}')

        if menu_help:
            if laser_process.detect_laser_help(frame, projector.help_boxes, click[-1]):
                menu_screen, menu_help = True, False
                projector.show_image(image=projector.menu_img)


        if cv_menu == 'perspective':
            transform_perspective = True
            if camera_process.process_frame(frame):
                menu_screen = True
                projector.show_image(image=projector.menu_img)
                print('Menu bbx: ', projector.menu_boxes)
                cv2.waitKey(int(1e3 / fps))

        if cv_menu == 'reset':
            projector.show_image(projector.marked_image)
            transform_perspective, start_test, menu_help = False, False, False

        fps, frame_counter = camera_process.frame_counter(frame_counter, fps)

        if k == ord('q'):
            break

        if debug:
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Webcam Feed', frame)

    camera_process.release_camera()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    projector_output_source = '\\\\.\\DISPLAY1'
    main(projector_output_source)
