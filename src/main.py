import os
import time
import cv2
import matplotlib.pyplot as plt
from include.image_projector import ImageProjector
from include.process_laser import ProcessLaser
from include.camera_control import CameraClass


def mouse_callback(event, x, y, flags, param):
    # if event == cv2.EVENT_MOUSEMOVE:
        # param.append((x, y))
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
    elif k == 27:
        return 'menu'
    elif k == ord('q'):
        return 'quit'
    return None


def main(projector_output, debug=True):
    frame_counter, fps = 0, 0
    elapsed_time = 10
    test_number = 1  # Start with the first test (1: horizontal, 2: vertical)
    click = [(0, 0)]
    t0 = None

    projector = ImageProjector(display=projector_output, marker_size=200)
    width, height = projector.get_screen_info()
    camera_process = CameraClass(
        monitor_resoltuion=(width, height),
        camera_input=1,
        camera_resolution=(width, height),
        marker_size=200
    )
    laser_process = ProcessLaser(img_resolution=(height, width))

    cv2.setMouseCallback(projector.window_name, mouse_callback, click)
    projector.show_image(image=projector.marked_image)

    print('Opened Camera')
    if debug:
        cv2.namedWindow('Webcam Feed', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Webcam Feed', 800, 600)

    menu_screen = False
    menu_help = False
    start_test = False
    test_started = False
    transform_perspective = False
    result_test = False

    while camera_process.camera.isOpened():
        ret, frame = camera_process.camera.read()
        if not ret:
            break
        frame_counter += 1
        if transform_perspective:
            frame = camera_process.correct_perspective(frame=frame)
            click.append(laser_process.detect_laser(frame=frame,debug=True))
            # if result_test and test_number > 2:
            #     result = frame[projector.result_box[1][0]:projector.result_box[1][1],
            #     projector.result_box[0][0]:projector.result_box[0][1]]
            #     click.append(laser_process.detect_laser(frame=result,debug=True))
        k = cv2.waitKey(1)
        cv_menu = opencv_menu(k)

        if cv_menu == 'perspective':
            if camera_process.process_frame(frame):
                transform_perspective = True
                menu_screen = True
                projector.show_image(image=projector.menu_img)
                print('Menu bbx: ', projector.menu_boxes)
                cv2.waitKey(int(1e3 / fps))

        if menu_screen:
            laser_menu = laser_process.detect_laser_menu(frame, projector.menu_boxes, click=click[-1], debug=False)
            if laser_menu == 'start':
                start_test, result_test, menu_screen = True, False, False
                test_number = 1  # Start from horizontal test
                projector.show_image(image=projector.h_test_img)
                t0 = time.time()

            elif laser_menu == 'help':
                projector.show_image(image=projector.help_img)
                menu_help, menu_screen = True, False
            elif laser_menu == 'quit':
                break

        if start_test:
            # Detect laser start and perform test logic
            if laser_process.detect_test_start(frame, test_number=test_number, laser_boxes=projector.test_boxes,
                                               click=click[-1]) or test_started:
                if round(time.time() - t0, 2) > 3:  # Wait for 3 seconds in the start box
                    test_started = True

                    # Check for the end condition of the test
                    if not laser_process.detect_test_end(frame, test_number=test_number, laser_boxes=projector.test_boxes,
                                                         click=click[-1]):
                        # Update the circle position during the test
                        frame_with_circle = projector.update_circle_position(test_number=test_number,
                                                                             start_time=t0, elapsed_time=elapsed_time)
                        laser_process.detect_laser_on_test(test_number=test_number, click=click[-1])
                        projector.show_image(frame_with_circle)
                    else:
                        # Test is done, save results and transition
                        img_detected_laser = laser_process.write_detected_laser_pos(frame, test_number=test_number)
                        projector.show_image(image=img_detected_laser)
                        cv2.waitKey(2000)  # Wait 2 seconds before transitioning
                        laser_process.reset_laser_pos()
                        test_number += 1
                        test_started = False  # Reset test state for the next test

                        if test_number == 2:
                            projector.show_image(image=projector.v_test_img)
                            t0 = time.time()  # Restart the timer for the vertical test
                            result_test = True
                        else:
                            # Both tests are done, show results
                            result_image = projector.result_image(
                                result_h=laser_process.test_1_points,
                                result_v=laser_process.test_2_points
                            )
                            projector.show_image(image=result_image)
                            start_test, menu_screen, result_test, test_number = False, False, False, 1

        # Save the current frame if 's' is pressed
        if cv_menu == 'save_img':
            os.makedirs('../images', exist_ok=True)
            cv2.imwrite(f'../images/frame{frame_counter}.jpg', frame)
            print(f'Saved image: {frame_counter}')

        # Handle help menu interaction
        if menu_help:
            if laser_process.detect_laser_help(frame, projector.help_boxes, click=click[-1]):
                menu_screen, menu_help = True, False
                projector.show_image(image=projector.menu_img)

        # Reset the application state
        if cv_menu == 'reset':
            projector.show_image(projector.marked_image)
            start_test, test_started, menu_help, transform_perspective = False, False, False, False
            test_number = 1

        if cv_menu == 'menu':
            laser_process.reset_laser_pos()
            projector.show_image(image=projector.menu_img)
            cv2.waitKey(1000)
            menu_screen = True

        if cv_menu == 'quit':
            break
        fps, frame_counter = camera_process.frame_counter(frame_counter, fps)


        if debug:
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('Webcam Feed', frame)

    camera_process.release_camera()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    projector_output_source = '\\\\.\\DISPLAY4' # Number 4 is from projector at class room
    main(projector_output_source)
