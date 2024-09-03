import cv2
from include.image_projector import ImageProjector


def main(projector_output):

    projector = ImageProjector(display=projector_output, marker_size=400, camera_index=1)

    width, height = projector.get_screen_info()
    projector.show_image()
    cap = cv2.VideoCapture(projector.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    transform_perspective = False
    while cap.isOpened():
        ret, frame = cap.read()
        if transform_perspective:
            frame = projector.correct_perspective(frame)
        if not ret:
            break
        cv2.imshow('Webcam Feed', frame)
        k = cv2.waitKey(1)
        if k == 32:
            if not transform_perspective:  # Space key
                projector.process_frame(frame)
                transform_perspective = True
            else:
                print("Transform already made")
        if k == ord('r'):
            print("Reseting transform")
            transform_perspective = False

        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    projector_output_source = '\\\\.\\DISPLAY1'
    main(projector_output_source)