import numpy as np
import cv2

def update_bgr_mask(image, lower_bgr, upper_bgr, kernel_size):
    """Update and display the mask in BGR color space."""
    mask = cv2.inRange(image, lower_bgr, upper_bgr)
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]

    # Apply morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel)

    # Apply the mask to the original BGR image to retain the color and make everything else black
    bgr_mask = cv2.bitwise_and(image, image, mask=mask)

    # Stack the original BGR image and the BGR mask side by side
    stacked_result = np.hstack((image, bgr_mask))

    # Display both the original BGR image and the BGR mask
    cv2.imshow('BGR Image and Mask', stacked_result)

def update_hsv_mask(image, lower_hsv, upper_hsv, kernel_size):
    """Update and display the mask in HSV color space."""
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]

    # Apply morphological operations
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.dilate(mask, kernel)

    # Apply the mask to the original BGR image to retain the color and make everything else black
    hsv_mask = cv2.bitwise_and(image, image, mask=mask)

    # Stack the original BGR image and the HSV mask side by side
    stacked_result = np.hstack((image, hsv_mask))

    # Display both the original BGR image and the HSV mask
    cv2.imshow('HSV Image and Mask', stacked_result)

def on_bgr_trackbar_change(val):
    # Get the current values from the BGR trackbars
    lower_b = cv2.getTrackbarPos('Lower Blue', 'BGR Trackbars')
    lower_g = cv2.getTrackbarPos('Lower Green', 'BGR Trackbars')
    lower_r = cv2.getTrackbarPos('Lower Red', 'BGR Trackbars')
    upper_b = cv2.getTrackbarPos('Upper Blue', 'BGR Trackbars')
    upper_g = cv2.getTrackbarPos('Upper Green', 'BGR Trackbars')
    upper_r = cv2.getTrackbarPos('Upper Red', 'BGR Trackbars')
    kernel_size = cv2.getTrackbarPos('Kernel Size', 'BGR Trackbars')

    # Update lower and upper bounds in BGR color space
    lower_bgr = np.array([lower_b, lower_g, lower_r])
    upper_bgr = np.array([upper_b, upper_g, upper_r])

    # Ensure kernel size is odd and greater than 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 1:
        kernel_size = 1

    # Update the mask display
    update_bgr_mask(image, lower_bgr, upper_bgr, kernel_size)

def on_hsv_trackbar_change(val):
    # Get the current values from the HSV trackbars
    lower_h = cv2.getTrackbarPos('Lower Hue', 'HSV Trackbars')
    lower_s = cv2.getTrackbarPos('Lower Sat', 'HSV Trackbars')
    lower_v = cv2.getTrackbarPos('Lower Val', 'HSV Trackbars')
    upper_h = cv2.getTrackbarPos('Upper Hue', 'HSV Trackbars')
    upper_s = cv2.getTrackbarPos('Upper Sat', 'HSV Trackbars')
    upper_v = cv2.getTrackbarPos('Upper Val', 'HSV Trackbars')
    kernel_size = cv2.getTrackbarPos('Kernel Size', 'HSV Trackbars')

    # Update lower and upper bounds in HSV color space
    lower_hsv = np.array([lower_h, lower_s, lower_v])
    upper_hsv = np.array([upper_h, upper_s, upper_v])

    # Ensure kernel size is odd and greater than 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 1:
        kernel_size = 1

    # Update the mask display
    update_hsv_mask(image, lower_hsv, upper_hsv, kernel_size)

def main():
    global image
    # Load the image
    image_path = '../images/frame7.jpg'  # Update your path here
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        print("Error: Image not found!")
        return

    # Create windows for BGR and HSV trackbars
    # cv2.namedWindow('BGR Trackbars')
    cv2.namedWindow('HSV Trackbars')

    # # Create trackbars for lower and upper BGR bounds and kernel size
    # cv2.createTrackbar('Lower Blue', 'BGR Trackbars', 0, 255, on_bgr_trackbar_change)
    # cv2.createTrackbar('Lower Green', 'BGR Trackbars', 0, 255, on_bgr_trackbar_change)
    # cv2.createTrackbar('Lower Red', 'BGR Trackbars', 0, 255, on_bgr_trackbar_change)
    # cv2.createTrackbar('Upper Blue', 'BGR Trackbars', 255, 255, on_bgr_trackbar_change)
    # cv2.createTrackbar('Upper Green', 'BGR Trackbars', 255, 255, on_bgr_trackbar_change)
    # cv2.createTrackbar('Upper Red', 'BGR Trackbars', 255, 255, on_bgr_trackbar_change)
    # cv2.createTrackbar('Kernel Size', 'BGR Trackbars', 5, 30, on_bgr_trackbar_change)

    # Create trackbars for lower and upper HSV bounds and kernel size
    cv2.createTrackbar('Lower Hue', 'HSV Trackbars', 0, 179, on_hsv_trackbar_change)
    cv2.createTrackbar('Lower Sat', 'HSV Trackbars', 0, 255, on_hsv_trackbar_change)
    cv2.createTrackbar('Lower Val', 'HSV Trackbars', 0, 255, on_hsv_trackbar_change)
    cv2.createTrackbar('Upper Hue', 'HSV Trackbars', 179, 179, on_hsv_trackbar_change)
    cv2.createTrackbar('Upper Sat', 'HSV Trackbars', 255, 255, on_hsv_trackbar_change)
    cv2.createTrackbar('Upper Val', 'HSV Trackbars', 255, 255, on_hsv_trackbar_change)
    cv2.createTrackbar('Kernel Size', 'HSV Trackbars', 5, 30, on_hsv_trackbar_change)

    # Initial call to display mask with default values
    # on_bgr_trackbar_change(0)
    on_hsv_trackbar_change(0)

    # Wait until the user presses any key
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
