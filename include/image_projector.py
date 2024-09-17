import cv2
import numpy as np
import screeninfo


class ImageProjector:
    def __init__(self, display='HDMI-0', marker_size=400, marker_ids=None):
        # Initialize the projector with display details, marker size, marker IDs, and camera index
        if marker_ids is None:
            marker_ids = [0, 1, 2, 3]

        # Monitor configuration
        self.display = display
        self.monitor_position = (0, 0)
        self.monitor_resolution = (1920, 1080)
        self.window_name = 'Projector'

        # ArUco marker settings
        self.marker_size = marker_size
        self.marker_ids = marker_ids
        self.marked_image = np.zeros(self.monitor_resolution, np.uint8)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

        # Set up screen info, window, and create marker image
        self.find_screen_info()
        self.create_window2projector()
        self.create_marker_image()
        self.homography_matrix = None

        # Start menu boxes
        self.menu_boxes = []
        self.help_boxes = []

    def get_screen_info(self):
        """ Get screen resolution Returns: monitor resolution """
        return self.monitor_resolution

    def show_image(self, image=np.zeros((600, 800), np.uint8)):
        """ Display the marked image on the projector window. """
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1000)  # Show the marker image for 1 second

    def find_screen_info(self):
        """ Find the screen information for the specified display. """
        for monitor in screeninfo.get_monitors():
            if monitor.name == self.display:
                self.monitor_position, self.monitor_resolution = (monitor.x, monitor.y), (monitor.width, monitor.height)
        if self.monitor_resolution == (0, 0):
            print("Unable to find the screen information.")
            return

    def create_window2projector(self):
        """ Create a window for the projector display in fullscreen mode. """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.window_name, self.monitor_position[0], self.monitor_position[1])
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def generate_aruco_marker(self, marker_id):
        """ Generate an ArUco marker image for a given marker ID. """
        marker_img = np.zeros((self.marker_size, self.marker_size), dtype=np.uint8)
        marker_img = cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, self.marker_size, marker_img, 1)
        return marker_img

    def create_marker_image(self):
        """ Create an image with multiple ArUco markers arranged on a canvas. """
        canvas = np.ones((self.monitor_resolution[1], self.monitor_resolution[0]),
                         dtype=np.uint8) * 255  # Create a white image
        markers = [self.generate_aruco_marker(marker_id) for marker_id in self.marker_ids]  # Generate all markers

        # Predefined positions to place markers on the canvas
        positions = [
            (50, 50),  # Top-left
            (self.monitor_resolution[0] - self.marker_size - 50, 50),  # Top-right
            (50, self.monitor_resolution[1] - self.marker_size - 50),  # Bottom-left
            (self.monitor_resolution[0] - self.marker_size - 50, self.monitor_resolution[1] - self.marker_size - 50)
            # Bottom-right
        ]

        # Place each marker on the canvas at the calculated positions
        for marker, pos in zip(markers, positions):
            x, y = pos
            canvas[y:y + self.marker_size, x:x + self.marker_size] = marker

        self.marked_image = canvas  # Update the marked image



    def menu_image(self, rectangle_size):
        height, width = self.monitor_resolution
        image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Define font type and color
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 0)  # Black color

        # Define text position
        y0, dy = height // 4, 80  # Starting y position and vertical spacing
        offset_box = 150
        text_lines = [
            "Welcome to the Trembling Meter!",
            "Start",
            "Help",
            "Quit"
        ]
        # Draw text on the image
        for i, line in enumerate(text_lines):
            if i == 0:
                y = y0 + i * dy
                cv2.putText(image, line, (width // 10, y), font, fontScale=1.2, color=(0, 127, 255), thickness=2)
            else:
                y = y0 + i * dy
                cv2.putText(image, line, (width // 4, y), font, fontScale=1, color=(0, 0, 0), thickness=2)
            if i > 0:
                cv2.rectangle(image, (width // 4 - rectangle_size + offset_box, y - rectangle_size),
                              (width // 4 + rectangle_size + offset_box, y + rectangle_size), color=(0, 0, 0),
                              thickness=2)
                self.menu_boxes.append((width // 4 + rectangle_size + offset_box, y + rectangle_size))

        cv2.putText(image, 'Fundamentos da Visao Computacional', (width - width // 2, height - height // 8), font,
                    fontScale=0.5, color=(0, 0, 0), thickness=2)
        cv2.putText(image, 'LABMETRO - UFSC', (width - width // 4, height - height // 8 + 20), font, fontScale=0.5,
                    color=(0, 0, 0), thickness=2)
        return image

    def help_menu_image(self):
        height, width = self.monitor_resolution
        image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Define font type and color
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (0, 0, 0)  # Black color

        # Define text position
        y0, dy = self.monitor_resolution[0] // 4, 60  # Starting y position and vertical spacing

        text_lines = [
            "Welcome to the Trembling Meter!",
            "Start",
            "Help",
            "Quit"
        ]
        # Draw text on the image
        for i, line in enumerate(text_lines):
            y = y0 + i * dy
            cv2.putText(image, line, (width // 4, y), font, fontScale=1, color=(0, 0, 0), thickness=2)
        cv2.rectangle(image, (width  -200, height - 200), (width - 150, height -150), color=(0, 0, 0), thickness=2)
        return image