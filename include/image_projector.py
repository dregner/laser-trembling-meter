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

        # ArUco marker settings
        self.marker_size = marker_size
        self.marker_ids = marker_ids
        self.marked_image = np.zeros(self.monitor_resolution, np.uint8)

        # Initialize ArUco dictionary and detector parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary=self.aruco_dict, detectorParams=self.parameters)

        # Set up screen info, window, and create marker image
        self.find_screen_info()
        self.create_window2projector()
        self.create_marker_image()
        self.homography_matrix = None


    def get_screen_info(self):
        """ Get screen resolution Returns: monitor resolution """
        return self.monitor_resolution

    def show_image(self, image=np.zeros((600,800), np.uint8)):
        """ Display the marked image on the projector window. """
        cv2.imshow('Projector', image)
        cv2.waitKey(1000)  # Show the marker image for 1 second

    def find_screen_info(self):
        """ Find the screen information for the specified display. """
        for monitor in screeninfo.get_monitors():
            if monitor.name == self.display:
                self.monitor_position, self.monitor_resolution = (monitor.x, monitor.y), (monitor.width, monitor.height)
        if self.monitor_resolution == (0, 0):
            print("Unable to find the screen information.")
            return

    def create_window2projector(self, name='Projector'):
        """ Create a window for the projector display in fullscreen mode. """
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(name, self.monitor_position[0], self.monitor_position[1])
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

    def detect_aruco_markers(self, frame):
        """ Detect ArUco markers in the given frame and return their corners and IDs. """
        corners, ids, _ = self.detector.detectMarkers(frame)
        return corners, ids

    def process_frame(self, frame):
        """ Process a frame to detect ArUco markers and correct the perspective. """
        corners, ids = self.detect_aruco_markers(frame)  # Detect markers in the frame
        if ids is not None:  # Check if any markers were detected
            # Define known positions for the markers
            marker_positions = np.array([
                [50, 50],  # Top-left marker
                [self.monitor_resolution[0] - self.marker_size - 50, 50],  # Top-right marker
                [50, self.monitor_resolution[1] - self.marker_size - 50],  # Bottom-left marker
                [self.monitor_resolution[0] - self.marker_size - 50, self.monitor_resolution[1] - self.marker_size - 50]
                # Bottom-right marker
            ], dtype=np.float32)

            # Initialize array to store detected points
            detected_points = np.zeros((len(self.marker_ids), 2), dtype=np.float32)
            ids = ids.flatten()  # Flatten the marker IDs array

            # Map detected marker IDs to their corresponding positions
            for i, marker_id in enumerate(ids):
                if marker_id in self.marker_ids:
                    detected_points[marker_id] = corners[i][0][0]  # Store detected corner point

            # If all markers are detected, compute the homography
            if len(detected_points) == len(self.marker_ids):
                self.homography_matrix, _ = cv2.findHomography(detected_points, marker_positions)
                return True
            else:
                return False
        else:
            return False



    def correct_perspective(self, frame):
        """ Correct frame perspective after homography matrix defined"""
        if self.homography_matrix is not None:
            return cv2.warpPerspective(frame, self.homography_matrix, (frame.shape[1], frame.shape[0]))
        else:
            print("Homography matrix not defined")
            return frame




