import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from fontTools.misc.cython import returns


class CameraClass:
    def __init__(self, camera_input=1, camera_resolution=(800,600), monitor_resoltuion=(800,600), marker_size=400):
        self.camera_resolution = camera_resolution
        self.camera_input = camera_input
        self.camera = cv2.VideoCapture(self.camera_input)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])

        self.homography_matrix = None
        self.monitor_resolution = monitor_resoltuion
        self.start_time = time.time()
        # ArUco marker settings
        self.marker_size = marker_size
        self.marker_ids = [0, 1, 2, 3]
        # Initialize ArUco dictionary and detector parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary=self.aruco_dict, detectorParams=self.parameters)


    def release_camera(self):
        self.camera.release()

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
                [self.camera_resolution[0] - self.marker_size - 50, 50],  # Top-right marker
                [50, self.camera_resolution[1] - self.marker_size - 50],  # Bottom-left marker
                [self.camera_resolution[0] - self.marker_size - 50, self.camera_resolution[1] - self.marker_size - 50]
                # Bottom-right marker
            ], dtype=np.float32)

            detected_points = np.zeros((len(self.marker_ids), 2), dtype=np.float32)
            ids = ids.flatten()  # Flatten the marker IDs array

            # Map detected marker IDs to their corresponding positions
            for i, marker_id in enumerate(ids):
                if marker_id in self.marker_ids:
                    detected_points[marker_id] = corners[i][0][0]  # Store detected corner point

            # If all markers are detected, compute the homography
            if len(detected_points) == len(self.marker_ids):
                self.homography_matrix, _ = cv2.findHomography(detected_points, marker_positions)
                # scale_x = self.monitor_resolution[0] / self.camera_resolution[0]
                # scale_y = self.monitor_resolution[1] / self.camera_resolution[1]
                # self.homography_matrix[0,:]*=scale_x
                # self.homography_matrix[1,:]*=scale_y
                return True
            else:
                return False
        else:
            return False

    def correct_perspective(self, frame):
        """ Correct frame perspective after homography matrix defined"""
        if self.homography_matrix is not None:
            return cv2.warpPerspective(frame, self.homography_matrix, (self.camera_resolution[0], self.camera_resolution[1]))
        else:
            # print("Homography matrix not defined")
            return frame

    def frame_counter(self, frame_counter, fps):
        if (time.time() - self.start_time) > 1:
            fps = round(frame_counter / (time.time() - self.start_time), 1)
            frame_counter = 0
            self.start_time = time.time()
        return fps, frame_counter