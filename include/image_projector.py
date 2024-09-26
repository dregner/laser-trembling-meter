import cv2
import numpy as np
import screeninfo
import time
import matplotlib.pyplot as plt
from io import BytesIO


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

        # images
        self.menu_img, self.menu_boxes = self.create_menu_img(rectangle_size=50)
        self.h_test_img, self.v_test_img, self.test_boxes, self.test_lines = self.create_test_img()
        self.help_img, self.help_boxes = self.create_help_img()
        # print('Projector Initiate')
        self.result_box = []
        print('Menu boxes', self.menu_boxes)
        print('Help boxes', self.help_boxes)

    def get_screen_info(self):
        """ Get screen resolution Returns: monitor resolution """
        return self.monitor_resolution

    def show_image(self, image=np.zeros((600, 800), np.uint8)):
        """ Display the marked image on the projector window. """
        cv2.imshow(self.window_name, image)

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
        preview = np.ones((self.monitor_resolution[1], self.monitor_resolution[0], 3), np.uint8)
        cv2.putText(preview, 'WAITING FOR CAMERA . . .',
                    (self.monitor_resolution[1] // 4, self.monitor_resolution[0] // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.2, color=(255, 255, 0), thickness=2)
        cv2.imshow(self.window_name, preview)

    def generate_aruco_marker(self, marker_id):
        """ Generate an ArUco marker image for a given marker ID. """
        marker_img = np.zeros((self.marker_size, self.marker_size), dtype=np.uint8)
        marker_img = cv2.aruco.generateImageMarker(self.aruco_dict, marker_id, self.marker_size, marker_img, 1)
        return marker_img

    def create_marker_image(self):
        """ Create an image with multiple ArUco markers arranged on a canvas. """
        canvas = np.ones((self.monitor_resolution[1], self.monitor_resolution[0]),
                         dtype=np.uint8) * 255 # Create a white image
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

        cv2.putText(canvas, '\'space\' to correct perspective', (self.monitor_resolution[0] // 3,
                    self.monitor_resolution[1] // 3), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=(0, 0, 0), thickness=2)
        cv2.putText(canvas, '\'r\' to reset until it works', (self.monitor_resolution[0] // 3,
                    self.monitor_resolution[1] // 3+100), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8, color=(0, 0, 0), thickness=2)
        self.marked_image = canvas  # Update the marked image

    def create_menu_img(self, rectangle_size):
        width, height = self.monitor_resolution
        image = np.ones((height, width, 3), dtype=np.uint8)
        menu_boxes = []
        # Define font type and color
        font = cv2.FONT_HERSHEY_SIMPLEX

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
                cv2.putText(image, line, (width // 10, y), font, fontScale=1.2, color=(255, 255, 0), thickness=2)
            else:
                y = y0 + i * dy
                cv2.putText(image, line, (width // 4, y), font, fontScale=1, color=(255, 255, 0), thickness=2)
            if i > 0:
                cv2.rectangle(image, (width // 4 - rectangle_size // 2 + offset_box, y - rectangle_size // 2),
                              (width // 4 + rectangle_size // 2 + offset_box, y + rectangle_size // 2), color=(255, 255, 0),
                              thickness=2)
                menu_boxes.append([(width // 4 - rectangle_size // 2 + offset_box, y - rectangle_size // 2),
                                   (width // 4 + rectangle_size // 2 + offset_box, y + rectangle_size // 2)])

        cv2.putText(image, 'Fundamentos da Visao Computacional', (width - width // 2 - 100, height - height // 8), font,
                    fontScale=0.6, color=(255, 255, 0), thickness=1)
        cv2.putText(image, 'LABMETRO - UFSC', (width - width // 4, height - height // 8 + 20), font, fontScale=0.6,
                    color=(255, 255, 0), thickness=1)
        return image, menu_boxes

    def create_help_img(self, rect_size=100):
        width, height = self.monitor_resolution
        image = np.ones((height, width, 3), dtype=np.uint8)
        boxes = []
        # Define font type and color
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Define text position
        y0, dy = self.monitor_resolution[0] // 4, 60  # Starting y position and vertical spacing

        text_lines = [
            "Bem Vindo ao Tremometro Laser!",
            "Inicio",
            "Ajuda",
            "Sair"
        ]
        # Draw text on the image
        for i, line in enumerate(text_lines):
            y = y0 + i * dy
            cv2.putText(image, line, (width // 4, y), font, fontScale=1, color=(255, 255, 0), thickness=2)
        cv2.rectangle(image, (width - 300, height - 300), (width - 200, height - 200), color=(255, 255, 0), thickness=2)
        boxes = [(width - 300, height - 300), (width - 200, height - 200)]

        return image, boxes

    def create_test_img(self, rect_size=25):
        # top-left and bottom-right points for horizontal boxesl
        h_boxes = [[(100, self.monitor_resolution[1] // 2 - rect_size), (150, self.monitor_resolution[1] // 2 + rect_size)],
                   [(self.monitor_resolution[0] - 125 - rect_size, self.monitor_resolution[1] // 2 - rect_size),
                    (self.monitor_resolution[0] - 75 - rect_size, self.monitor_resolution[1] // 2 + rect_size)]]

        # top-left and bottom-right points for vertical boxes
        v_boxes = [[(self.monitor_resolution[0] // 2 - rect_size, 50), (self.monitor_resolution[0] // 2 + rect_size, 100)],
                   [(self.monitor_resolution[0] // 2 - rect_size, self.monitor_resolution[1] - 75 - rect_size),
                    (self.monitor_resolution[0] // 2 + rect_size, self.monitor_resolution[1] - 25 - rect_size)]]

        h_img = np.ones((self.monitor_resolution[1], self.monitor_resolution[0], 3), dtype=np.uint8)
        cv2.line(h_img, (h_boxes[0][1][0], h_boxes[0][1][1] - rect_size)
                 , (h_boxes[1][0][0], h_boxes[1][1][1] - rect_size), color=(255, 0, 0), thickness=4)
        cv2.rectangle(h_img, h_boxes[0][0], h_boxes[0][1], color=(255, 255, 0), thickness=2)
        cv2.rectangle(h_img, h_boxes[1][0], h_boxes[1][1], color=(255, 255, 0), thickness=2)
        lines_h = [(h_boxes[0][1][0], h_boxes[0][1][1] - rect_size), (h_boxes[1][1][0], h_boxes[1][1][1] - rect_size)]

        v_img = np.ones_like(h_img, dtype=np.uint8)
        cv2.line(v_img, (v_boxes[0][0][0] + rect_size, v_boxes[0][1][1])
                 , (v_boxes[1][0][0] + rect_size, v_boxes[1][0][1]), color=(255, 0, 0), thickness=4)
        cv2.rectangle(v_img, v_boxes[0][0], v_boxes[0][1], color=(255, 255, 0), thickness=2)
        cv2.rectangle(v_img, v_boxes[1][0], v_boxes[1][1], color=(255, 255, 0), thickness=2)
        lines_v = [(v_boxes[0][0][0] + rect_size, v_boxes[0][1][1]), (v_boxes[1][0][0] + rect_size, v_boxes[1][0][1])]

        return h_img, v_img, np.concatenate((h_boxes, v_boxes)), np.concatenate((lines_h, lines_v))

    def update_circle_position(self, test_number, start_time, elapsed_time, laser_pos):
        """Calculates and displays the moving circle."""
        t_percentage = (time.time() - start_time - 3) / elapsed_time
        if test_number == 1:
            frame_with_circle = self.h_test_img.copy()
            line_points = self.test_lines[:2]
        elif test_number == 2:
            frame_with_circle = self.v_test_img.copy()
            line_points = self.test_lines[2:]

        start_point = line_points[0]
        end_point = line_points[1]

        current_pos = (
            int(start_point[0] + t_percentage * (end_point[0] - start_point[0])),
            int(start_point[1] + t_percentage * (end_point[1] - start_point[1]))
        )
        image = cv2.circle(frame_with_circle, current_pos, radius=10, color=(255, 255, 0), thickness=-1)
        if laser_pos is not None:
            output_img = self.update_tracked_laser(image, laser_pos)
        else:
            output_img = image
        if current_pos[0] >= end_point[0] and current_pos[1] >= end_point[1]:
            return output_img
        else:
            pass
            return output_img

    def update_tracked_laser(self, image, laser_pos):
        if len(laser_pos) > 0:
            # Ensure laser_pos is a NumPy array of integers and has the correct shape
            laser_pos = np.array(laser_pos, dtype=np.int32).reshape((-1, 1, 2))

            # Draw the polyline on the image
            return cv2.polylines(image, [laser_pos], False, (0, 255, 0), thickness=1)
        else:
            return image

    def result_image(self, result_h, result_v, dpi=100):
        # Calculate the figure size in inches from the projector resolution and DPI
        width_in_inches = self.monitor_resolution[0] / dpi
        height_in_inches = self.monitor_resolution[1] / dpi

        # Create a 2x2 grid of plots with extra space for additional info (use gridspec for layout adjustment)
        fig, axs = plt.subplots(2, 2, figsize=(width_in_inches, height_in_inches), dpi=dpi)

        # Set a title for the figure
        fig.suptitle('2x2 Plots with Extra Information', fontsize=16)

        # First plot: Sine wave
        t = np.linspace(0, 2 * np.pi, 100)
        axs[0, 0].plot(t, np.sin(t), 'r')
        axs[0, 0].set_title('Sine Wave')

        # Second plot: Cosine wave
        axs[0, 1].plot(t, np.cos(t), 'b')
        axs[0, 1].set_title('Cosine Wave')

        # Third plot: Tangent wave
        axs[1, 0].plot(t, np.tan(t), 'g')
        axs[1, 0].set_ylim([-10, 10])  # Limit y axis for tan
        axs[1, 0].set_title('Tangent Wave')

        # Fourth plot: Exponential decay
        axs[1, 1].plot(t, np.exp(-t), 'm')
        axs[1, 1].set_title('Exponential Decay')

        # Add extra info (annotations, text, etc.) next to the plots
        plt.figtext(0.7, 0.85, "Extra Information:", fontsize=12, ha='left', va='center', color='black')
        plt.figtext(0.7, 0.80, "- Sine wave in red", fontsize=10, ha='left', va='center', color='red')
        plt.figtext(0.7, 0.75, "- Cosine wave in blue", fontsize=10, ha='left', va='center', color='blue')
        plt.figtext(0.7, 0.70, "- Tangent wave in green", fontsize=10, ha='left', va='center', color='green')
        plt.figtext(0.7, 0.65, "- Exponential decay in magenta", fontsize=10, ha='left', va='center', color='magenta')

        # Adjust layout to make room for the text
        plt.tight_layout(rect=[0, 0, 0.7, 1])  # Leave space on the right for the extra info

        # cv2.putText(image,'Trembling Result', (self.monitor_resolution[0] // 2,100),
        #             cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        # Convert the figure to a NumPy array (instead of saving to a file)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        # Convert the buffer to a NumPy array and then to an OpenCV image (Mat)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(img_arr, 1)  # Convert the PNG buffer into an OpenCV image
        cv2.putText(image, 'Menu', (self.monitor_resolution[0]-200, self.monitor_resolution[1]-250),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 0,0), thickness=1)
        cv2.putText(image, 'Aperte \'esc\'', (self.monitor_resolution[0]-20, self.monitor_resolution[1]-200),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 0,0), thickness=1)
        # cv2.rectangle(image, (self.monitor_resolution[0] - 125, self.monitor_resolution[1] - 125),
        #               (self.monitor_resolution[0] - 25, self.monitor_resolution[1] - 25), color=(0, 0, 0), thickness=-1)
        #
        # self.result_box = [(self.monitor_resolution[0] - 100, self.monitor_resolution[1] - 100),
        #                    (self.monitor_resolution[0] - 50, self.monitor_resolution[1] - 50)]

        return image

