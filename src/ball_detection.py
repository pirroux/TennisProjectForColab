import sys
sys.path.append('/content/TennisProject/src')
import numpy as np
import cv2
import torch
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks
from ball_tracker_net import BallTrackerNet
from detection import center_of_box
from utils import get_video_properties
import tensorflow as tf


def combine_three_frames(frame1, frame2, frame3, width, height):
    """
    Combine three frames into one input tensor for detecting the ball
    """

    # Resize and type converting for each frame
    img = cv2.resize(frame1, (width, height))
    # input must be float type
    img = img.astype(np.float32)

    # resize it
    img1 = cv2.resize(frame2, (width, height))
    # input must be float type
    img1 = img1.astype(np.float32)

    # resize it
    img2 = cv2.resize(frame3, (width, height))
    # input must be float type
    img2 = img2.astype(np.float32)

    # combine three imgs to  (width , height, rgb*3)
    imgs = np.concatenate((img, img1, img2), axis=2)

    # since the odering of TrackNet  is 'channels_first', so we need to change the axis
    imgs = np.rollaxis(imgs, 2, 0)
    return np.array(imgs)


class BallDetector:
    """
    Ball Detector model responsible for receiving the frames and detecting the ball
    """
    def __init__(self, weights_path, out_channels=2):
        """
        Initialize ball detector model
        :param weights_path: path to model weights
        """
        # Determine device (CUDA if available, else CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model weights
        checkpoint = torch.load(weights_path, map_location=self.device)

        # Extract model state from checkpoint
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint

        print("Available keys in checkpoint:", checkpoint.keys())

        # Initialize model parameters
        self.detector = BallTrackerNet(out_channels=out_channels)

        # Try to load state dict with strict=False first
        try:
            self.detector.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            print(f"Warning: Failed to load state dict strictly. Error: {e}")
            print("Attempting to load with strict=False...")
            self.detector.load_state_dict(state_dict, strict=False)

        self.detector.to(self.device)
        self.detector.eval()

        # Initialize tracking parameters
        self.xy_coordinates = np.array([[None, None]])
        self.bounces_indices = []
        self.threshold_dist = 100
        self.video_width = None
        self.video_height = None
        self.model_input_width = 640
        self.model_input_height = 360

        self.current_frame = None
        self.last_frame = None
        self.before_last_frame = None


    # def detect_ball(self, frame):
    #     """
    #     After receiving 3 consecutive frames, the ball will be detected using TrackNet model
    #     :param frame: current frame
    #     """
    #     # Save frame dimensions
    #     if self.video_width is None:
    #         self.video_width = frame.shape[1]
    #         self.video_height = frame.shape[0]
    #     self.last_frame = self.before_last_frame
    #     self.before_last_frame = self.current_frame
    #     self.current_frame = frame.copy()

    #     # detect only in 3 frames were given
    #     if self.last_frame is not None:
    #         # combine the frames into 1 input tensor
    #         frames = combine_three_frames(self.current_frame, self.before_last_frame, self.last_frame,
    #                                       self.model_input_width, self.model_input_height)
    #         frames = (torch.from_numpy(frames) / 255).to(self.device)
    #         # Inference (forward pass)
    #         x, y = self.detector.inference(frames)
    #         if x is not None:
    #             # Rescale the indices to fit frame dimensions
    #             x = x * (self.video_width / self.model_input_width)
    #             y = y * (self.video_height / self.model_input_height)

    #             # Check distance from previous location and remove outliers
    #             if self.xy_coordinates[-1][0] is not None:
    #                 if np.linalg.norm(np.array([x,y]) - self.xy_coordinates[-1]) > self.threshold_dist:
    #                     x, y = None, None
    #         self.xy_coordinates = np.append(self.xy_coordinates, np.array([[x, y]]), axis=0)
    def detect_ball(self, frame):
        """
        After receiving 3 consecutive frames, the ball will be detected using TrackNet model
        :param frame: current frame
        """
        # Save frame dimensions
        if self.video_width is None:
            self.video_width = frame.shape[1]
            self.video_height = frame.shape[0]

        self.last_frame = self.before_last_frame
        self.before_last_frame = self.current_frame
        self.current_frame = frame.copy()

        # Detect only if 3 frames are available
        if self.last_frame is not None:
            # Combine the frames into 1 input tensor
            frames = combine_three_frames(
                self.current_frame,
                self.before_last_frame,
                self.last_frame,
                self.model_input_width,
                self.model_input_height
            )

            # Convert to PyTorch tensor and move to device
            frames = torch.from_numpy(frames).float().to(self.device) / 255.0
            frames = frames.unsqueeze(0)  # Add batch dimension

            # Inference (forward pass)
            with torch.no_grad():
                output = self.detector(frames)
                output = self.detector.softmax(output)
                output = output.reshape(-1, self.model_input_height, self.model_input_width)
                output = output.argmax(dim=0).detach().cpu().numpy()
                if self.detector.out_channels == 2:
                    output *= 255

                x, y = self.detector.get_center_ball(output)

            if x is not None:
                # Rescale the indices to fit frame dimensions
                x = x * (self.video_width / self.model_input_width)
                y = y * (self.video_height / self.model_input_height)

                # Check distance from the previous location and remove outliers
                if self.xy_coordinates[-1][0] is not None:
                    if np.linalg.norm(np.array([x, y]) - self.xy_coordinates[-1]) > self.threshold_dist:
                        x, y = None, None

            # Append the coordinates to the list
            self.xy_coordinates = np.append(self.xy_coordinates, np.array([[x, y]]), axis=0)

    def mark_positions(self, frame, mark_num=4, frame_num=None, ball_color='yellow'):
        """
        Mark the last 'mark_num' positions of the ball in the frame
        :param frame: the frame we mark the positions in
        :param mark_num: number of previous detection to mark
        :param frame_num: current frame number
        :param ball_color: color of the marks
        :return: the frame with the ball annotations
        """
        bounce_i = None
        # if frame number is not given, use the last positions found
        if frame_num is not None:
            q = self.xy_coordinates[frame_num-mark_num+1:frame_num+1, :]
            for i in range(frame_num - mark_num + 1, frame_num + 1):
                if i in self.bounces_indices:
                    bounce_i = i - frame_num + mark_num - 1
                    break
        else:
            q = self.xy_coordinates[-mark_num:, :]
        pil_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(pil_image)
        # Mark each position by a circle
        for i in range(q.shape[0]):
            if q[i, 0] is not None:
                draw_x = q[i, 0]
                draw_y = q[i, 1]
                bbox = (draw_x - 2, draw_y - 2, draw_x + 2, draw_y + 2)
                draw = ImageDraw.Draw(pil_image)
                if bounce_i is not None and i == bounce_i:
                    draw.ellipse(bbox, outline='red')
                else:
                    draw.ellipse(bbox, outline=ball_color)

            # Convert PIL image format back to opencv image format
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return frame

    def calculate_ball_positions(self):
        return self.xy_coordinates

    #---------------------------------------------------------xav--------------------------------------------------
    def calculate_ball_position_top_view(self, court_detector):
        inv_mats = court_detector.game_warp_matrix
        xy_coordinates_top_view = []

        # Use the last available matrix if we run out of matrices
        for i, pos in enumerate(self.xy_coordinates):
            matrix_idx = min(i, len(inv_mats) - 1) if inv_mats else 0

            if not inv_mats:
                # If no matrices available, return the original coordinates
                xy_coordinates_top_view.append(pos)
                continue

            if pos[0] is None:
                ball_pos = np.array([100.25, 100.89]).reshape((1, 1, 2))
            else:
                ball_pos = np.array([pos[0], pos[1]]).reshape((1, 1, 2))

            try:
                ball_court_pos = cv2.perspectiveTransform(ball_pos, inv_mats[matrix_idx]).reshape(-1)
                xy_coordinates_top_view.append(ball_court_pos)
            except Exception as e:
                print(f"Error transforming ball position at frame {i}: {e}")
                xy_coordinates_top_view.append(np.array([None, None]))

        return xy_coordinates_top_view
    #---------------------------------------------------------fin xav----------------------------------------------


    def show_y_graph(self, player_1_boxes, player_2_boxes):

        player_1_centers = np.array([center_of_box(box) for box in player_1_boxes])
        player_1_y_values = player_1_centers[:, 1] - np.array([(box[3] - box[1]) // 4 for box in player_1_boxes])

        player_2_centers = np.array([center_of_box(box) if box[0] is not None else [None, None] for box in player_2_boxes])
        player_2_y_values = player_2_centers[:, 1]

        y_values = self.xy_coordinates[:, 1].copy()
        x_values = self.xy_coordinates[:, 0].copy()

        plt.figure()
        plt.scatter(range(len(y_values)), y_values, marker='o', label='Ball', color='blue')
        plt.plot(range(len(player_1_y_values)), player_1_y_values, color='r', marker='o', linestyle='-', label='Player 1')
        plt.plot(range(len(player_2_y_values)), player_2_y_values, color='g', marker='o', linestyle='-', label='Player 2')

        plt.xlabel('Frame Index')
        plt.ylabel('Y-Index Position')
        plt.title('Ball and Players Y-Index Positions Over Frames')
        plt.legend()
        plt.savefig('graph_video.jpg')


if __name__ == "__main__":
    ball_detector = BallDetector('saved states/tracknet_weights_2_classes.pth')
    cap = cv2.VideoCapture('../videos/vid1.mp4')
    # get videos properties
    fps, length, v_width, v_height = get_video_properties(cap)

    frame_i = 0
    while True:
        ret, frame = cap.read()
        frame_i += 1
        if not ret:
            break

        ball_detector.detect_ball(frame)


    cap.release()
    cv2.destroyAllWindows()

    from scipy.interpolate import interp1d

    y_values = ball_detector.xy_coordinates[:,1]

    new = signal.savgol_filter(y_values, 3, 2)

    x = np.arange(0, len(new))
    indices = [i for i, val in enumerate(new) if np.isnan(val)]
    x = np.delete(x, indices)
    y = np.delete(new, indices)
    f = interp1d(x, y, fill_value="extrapolate")
    f2 = interp1d(x, y, kind='cubic', fill_value="extrapolate")
    xnew = np.linspace(0, len(y_values), num=len(y_values), endpoint=True)
    plt.plot(np.arange(0, len(new)), new, 'o',xnew,
             f2(xnew), '-r')
    plt.legend(['data', 'inter'], loc='best')
    plt.show()

    positions = f2(xnew)
    peaks, _ = find_peaks(positions, distance=30)
    a = np.diff(peaks)
    plt.plot(positions)
    plt.plot(peaks, positions[peaks], "x")
    plt.show()
