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
import torch.nn.functional as F


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
    _instance = None

    def __new__(cls, weights_path=None, out_channels=2):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            print("Initializing BallDetector...")
            cls._instance.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            cls._instance.detector = BallTrackerNet(out_channels=out_channels).to(cls._instance.device)

            if weights_path:
                print("Loading weights from:", weights_path)
                try:
                    checkpoint = torch.load(weights_path, map_location=cls._instance.device)
                    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                        state_dict = checkpoint['model_state']
                    else:
                        state_dict = checkpoint
                    cls._instance.detector.load_state_dict(state_dict, strict=False)
                    print("Weights loaded successfully")
                except Exception as e:
                    print(f"Error loading weights: {str(e)}")
                    raise

            cls._instance.detector.eval()
            cls._instance.frame_buffer = []
            cls._instance.buffer_size = 3
            cls._instance.xy_coordinates = []
            print("BallDetector initialized successfully")

        return cls._instance

    def __init__(self, weights_path=None, out_channels=2):
        """This init method is called after __new__ but we don't need to do anything here"""
        pass

    def preprocess_frame(self, frame):
        """Convert frame to tensor and normalize"""
        if frame is None:
            return None
        try:
            frame = cv2.resize(frame, (640, 360))
            frame = torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0
            return frame
        except Exception as e:
            print(f"Error in preprocess_frame: {str(e)}")
            return None

    def detect_ball(self, frame):
        """
        Detect ball in the current frame using a buffer of 3 frames
        Returns: x, y coordinates of the ball as Python scalars
        """
        try:
            # Preprocess current frame
            processed_frame = self.preprocess_frame(frame)
            if processed_frame is None:
                self.xy_coordinates.append([None, None])
                return None, None

            # Update frame buffer
            self.frame_buffer.append(processed_frame)
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)

            # If we don't have enough frames yet, return None
            if len(self.frame_buffer) < self.buffer_size:
                self.xy_coordinates.append([None, None])
                return None, None

            # Stack frames into single input tensor
            input_tensor = torch.cat(self.frame_buffer, dim=0).unsqueeze(0)
            input_tensor = input_tensor.to(self.device)

            # Get model prediction
            with torch.no_grad():
                output = self.detector(input_tensor)

            if output is None:
                self.xy_coordinates.append([None, None])
                return None, None

            # Convert output to probability map using softmax
            prob_map = F.softmax(output[0], dim=0)[1].cpu().numpy()

            # Find ball center using maximum probability
            max_prob = prob_map.max()
            if max_prob < 0.5:  # Confidence threshold
                self.xy_coordinates.append([None, None])
                return None, None

            # Get coordinates of maximum probability
            y, x = np.unravel_index(np.argmax(prob_map), prob_map.shape)

            # Convert to Python scalars and store
            x, y = float(x), float(y)
            self.xy_coordinates.append([x, y])
            return x, y

        except Exception as e:
            print(f"Error in detect_ball: {str(e)}")
            self.xy_coordinates.append([None, None])
            return None, None

    def calculate_ball_positions(self):
        """Return list of ball positions"""
        return self.xy_coordinates

    def calculate_ball_position_top_view(self, court_detector):
        """Calculate ball positions in top view"""
        try:
            if not self.xy_coordinates:
                return [[None, None]]

            positions = []
            for pos in self.xy_coordinates:
                if pos[0] is None or pos[1] is None:
                    positions.append([None, None])
                    continue

                if not court_detector.game_warp_matrix:
                    positions.append([None, None])
                    continue

                # Use the last available transformation matrix
                matrix_idx = min(len(court_detector.game_warp_matrix) - 1, len(positions))
                matrix = court_detector.game_warp_matrix[matrix_idx]

                # Transform point
                point = np.array([[pos]], dtype=np.float32)
                transformed_point = cv2.perspectiveTransform(point, matrix)

                if transformed_point is not None:
                    positions.append([float(transformed_point[0][0][0]), float(transformed_point[0][0][1])])
                else:
                    positions.append([None, None])

            return positions

        except Exception as e:
            print(f"Error in calculate_ball_position_top_view: {str(e)}")
            return [[None, None]]

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
