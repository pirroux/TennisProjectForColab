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
from torch.serialization import add_safe_globals
from numpy.core.multiarray import scalar
from numpy import dtype


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

    def __new__(cls, weights_path):
        if cls._instance is None:
            print("Initializing BallDetector...")
            cls._instance = super(BallDetector, cls).__new__(cls)
            cls._instance.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            cls._instance.detector = BallTrackerNet(out_channels=2).to(cls._instance.device)
            cls._instance.confidence_threshold = 0.5  # Add confidence threshold
            cls._instance.frame_buffer = []
            cls._instance.buffer_size = 3
            cls._instance.xy_coordinates = []

            print("Loading weights from:", weights_path)
            try:
                add_safe_globals([scalar, dtype])
                checkpoint = torch.load(weights_path, weights_only=True)
            except Exception as e:
                print(f"Warning: Failed to load with weights_only=True, attempting without: {str(e)}")
                checkpoint = torch.load(weights_path, weights_only=False)

            if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
                state_dict = checkpoint['model_state']
            else:
                state_dict = checkpoint

            try:
                cls._instance.detector.load_state_dict(state_dict, strict=True)
            except RuntimeError as e:
                print(f"Warning: Strict loading failed, attempting non-strict loading: {str(e)}")
                cls._instance.detector.load_state_dict(state_dict, strict=False)

            cls._instance.detector.eval()
            print("Weights loaded successfully")

        return cls._instance

    def __init__(self, weights_path=None, out_channels=2):
        """This init method is called after __new__ but we don't need to do anything here"""
        pass

    def preprocess_frame(self, frame):
        """Preprocess a single frame for ball detection."""
        try:
            # Convert frame to float and normalize
            frame = frame.astype(np.float32) / 255.0

            # Convert to channels first format (C, H, W)
            frame = np.transpose(frame, (2, 0, 1))

            # Convert to tensor and add batch dimension
            frame_tensor = torch.from_numpy(frame).unsqueeze(0)

            # Move to appropriate device
            frame_tensor = frame_tensor.to(self.device)

            return frame_tensor

        except Exception as e:
            print(f"Error in frame preprocessing: {str(e)}")
            return None

    def detect_ball(self, frame):
        """Detect ball in a single frame."""
        try:
            # Preprocess frame
            frame_tensor = self.preprocess_frame(frame)

            # Get model output
            with torch.no_grad():
                output = self.detector(frame_tensor)

            # Convert output to probability map
            prob_map = torch.nn.functional.softmax(output[0], dim=0)[1].cpu().numpy()

            # Find ball position
            max_prob = float(np.max(prob_map))  # Convert to Python scalar safely
            if max_prob > self.confidence_threshold:
                # Get indices of maximum probability
                max_idx = np.unravel_index(np.argmax(prob_map), prob_map.shape)
                y_idx, x_idx = max_idx

                # Convert to original frame coordinates
                h, w = frame.shape[:2]
                x = float(x_idx * w / prob_map.shape[1])  # Convert to Python scalar safely
                y = float(y_idx * h / prob_map.shape[0])  # Convert to Python scalar safely

                print(f"Ball detected at ({x:.1f}, {y:.1f}) with confidence {max_prob:.3f}")
                return x, y

            return None, None

        except Exception as e:
            print(f"Error in ball detection: {str(e)}")
            return None, None

    def calculate_ball_positions(self):
        """Return all detected ball positions."""
        return self.xy_coordinates

    def calculate_ball_position_top_view(self, court_detector):
        """Calculate ball positions in top view"""
        try:
            if not self.xy_coordinates:
                return [[None, None]]

            positions = []
            for i, pos in enumerate(self.xy_coordinates):
                try:
                    # Skip if no ball detected
                    if pos[0] is None or pos[1] is None:
                        positions.append([None, None])
                        continue

                    # Skip if no court detection
                    if not court_detector.game_warp_matrix:
                        positions.append([None, None])
                        continue

                    # Get appropriate transformation matrix
                    matrix_idx = min(len(court_detector.game_warp_matrix) - 1, i)
                    matrix = court_detector.game_warp_matrix[matrix_idx]

                    # Prepare point for transformation
                    point = np.array([[[float(pos[0]), float(pos[1])]]], dtype=np.float32)

                    # Transform point
                    transformed_point = cv2.perspectiveTransform(point, matrix)

                    if transformed_point is not None and transformed_point.size > 0:
                        x = float(transformed_point[0, 0, 0])
                        y = float(transformed_point[0, 0, 1])
                        positions.append([x, y])
                    else:
                        positions.append([None, None])

                except Exception as e:
                    print(f"Error transforming point at index {i}: {str(e)}")
                    positions.append([None, None])

            return positions

        except Exception as e:
            print(f"Error in calculate_ball_position_top_view: {str(e)}")
            return [[None, None]] * len(self.xy_coordinates)

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

    def preprocess_frames(self, frames):
        """
        Preprocess frames for ball detection.
        Args:
            frames: numpy array of shape [batch_size, height, width, channels]
        Returns:
            torch tensor of shape [batch_size, channels, height, width]
        """
        try:
            # Convert to float and normalize
            frames = frames.astype(np.float32) / 255.0

            # Convert to torch tensor
            frames = torch.from_numpy(frames)

            # Reshape if necessary [batch_size, height, width, channels] -> [batch_size, channels, height, width]
            if frames.shape[-1] == 3:
                frames = frames.permute(0, 3, 1, 2)

            # Move to device
            frames = frames.to(self.device)

            print(f"Preprocessed frames shape: {frames.shape}")
            return frames

        except Exception as e:
            print(f"Error in preprocessing frames: {str(e)}")
            return None

    def process_video_frames(self, frames):
        """Process a batch of frames for ball detection."""
        if len(frames.shape) != 4:  # batch, height, width, channels
            raise ValueError(f"Expected 4D input (batch,H,W,C), got shape {frames.shape}")

        batch_size = frames.shape[0]
        print(f"Processing batch of {batch_size} frames")

        try:
            # Preprocess batch
            frames_tensor = self.preprocess_frames_batch(frames)

            # Get predictions
            with torch.no_grad():
                output = self.detector(frames_tensor)
                prob_maps = torch.softmax(output, dim=1).cpu().numpy()

            # Process each frame's predictions
            for i in range(batch_size):
                prob_map = prob_maps[i, 1]  # Get ball probability map
                if prob_map.max() > self.confidence_threshold:
                    # Get coordinates of maximum probability
                    y_idx, x_idx = np.unravel_index(prob_map.argmax(), prob_map.shape)
                    # Scale coordinates back to original size
                    scale_y = frames.shape[1] / prob_map.shape[0]
                    scale_x = frames.shape[2] / prob_map.shape[1]
                    x = float(x_idx * scale_x)
                    y = float(y_idx * scale_y)
                    self.xy_coordinates.append([x, y])
                else:
                    self.xy_coordinates.append([None, None])

            return self.xy_coordinates[-batch_size:]

        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            return [[None, None]] * batch_size

    def preprocess_frames_batch(self, frames):
        """Preprocess a batch of frames."""
        # Convert to float and normalize
        frames = frames.astype(np.float32) / 255.0

        # Transpose to (batch, channels, height, width)
        frames = np.transpose(frames, (0, 3, 1, 2))

        # Convert to tensor
        frames_tensor = torch.from_numpy(frames).to(self.device)

        return frames_tensor


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
