import os
import time
import sys
import json


# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
import sys
sys.path.append('/content/TennisProject/src')
from src.detection import DetectionModel, center_of_box
from src.pose import PoseExtractor
from src.smooth import Smooth
from src.ball_detection import BallDetector
from src.my_statistics import Statistics
from src.stroke_recognition import ActionRecognition
from src.utils import get_video_properties, get_dtype, get_stickman_line_connection
from src.court_detection import CourtDetector

def get_stroke_predictions(video_path, stroke_recognition, strokes_frames, player_boxes):
    """
    Get the stroke prediction for all sections where we detected a stroke
    """
    predictions = {}
    cap = cv2.VideoCapture(video_path)
    fps, length, width, height = get_video_properties(cap)
    video_length = 2
    # For each stroke detected trim video part and predict stroke
    predicted_strokes = []
    for frame_num in strokes_frames:
        # Trim the video (only relevant frames are taken)
        starting_frame = max(0, frame_num - int(video_length * fps * 2 / 3))
        cap.set(1, starting_frame)
        i = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            stroke_recognition.add_frame(frame, player_boxes[starting_frame + i])
            i += 1
            if i == int(video_length * fps):
                break
        # predict the stroke
        probs, stroke = stroke_recognition.predict_saved_seq()
        predictions[frame_num] = {'probs': probs, 'stroke': stroke}
        predicted_strokes.append(stroke)
    cap.release()
    print(predicted_strokes)
    return predictions, predicted_strokes


from scipy import signal
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def find_strokes_indices(player_1_boxes, player_2_boxes, ball_positions, skeleton_df, verbose=0):
    """
    Detect strokes frames using location of the ball and players
    """
    ball_x, ball_y = ball_positions[:, 0], ball_positions[:, 1]
    smooth_x = signal.savgol_filter(ball_x, 3, 2)
    smooth_y = signal.savgol_filter(ball_y, 3, 2)

    # Ball position interpolation
    x = np.arange(0, len(smooth_y))
    indices = [i for i, val in enumerate(smooth_y) if np.isnan(val)]
    x = np.delete(x, indices)
    y1 = np.delete(smooth_y, indices)
    y2 = np.delete(smooth_x, indices)

    # Check if the arrays are empty before interpolation
    if len(x) == 0 or len(y1) == 0:
        print("Error: x or y1 arrays are empty.")
        return None  # or handle the error appropriately

    ball_f2_y = interp1d(x, y1, kind='cubic', fill_value="extrapolate")
    ball_f2_x = interp1d(x, y2, kind='cubic', fill_value="extrapolate")
    xnew = np.linspace(0, len(ball_y), num=len(ball_y), endpoint=True)

    if verbose:
        plt.plot(np.arange(0, len(smooth_y)), smooth_y, 'o', xnew,
                 ball_f2_y(xnew), '-r')
        plt.legend(['data', 'inter'], loc='best')
        plt.show()

    # Player 2 position interpolation
    player_2_centers = np.array([center_of_box(box) for box in player_2_boxes])
    player_2_x, player_2_y = player_2_centers[:, 0], player_2_centers[:, 1]
    player_2_x = signal.savgol_filter(player_2_x, 3, 2)
    player_2_y = signal.savgol_filter(player_2_y, 3, 2)
    x = np.arange(0, len(player_2_y))
    indices = [i for i, val in enumerate(player_2_y) if np.isnan(val)]
    x = np.delete(x, indices)
    y1 = np.delete(player_2_y, indices)
    y2 = np.delete(player_2_x, indices)

    # Check if the arrays are empty before interpolation
    if len(x) == 0 or len(y1) == 0:
        print("Error: x or y1 arrays for player 2 are empty.")
        return None  # or handle the error appropriately

    player_2_f_y = interp1d(x, y1, fill_value="extrapolate")
    player_2_f_x = interp1d(x, y2, fill_value="extrapolate")
    xnew = np.linspace(0, len(player_2_y), num=len(player_2_y), endpoint=True)

    if verbose:
        plt.plot(np.arange(0, len(player_2_y)), player_2_y, 'o', xnew, player_2_f_y(xnew), '--g')
        plt.legend(['data', 'inter_cubic', 'inter_lin'], loc='best')
        plt.show()

    coordinates = ball_f2_y(xnew)
    # Find all peaks of the ball y index
    peaks, _ = find_peaks(coordinates)
    if verbose:
        plt.plot(coordinates)
        plt.plot(peaks, coordinates[peaks], "x")
        plt.show()

    neg_peaks, _ = find_peaks(coordinates * -1)
    if verbose:
        plt.plot(coordinates)
        plt.plot(neg_peaks, coordinates[neg_peaks], "x")
        plt.show()

    # Get bottom player wrists positions
    left_wrist_index = 9
    right_wrist_index = 10
    skeleton_df = skeleton_df.fillna(-1)
    left_wrist_pos = skeleton_df.iloc[:, [left_wrist_index, left_wrist_index + 15]].values
    right_wrist_pos = skeleton_df.iloc[:, [right_wrist_index, right_wrist_index + 15]].values

    dists = []
    # Calculate dist between ball and bottom player
    for i, player_box in enumerate(player_1_boxes):
        box_dist = np.inf  # Initialize box_dist
        if player_box[0] is not None:
            player_center = center_of_box(player_box)
            ball_pos = np.array([ball_f2_x(i), ball_f2_y(i)])
            right_wrist_dist, left_wrist_dist = np.inf, np.inf
            if right_wrist_pos[i, 0] > 0:
                right_wrist_dist = np.linalg.norm(right_wrist_pos[i, :] - ball_pos)
            if left_wrist_pos[i, 0] > 0:
                left_wrist_dist = np.linalg.norm(left_wrist_pos[i, :] - ball_pos)
            box_dist = np.linalg.norm(player_center - ball_pos)  # Assign box_dist here
        dists.append(min(box_dist, right_wrist_dist, left_wrist_dist))
    dists = np.array(dists)

    dists2 = []
    # Calculate dist between ball and top player
    for i in range(len(player_2_centers)):
        ball_pos = np.array([ball_f2_x(i), ball_f2_y(i)])
        box_center = np.array([player_2_f_x(i), player_2_f_y(i)])
        box_dist = np.linalg.norm(box_center - ball_pos)
        dists2.append(box_dist)
    dists2 = np.array(dists2)

    strokes_1_indices = []
    # Find stroke for bottom player by thresholding the dists
    for peak in peaks:
        player_box_height = max(player_1_boxes[peak][3] - player_1_boxes[peak][1], 130)
        if dists[peak] < (player_box_height * 4 / 5):
            strokes_1_indices.append(peak)

    strokes_2_indices = []
    # Find stroke for top player by thresholding the dists
    for peak in neg_peaks:
        if dists2[peak] < 100:
            strokes_2_indices.append(peak)

    # Assert the diff between to consecutive strokes is below some threshold
    while True:
        diffs = np.diff(strokes_1_indices)
        to_del = []
        for i, diff in enumerate(diffs):
            if diff < 40:
                max_in = np.argmax([dists[strokes_1_indices[i]], dists[strokes_1_indices[i + 1]]])
                to_del.append(i + max_in)

        strokes_1_indices = np.delete(strokes_1_indices, to_del)
        if len(to_del) == 0:
            break

    # Assert the diff between to consecutive strokes is below some threshold
    while True:
        diffs = np.diff(strokes_2_indices)
        to_del = []
        for i, diff in enumerate(diffs):
            if diff < 40:
                max_in = np.argmax([dists2[strokes_2_indices[i]], dists2[strokes_2_indices[i + 1]]])
                to_del.append(i + max_in)

        strokes_2_indices = np.delete(strokes_2_indices, to_del)
        if len(to_del) == 0:
            break

    # Assume bounces frames are all the other peaks in the y index graph
    bounces_indices = [x for x in peaks if x not in strokes_1_indices]
    if verbose:
        plt.figure()
        plt.plot(coordinates)
        plt.plot(strokes_1_indices, coordinates[strokes_1_indices], "or")
        plt.plot(strokes_2_indices, coordinates[strokes_2_indices], "og")
        plt.legend(['data', 'player 1 strokes', 'player 2 strokes'], loc='best')
        plt.show()

    return strokes_1_indices, strokes_2_indices, bounces_indices, player_2_f_x, player_2_f_y


def mark_player_box(frame, boxes, frame_num):
    box = boxes[frame_num]
    if box[0] is not None:
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), [255, 0, 255], 2)
    return frame


def mark_skeleton(skeleton_df, img, img_no_frame, frame_number):
    """
    Mark the skeleton of the bottom player on the frame
    """
    # landmarks colors
    circle_color, line_color = (0, 0, 255), (255, 0, 0)
    stickman_pairs = get_stickman_line_connection()

    skeleton_df = skeleton_df.fillna(-1)
    values = np.array(skeleton_df.values[frame_number], int)
    points = list(zip(values[5:17], values[22:]))
    # draw key points
    for point in points:
        if point[0] >= 0 and point[1] >= 0:
            xy = tuple(np.array([point[0], point[1]], int))
            cv2.circle(img, xy, 2, circle_color, 2)
            cv2.circle(img_no_frame, xy, 2, circle_color, 2)

    # Draw stickman
    for pair in stickman_pairs:
        partA = pair[0] - 5
        partB = pair[1] - 5
        if points[partA][0] >= 0 and points[partA][1] >= 0 and points[partB][0] >= 0 and points[partB][1] >= 0:
            cv2.line(img, points[partA], points[partB], line_color, 1, lineType=cv2.LINE_AA)
            cv2.line(img_no_frame, points[partA], points[partB], line_color, 1, lineType=cv2.LINE_AA)
    return img, img_no_frame

def add_data_to_video(video_path, output_folder, output_file, minimap_path,
                      players_detector=None, ball_detector=None, strokes_predictions=None,
                      skeleton_df=None, statistics=None, with_frame=0, show_video=False,
                      p1=None, p2=None, f_x=None, f_y=None):

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        print(f"Creating output folder: {output_folder}")
        os.makedirs(output_folder)

    # Initialize the video capture object for the main video and minimap video
    cap = cv2.VideoCapture(video_path)
    small_video = cv2.VideoCapture(minimap_path)

    # Check if video files opened successfully
    if not cap.isOpened():
        print(f"Error: Couldn't open video file: {video_path}")
        return None  # Return None to handle failure gracefully
    if not small_video.isOpened():
        print(f"Error: Couldn't open minimap video file: {minimap_path}")
        return None  # Return None to handle failure gracefully

    # Get frame dimensions from the main video
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and output video format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_folder, output_file + '.mp4'), fourcc, 30, (frame_width, frame_height))

    # Initialize frame counters
    frame_number = 0

    while True:
        ret, img = cap.read()
        ret_small, img_small = small_video.read()  # Read minimap frame

        if not ret or not ret_small:
            print(f"Error: Could not read frame {frame_number}. Ending processing.")
            break

        # Debug: Check frame shape
        print(f"Processing frame {frame_number}...")
        print(f"Frame shape: {img.shape}")

        # Here you would do your processing (e.g., overlay court, players, etc.)
        # Assuming img_no_frame is processed here (without the frame overlay)
        img_no_frame = img.copy()  # Example placeholder, replace with actual processing

        # Resize the minimap image if needed (e.g., to fit it into the corner of the frame)
        img_small_resized = cv2.resize(img_small, (200, 200))  # Resize minimap to fit corner

        # Overlay minimap onto the main video frame
        img[0:200, 0:200] = img_small_resized

        # Create the final frame based on the `with_frame` argument
        if with_frame == 0:
            final_frame = img_no_frame  # No frame overlay
        elif with_frame == 1:
            final_frame = img  # Full frame with overlay
        else:
            final_frame = np.concatenate([img, img_no_frame], 1)  # Concatenate images side by side

        # Debugging: Check final frame shape
        if final_frame is not None:
            print(f"Final frame shape: {final_frame.shape}")
        else:
            print("Error: final_frame is None!")

        # Write the processed frame to the output video
        out.write(final_frame)

        # Optionally show the video during processing
        if show_video:
            cv2.imshow('Output', final_frame)
            if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
                break

        frame_number += 1

    # Release the video capture and writer objects
    cap.release()
    small_video.release()
    out.release()

    # Close the OpenCV window if it was opened
    if show_video:
        cv2.destroyAllWindows()

    print(f"Video saved to {os.path.join(output_folder, output_file + '.mp4')}")
    return "Processing Complete"



def create_top_view(court_detector, detection_model, ball_detector, fps='30'):
    """
    Creates top view video of the gameplay
    """

    court = court_detector.court_reference.court.copy()
    court = cv2.line(court, *court_detector.court_reference.net, (255, 255, 255), thickness=5)
    v_width, v_height = court.shape[::-1]
    court = cv2.cvtColor(court, cv2.COLOR_GRAY2BGR)
    out = cv2.VideoWriter('output/top_view.mp4',
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (v_width, v_height))
    # players and ball location on court
    smoothed_1, smoothed_2 = detection_model.calculate_feet_positions(court_detector)
    ball_positions = ball_detector.calculate_ball_positions()       ## ---------------------xav----------------------------------
    ball_position_top_view = ball_detector.calculate_ball_position_top_view(court_detector)  #---------------------------------xav--------------------

    for feet_pos_1, feet_pos_2, ball_pos in zip(smoothed_1, smoothed_2, ball_position_top_view):
        frame = court.copy()
        frame = cv2.circle(frame, (int(feet_pos_1[0]), int(feet_pos_1[1])), 10, (255, 105, 180), 15)
        if feet_pos_2[0] is not None:
            frame = cv2.circle(frame, (int(feet_pos_2[0]), int(feet_pos_2[1])), 10, (255, 105, 180), 15)
        if ball_pos[0] is not None:
            frame = cv2.circle(frame, (int(ball_pos[0]), int(ball_pos[1])), 10, (0, 255, 255), 15)
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()

def video_process(video_path, show_video=False, include_video=True,
                  stickman=True, stickman_box=True, output_file='output',
                  output_folder='output', create_minimap=True):
    """
    Process tennis video to detect court, ball, player 1's position, pose and strokes.
    Creates a clean output video showing player detection box and pose.
    """
    dtype = get_dtype()

    # Initialize models
    detection_model = DetectionModel()  # YOLOv11 for player detection
    pose_extractor = PoseExtractor(person_num=1, box=stickman_box, dtype=dtype) if stickman else None
    stroke_recognition = ActionRecognition('storke_classifier_weights.pth')
    ball_detector = BallDetector('saved states/tracknet_weights_2_classes.pth', out_channels=2)
    court_detector = CourtDetector()

    # Load video
    video = cv2.VideoCapture(video_path)
    fps, length, v_width, v_height = get_video_properties(video)

    # Initialize video writer
    if include_video:
        output_video_path = os.path.join(output_folder, f"{output_file}_output_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (v_width, v_height))

    frame_i = 0
    total_time = 0
    stroke_frames = []
    stroke_predictions = []
    ball_positions = []
    player_positions = []

    while True:
        ret, frame = video.read()
        if ret:
            start_time = time.time()
            analysis_frame = frame.copy()  # Create a copy for analysis

            # Court detection (only first frame)
            if frame_i == 1:
                court_detector.detect(analysis_frame)
                court_detection_time = time.time() - start_time
                start_time = time.time()

            # Track court (on analysis frame)
            court_detector.track_court(analysis_frame)

            # Detect player 1
            detection_model.detect_player_1(analysis_frame, court_detector)
            if len(detection_model.player_1_boxes) > 0:
                # Draw player detection box
                box = detection_model.player_1_boxes[-1]
                cv2.rectangle(frame,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            (0, 255, 0), 2)  # Green box
                player_positions.append(center_of_box(box))
            else:
                player_positions.append([None, None])

            # Extract pose if enabled
            if stickman and len(detection_model.player_1_boxes) > 0:
                stickman_frame = pose_extractor.extract_pose(frame, detection_model.player_1_boxes)
                frame = cv2.addWeighted(frame, 1, stickman_frame, 0.5, 0)

            # Detect ball (on analysis frame)
            ball_detector.detect_ball(analysis_frame)
            ball_positions.append(ball_detector.xy_coordinates[-1] if len(ball_detector.xy_coordinates) > 0 else [None, None])

            # Detect stroke
            if len(detection_model.player_1_boxes) > 0:
                probs, stroke = stroke_recognition.add_frame(analysis_frame, detection_model.player_1_boxes[-1])
                if stroke is not None:
                    stroke_frames.append(frame_i)
                    stroke_predictions.append(stroke)

            total_time += (time.time() - start_time)

            # Write frame with player box and pose overlay to video
            if include_video:
                out_video.write(frame)

            if show_video:
                cv2.imshow('Processed Video', frame)
            frame_i += 1
        else:
            break

    # Create top view video if requested
    if create_minimap:
        create_top_view(court_detector, detection_model, ball_detector, fps)

    # Count strokes by type
    stroke_counts = {}
    for stroke in stroke_predictions:
        stroke_counts[stroke] = stroke_counts.get(stroke, 0) + 1

    # Calculate total distance
    total_distance = calculate_player_distance(player_positions)

    # Create results dictionary
    dico = {
        'stroke': stroke_predictions,
        'stroke_counts': stroke_counts,
        'court_detection_time': court_detection_time,
        'ball_positions': ball_positions,
        'total_frames_analyzed': frame_i,
        'processing_time': total_time,
        'player_distance': total_distance
    }

    # Save results
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(os.path.join(output_folder, 'dico.txt'), 'w') as f:
        json.dump(dico, f, indent=4)

    # Cleanup
    video.release()
    if include_video:
        out_video.release()
    cv2.destroyAllWindows()

    return dico

def calculate_player_distance(positions):
    """
    Calculate total distance traveled by player
    """
    total_distance = 0
    for pos1, pos2 in zip(positions[:-1], positions[1:]):
        if None not in pos1 and None not in pos2:
            total_distance += np.linalg.norm(np.array(pos1) - np.array(pos2))
    return total_distance

def main(video_path):
    s = time.time()
    result_json = video_process(video_path="video_input6.mp4",
                                show_video=False,
                                stickman=True,
                                stickman_box=True,
                                smoothing=True,
                                court=True, top_view=True)
    computation_time = time.time() - s
    print(f'Total computation time: {computation_time:.2f} seconds')
    result_json['Total computation time (s)'] = computation_time
    return result_json

if __name__ == "__main__":
    video_path = "video_input6.mp4"
    result = main(video_path)
    print(result)
