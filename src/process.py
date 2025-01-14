import os
import time
import sys
import json
import torch


# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# Update imports to use correct paths
from src.detection import DetectionModel, center_of_box
from src.pose import PoseExtractor, PoseDetector
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

def interpolate_ball_positions(ball_positions, max_frames_to_interpolate=10):
    """
    Interpolate missing ball positions using linear interpolation.
    Only interpolates if the gap is not too large.
    """
    positions = np.array(ball_positions)

    # Find sequences of None values
    none_sequences = []
    start_idx = None

    for i in range(len(positions)):
        if positions[i][0] is None:
            if start_idx is None:
                start_idx = i
        elif start_idx is not None:
            none_sequences.append((start_idx, i))
            start_idx = None

    # Handle the case where the sequence ends with None
    if start_idx is not None:
        none_sequences.append((start_idx, len(positions)))

    # Interpolate each sequence if it's not too long
    for start, end in none_sequences:
        seq_length = end - start
        if seq_length > max_frames_to_interpolate:
            continue

        if start > 0 and end < len(positions):
            # Get the positions before and after the None sequence
            pos_before = positions[start - 1]
            pos_after = positions[end]

            # Only interpolate if we have valid positions on both sides
            if None not in pos_before and None not in pos_after:
                # Create interpolation for x and y separately
                for i in range(start, end):
                    fraction = (i - start + 1) / (seq_length + 1)
                    x = pos_before[0] + fraction * (pos_after[0] - pos_before[0])
                    y = pos_before[1] + fraction * (pos_after[1] - pos_before[1])
                    positions[i] = [x, y]

    return positions.tolist()

def video_process(video_path, output_path=None, create_minimap=False, save_video=True, save_json=True,
                  show_detection=False, show_pose=False, show_ball=False, show_court=False,
                  show_strokes=False, show_minimap=False):
    """Process video with batched operations for better performance."""
    try:
        start_time = time.time()
        # Initialize models
        ball_detector = BallDetector(os.path.join("/content/drive/MyDrive/Tennis_Weights", "tracknet_weights_2_classes.pth"))
        stroke_recognition = ActionRecognition(os.path.join("/content/drive/MyDrive/Tennis_Weights", "storke_classifier_weights.pth"))

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return {"status": "error", "message": "Could not open video", "computation_time": 0}

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize output paths
        if output_path is None:
            output_dir = 'output'
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, 'output.mp4')
            json_path = os.path.join(output_dir, 'shot_analysis.json')
        else:
            output_dir = os.path.dirname(output_path)
            os.makedirs(output_dir, exist_ok=True)
            json_path = os.path.join(output_dir, 'shot_analysis.json')

        # Initialize video writer if saving is enabled
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Initialize frame buffers and results
        frame_buffer = []  # For storing consecutive frames for ball detection
        stroke_frames = []  # Store frames for stroke recognition
        stroke_results = []  # Store stroke recognition results
        ball_positions = []  # Store all ball positions
        raw_ball_positions = []  # Store uninterpolated ball positions

        # Read first three frames
        frames = []
        for _ in range(3):
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Could not read initial frames")
            frames.append(frame.copy())

        frame_count = 0

        while True:
            if not ret or frame_count >= total_frames:
                break

            # Process ball detection using 3 consecutive frames
            try:
                # Combine three consecutive frames
                combined_frames = np.concatenate(frames, axis=2)  # Concatenate along channel dimension
                combined_frames = combined_frames.transpose((2, 0, 1))  # Convert to channels-first format
                combined_frames = torch.from_numpy(combined_frames).float().unsqueeze(0) / 255.0  # Normalize and add batch dimension

                # Detect ball
                x, y = ball_detector.detect_ball(combined_frames)
                ball_positions.append([x, y])
                raw_ball_positions.append([x, y])
            except Exception as e:
                print(f"Error in ball detection: {str(e)}")
                ball_positions.append([None, None])
                raw_ball_positions.append([None, None])

            # Process stroke recognition
            if frame_count % 32 == 0:  # Process every 32 frames
                try:
                    probs, stroke = stroke_recognition.predict_stroke(frames[1], None)  # Use middle frame
                    if stroke:
                        stroke_results.append({
                            "frame": frame_count,
                            "stroke_type": stroke,
                            "confidence": float(max(probs)) if probs is not None else 0.0
                        })
                except Exception as e:
                    print(f"Error in stroke recognition: {str(e)}")

            # Write frame to output video
            if save_video:
                processed_frame = frames[1].copy()  # Use middle frame
                if show_ball:
                    # Show both detected and interpolated positions
                    if x is not None and y is not None:
                        # Detected ball position in yellow
                        cv2.circle(processed_frame, (int(x), int(y)), 5, (0, 255, 255), -1)
                    elif len(ball_positions) > 0 and ball_positions[-1][0] is not None:
                        # Interpolated position in red (semi-transparent)
                        x, y = ball_positions[-1]
                        cv2.circle(processed_frame, (int(x), int(y)), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
                out.write(processed_frame)

            # Update frame count and progress
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")

            # Shift frames and read next frame
            frames[0] = frames[1]
            frames[1] = frames[2]
            ret, frame = cap.read()
            if ret:
                frames[2] = frame.copy()
            else:
                frames[2] = frames[1].copy()  # Duplicate last frame if no more frames

        # Interpolate missing ball positions
        interpolated_positions = interpolate_ball_positions(ball_positions)

        # Release resources
        cap.release()
        if save_video:
            out.release()
        cv2.destroyAllWindows()

        # Prepare results
        results = {
            "status": "success",
            "ball_positions": {
                "interpolated": interpolated_positions,
                "raw": raw_ball_positions
            },
            "strokes": stroke_results,
            "video_info": {
                "total_frames": total_frames,
                "fps": fps,
                "resolution": f"{width}x{height}"
            },
            "output_path": output_path if save_video else None,
            "computation_time": time.time() - start_time
        }

        # Save results to JSON file if requested
        if save_json:
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Analysis results saved to {json_path}")

        return results

    except Exception as e:
        print(f"Error in video processing: {str(e)}")
        # Clean up resources in case of error
        if 'cap' in locals():
            cap.release()
        if 'out' in locals() and save_video:
            out.release()
        cv2.destroyAllWindows()
        return {
            "status": "error",
            "message": str(e),
            "computation_time": time.time() - start_time
        }

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
    try:
        s = time.time()
        result_json = video_process(
            video_path=video_path,
            output_path='output/output.mp4',
            create_minimap=True,
            save_video=True,
            save_json=True,
            show_detection=True,
            show_pose=True,
            show_ball=False,
            show_court=False,
            show_strokes=True,
            show_minimap=True
        )

        if result_json is None:
            print("Video processing failed")
            return {
                'status': 'error',
                'message': 'Video processing failed',
                'computation_time': time.time() - s
            }

        computation_time = time.time() - s
        print(f'Total computation time: {computation_time:.2f} seconds')
        result_json['Total computation time (s)'] = computation_time
        result_json['status'] = 'success'
        return result_json

    except Exception as e:
        print(f"Error in main: {str(e)}")
        return {
            'status': 'error',
            'message': str(e),
            'computation_time': time.time() - s
        }

def process_batch(frames, detection_model, pose_extractor, ball_detector, court_detector):
    """Process a batch of frames in parallel"""
    results = []

    # Process court detection once for the batch
    court_matrices = [court_detector.track_court(frame) for frame in frames]

    # Detect players in batch
    player_boxes = detection_model.detect_batch(frames, court_detector)

    # Process each frame
    for frame, box, matrix in zip(frames, player_boxes, court_matrices):
        result = {
            'player_pos': None,
            'ball_pos': None,
            'stroke': None
        }

        if box is not None:
            result['player_pos'] = center_of_box(box)

            if pose_extractor:
                stickman_frame = pose_extractor.extract_pose(frame, [box])
                frame = cv2.addWeighted(frame, 1, stickman_frame, 0.5, 0)

        # Ball detection
        ball_detector.detect_ball(frame)
        result['ball_pos'] = (ball_detector.xy_coordinates[-1]
                            if len(ball_detector.xy_coordinates) > 0
                            else [None, None])

        results.append(result)

    return results

if __name__ == "__main__":
    video_path = "/content/drive/Shareddrives/Tennis Shot Identification/Docs de travail/test_yolov11/jono_short2.mp4"
    result = main(video_path)
    print(result)
