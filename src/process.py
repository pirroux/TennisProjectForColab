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
    #print(predicted_strokes)
    return predictions, predicted_strokes


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
        if player_box[0] is not None:
            player_center = center_of_box(player_box)
            ball_pos = np.array([ball_f2_x(i), ball_f2_y(i)])
            box_dist = np.linalg.norm(player_center - ball_pos)
            right_wrist_dist, left_wrist_dist = np.inf, np.inf
            if right_wrist_pos[i, 0] > 0:
                right_wrist_dist = np.linalg.norm(right_wrist_pos[i, :] - ball_pos)
            if left_wrist_pos[i, 0] > 0:
                left_wrist_dist = np.linalg.norm(left_wrist_pos[i, :] - ball_pos)
            dists.append(min(box_dist, right_wrist_dist, left_wrist_dist))
        else:
            dists.append(None)
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


def add_data_to_video(input_video, court_detector, players_detector, ball_detector, strokes_predictions, skeleton_df,
                      statistics,
                      show_video, with_frame, output_folder, output_file, p1, p2, f_x, f_y):
    """
    Creates new videos with pose stickman, face landmarks and blinks counter
    :param input_video: str, path to the input videos
    :param df: DataFrame, data of the pose stickman positions
    :param show_video: bool, display output videos while processing
    :param with_frame: int, output videos includes the original frame with the landmarks
    (0 - only landmarks, 1 - original frame with landmarks, 2 - original frame with landmarks and only
    landmarks (side by side))
    :param output_folder: str, path to output folder
    :param output_file: str, name of the output file
    :return: None
    """

    player1_boxes = players_detector.player_1_boxes
    player2_boxes = players_detector.player_2_boxes

    player1_dists = statistics.bottom_dists_array
    player2_dists = statistics.top_dists_array

    last_frame_distance_player1 = player1_dists[-1] / 100
    last_frame_distance_player2 = player2_dists[-1] / 100


    if skeleton_df is not None:
        skeleton_df = skeleton_df.fillna(-1)

    # Read videos file
    cap = cv2.VideoCapture(input_video)

    # read the small minimap top_view video

    small_video = cv2.VideoCapture('output/top_view.mp4')

    # videos properties
    fps, length, width, height = get_video_properties(cap)

    final_width = width * 2 if with_frame == 2 else width

    # minimpap top_view properties (réduction des la dimension de la vidéo)
    target_width=171
    target_height= 360


    # Video writer
    out = cv2.VideoWriter(os.path.join(output_folder, output_file + '.mp4'),
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (final_width, height))

    # initialize frame counters
    frame_number = 0
    orig_frame = 0
    while True:
        orig_frame += 1
        #print('Creating new videos frame %d/%d  ' % (orig_frame, length), '\r', end='')
        print('\n')
        if not orig_frame % 100:
            print('')
        ret, img = cap.read()
        ret_small, img_small = small_video.read()  # Lire le frame de la vidéo plus petite

        if not ret:
            break

        if not ret_small:
            break

        resized_frame_small = cv2.resize(img_small, (target_width, target_height))


        # initialize frame for landmarks only
        img_no_frame = np.ones_like(img) * 255

        # add Court location
        img = court_detector.add_court_overlay(img, overlay_color=(0, 0, 255), frame_num=frame_number)
        img_no_frame = court_detector.add_court_overlay(img_no_frame, overlay_color=(0, 0, 255), frame_num=frame_number)

        # add players locations
        img = mark_player_box(img, player1_boxes, frame_number)
        img = mark_player_box(img, player2_boxes, frame_number)
        img_no_frame = mark_player_box(img_no_frame, player1_boxes, frame_number)
        img_no_frame = mark_player_box(img_no_frame, player2_boxes, frame_number)

        # add ball location
        img = ball_detector.mark_positions(img, frame_num=frame_number)
        img_no_frame = ball_detector.mark_positions(img_no_frame, frame_num=frame_number, ball_color='black')

        # add pose stickman
        if skeleton_df is not None:
            img, img_no_frame = mark_skeleton(skeleton_df, img, img_no_frame, frame_number)

        # Superposition de la frame de la vidéo plus petite sur le frame de la vidéo principale
        img[8:target_height+8, width-target_width-8:width-8] = resized_frame_small

        # Add stroke prediction
        for i in range(-10, 10):
            if frame_number + i in strokes_predictions.keys():
                '''cv2.putText(img, 'STROKE HIT', (200, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255) if i != 0 else (255, 0, 0), 3)'''

                probs, stroke = strokes_predictions[frame_number + i]['probs'], strokes_predictions[frame_number + i][
                    'stroke']
                cv2.putText(img, 'Forehand - {:.2f}, Backhand - {:.2f}, Service - {:.2f}'.format(*probs),
                            (70, 400),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(img, f'Stroke : {stroke}',
                            (int(player1_boxes[frame_number][0]) - 10, int(player1_boxes[frame_number][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                break
        # Add stroke detected
        for i in range(-5, 10):
            '''if frame_number + i in p1:
                cv2.putText(img, 'Stroke detected', (int(player1_boxes[frame_number][0]) - 10, int(player1_boxes[frame_number][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255) if i != 0 else (255, 0, 0), 2)'''

            if frame_number + i in p2:
                cv2.putText(img, 'Stroke detected',
                            (int(f_x(frame_number)) - 30, int(f_y(frame_number)) - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255) if i != 0 else (255, 0, 0), 2)

        cv2.putText(img, 'Distance: {:.2f} m'.format(player1_dists[frame_number] / 100),
                    (50, 500),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(img, 'Distance: {:.2f} m'.format(player2_dists[frame_number] / 100),
                    (100, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        #distance after last frame
        last_frame_distance_player1 = player1_dists[-1] / 100
        last_frame_distance_player2 = player2_dists[-1] / 100

        # display frame
        if show_video:
            cv2.imshow('Output', img)
            if cv2.waitKey(1) & 0xff == 27:
                cv2.destroyAllWindows()

        # save output videos
        if with_frame == 0:
            final_frame = img_no_frame
        elif with_frame == 1:
            final_frame = img
        else:
            final_frame = np.concatenate([img, img_no_frame], 1)
        out.write(final_frame)
        frame_number += 1
    #print('Creating new video frames %d/%d  ' % (length, length), '\n', end='')
    #print(f'New videos created, file name - {output_file}.avi')
    print('\n')
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return last_frame_distance_player1, last_frame_distance_player2

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
                  stickman=True, stickman_box=True, court=False,
                  output_file='output', output_folder='output',
                  smoothing=True, top_view=False):
    """
    Takes videos of one person as input, and calculate the body pose and face landmarks, and saves them as csv files.
    Also, output a result videos with the keypoints marked.
    :param court:
    :param video_path: str, path to the videos
    :param show_video: bool, show processed videos while processing (default = False)
    :param include_video: bool, result output videos will include the original videos as well as the
    keypoints (default = True)
    :param stickman: bool, calculate pose and create stickman using the pose data (default = True)
    :param stickman_box: bool, show person bounding box in the output videos (default = False)
    :param output_file: str, output file name (default = 'output')
    :param output_folder: str, output folder name (default = 'output') will create new folder if it does not exist
    :param smoothing: bool, use smoothing on output data (default = True)
    :return: None
    """
    dtype = get_dtype()

    # initialize extractors
    court_detector = CourtDetector()
    detection_model = DetectionModel(dtype=dtype)
    pose_extractor = PoseExtractor(person_num=1, box=stickman_box, dtype=dtype) if stickman else None
    stroke_recognition = ActionRecognition('storke_classifier_weights.pth')
    ball_detector = BallDetector('saved states/tracknet_weights_2_classes.pth', out_channels=2)

    # Load videos from videos path
    video = cv2.VideoCapture(video_path)

    # get videos properties
    fps, length, v_width, v_height = get_video_properties(video)

    # frame counter
    frame_i = 0

    # time counter
    total_time = 0

    # Loop over all frames in the videos
    while True:
        start_time = time.time()
        ret, frame = video.read()
        frame_i += 1

        if ret:
            if frame_i == 1:
                start_time = time.time()
                court_detector.detect(frame)
                court_detection_time = time.time() - start_time
                #print(f'Court detection time :  {court_detection_time:.1f} seconds')

                #print('Court detection accuracy = %.1f' % court_accuracy)

            court_detector.track_court(frame)

            # detect
            detection_model.detect_player_1(frame.copy(), court_detector)
            detection_model.detect_top_persons(frame, court_detector, frame_i)

            # Create stick man figure (pose detection)
            if stickman:
                pose_extractor.extract_pose(frame, detection_model.player_1_boxes)

            ball_detector.detect_ball(court_detector.delete_extra_parts(frame))

            total_time += (time.time() - start_time)
            print('Processing frame %d/%d  FPS %04f' % (frame_i, length, frame_i / total_time), '\r', end='')
            if not frame_i % 100:
                print('')
        else:
            break
    #print('Processing frame %d/%d  FPS %04f' % (length, length, length / total_time), '\n', end='')
    #print('Processing completed')
    video.release()
    cv2.destroyAllWindows()

    detection_model.find_player_2_box()

    if top_view:
        create_top_view(court_detector, detection_model, ball_detector, fps=fps)

    # Save landmarks in csv files
    df = None
    # Save stickman data
    if stickman:
        df = pose_extractor.save_to_csv(output_folder)

    # smooth the output data for better results
    df_smooth = None
    if smoothing:
        smoother = Smooth()
        df_smooth = smoother.smooth(df)
        smoother.save_to_csv(output_folder)

    player_1_strokes_indices, player_2_strokes_indices, bounces_indices, f2_x, f2_y = find_strokes_indices(
        detection_model.player_1_boxes,
        detection_model.player_2_boxes,
        ball_detector.xy_coordinates,
        df_smooth)

    '''ball_detector.bounces_indices = bounces_indices
    ball_detector.coordinates = (f2_x, f2_y)'''
    predictions, prediction_list = get_stroke_predictions(video_path, stroke_recognition,
                                         player_1_strokes_indices, detection_model.player_1_boxes)

    statistics = Statistics(court_detector, detection_model)
    heatmap = statistics.get_player_position_heatmap()
    statistics.display_heatmap(heatmap, court_detector.court_reference.court, title='Heatmap')
    statistics.get_players_dists()

    last_frame_distance_p1, last_frame_distance_p2 = add_data_to_video(input_video=video_path, court_detector=court_detector, players_detector=detection_model,
                      ball_detector=ball_detector, strokes_predictions=predictions, skeleton_df=df_smooth,
                      statistics=statistics,
                      show_video=show_video, with_frame=1, output_folder=output_folder, output_file=output_file,
                      p1=player_1_strokes_indices, p2=player_2_strokes_indices, f_x=f2_x, f_y=f2_y)
    ball_detector.show_y_graph(detection_model.player_1_boxes, detection_model.player_2_boxes)
    #print(f'Last frame distance player 1 : {last_frame_distance_p1:.1f} m')
    #print(f'Last frame distance player 2 : {last_frame_distance_p2:.1f} m')

# Calculate stroke counts

    stroke_counts = {}
    for stroke_type in prediction_list:
        stroke_counts[stroke_type] = stroke_counts.get(stroke_type, 0) + 1

# Calculate percentages
    total_strokes = len(prediction_list)
    percentages = {stroke_type: (count / total_strokes) * 100 for stroke_type, count in stroke_counts.items()}

# Calculate court accuracy

    court_accuracy = court_detector._get_court_accuracy()

    # Create the dictionary with distances and stroke counts
    dico = {
        'distance': {
            'last_frame_distance_player1': round(last_frame_distance_p1, 1),
            'last_frame_distance_player2': round(last_frame_distance_p2, 1)
        },
        'stroke_sequence': prediction_list,
        'stroke_counts': stroke_counts,
        'court_detection_time': round(court_detection_time, 1),
        'court_accuracy': round(court_accuracy, 1),
        'total_frames_analyzed': length
    }

    print('MODEL METRICS')
    print(f'Court detection time :  {court_detection_time:.1f} seconds')
    print(f'Court detection accuracy = {court_accuracy:.1f}%')
    print(f'{length} frames analyzed by the model')
    print('\n')
    print('GAME STATISTICS')
    print(f'Last frame distance player 1 : {last_frame_distance_p1:.1f} m')
    print(f'Last frame distance player 2 : {last_frame_distance_p2:.1f} m')
    print("Shot sequence by player 1:", dico['stroke_sequence'])
    # Print percentages of each stroke type
    print('\nPERCENTAGE OF STROKE TYPES FOR PLAYER 1')
    for stroke_type, percentage in percentages.items():
        print(f'{stroke_type}: {percentage:.1f}%')

    # Ensure the output folder exists
    output_folder = 'output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Write the dictionary to a .txt file in the output folder
    with open(os.path.join(output_folder, 'dico.txt'), 'w') as file:
        json.dump(dico, file, indent=4)  # Writing JSON data in a human-readable format
    return dico

def main(video_path):
    s = time.time()
    result_json = video_process(video_path="video_input2.mp4", show_video=False, stickman=True, stickman_box=False, smoothing=True,
                  court=True, top_view=True)
    computation_time =  time.time() - s
    #print(f'Total computation time : {computation_time} seconds')
    result_json['Total computation time (s)'] = computation_time
    return result_json

if __name__ == "__main__":
    video_path = "video_input2.mp4"
    main(video_path)
