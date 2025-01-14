import cv2
import numpy as np
import torch
from ball_detection import BallDetector
from stroke_recognition import ActionRecognition
from court_detection import CourtDetector
from pose import PoseDetector

def test_ball_detection(video_path):
    """Test ball detection on a video file."""
    ball_detector = BallDetector('/content/drive/MyDrive/Tennis_Weights/tracknet_weights_2_classes.pth')
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        x, y = ball_detector.detect_ball(frame)
        if x is not None and y is not None:
            print(f"Ball detected at: ({x}, {y})")

    cap.release()

def test_stroke_recognition(video_path):
    """Test stroke recognition on a video file."""
    stroke_recognition = ActionRecognition('/content/drive/MyDrive/Tennis_Weights/storke_classifier_weights.pth')
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        stroke_recognition.add_frame(frame)
        if len(stroke_recognition.frames) >= 32:  # Process every 32 frames
            stroke = stroke_recognition.predict_stroke()
            print(f"Detected stroke: {stroke}")

    cap.release()

def test_court_detection(video_path):
    """Test court detection on a video file."""
    court_detector = CourtDetector()
    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if ret:
        court_points = court_detector.detect(frame)
        if court_points is not None:
            print("Court detected successfully")
        else:
            print("Court detection failed")

    cap.release()

def test_pose_detection(video_path):
    """Test pose detection on a video file."""
    pose_detector = PoseDetector()
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        pose_detector.detect_pose(frame)
        keypoints = pose_detector.get_pose_keypoints()
        if keypoints:
            print(f"Detected {len(keypoints)} keypoints")

    cap.release()

if __name__ == "__main__":
    video_path = "/content/drive/Shareddrives/Tennis Shot Identification/Docs de travail/test_yolov11/jono_short2.mp4"  # Update with your video path

    print("Testing Ball Detection...")
    test_ball_detection(video_path)

    print("\nTesting Stroke Recognition...")
    test_stroke_recognition(video_path)

    print("\nTesting Court Detection...")
    test_court_detection(video_path)

    print("\nTesting Pose Detection...")
    test_pose_detection(video_path)
