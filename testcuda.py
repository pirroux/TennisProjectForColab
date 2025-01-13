import cv2

video_path = "video_input6_short.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Couldn't open the video file.")
else:
    print("Video fileo pened successfully.")
cap.release()
