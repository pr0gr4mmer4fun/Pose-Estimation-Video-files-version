import sys
import cv2
import mediapipe as mp
import imutils
import keyboard

print('Running\n')

print('cv2 version:', cv2.__version__)
print('MediaPipe version:', mp.__version__)
print('imutils version:', imutils.__version__, '\n')

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Replace this line with the path to your video file
video_file_path = "gameplay.mp4"

cap = cv2.VideoCapture(video_file_path)

fps = 20
cap.set(cv2.CAP_PROP_FPS, fps)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

with mp_pose.Pose(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.7) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Reached the end of the video file.")
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            None,
            #mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        square_size = 225
        square_color = (255, 255, 0)

        frame_height, frame_width, _ = image.shape
        top_left_corner = (frame_width // 2 - square_size // 2, frame_height // 2 - square_size // 2)
        bottom_right_corner = (frame_width // 2 + square_size // 2, frame_height // 2 + square_size // 2)

        overlay = image.copy()
        border_thickness = 1
        cv2.rectangle(overlay, top_left_corner, bottom_right_corner, square_color, border_thickness)

        alpha = 0.3
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

        center = (frame_width // 2, frame_height // 2)
        dot_radius = 1
        cv2.circle(image, center, dot_radius, square_color, -1)

        cv2.imshow('Video-Feed', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
