import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

def main():
    st.title("Pose Estimation with OpenCV and MediaPipe")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])

    if uploaded_file is not None:
        # Create a temporary directory to store the uploaded video
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, "temp_video.mp4")

        # Save the uploaded file to the temporary directory
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Initialize MediaPipe Pose solution
        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils
        pose = mp_pose.Pose()

        # Open the video file with OpenCV
        cap = cv2.VideoCapture(temp_file_path)

        if not cap.isOpened():
            st.error("Error: Could not open video file.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame
            frame = cv2.resize(frame, (600, 400))

            # Perform pose estimation on the frame
            results = pose.process(frame)

            # Draw the detected pose on the video frame
            mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                   mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                                   mp_draw.DrawingSpec((255, 0, 255), 2, 2))

            # Display the frame with annotated pose
            st.image(frame, channels="BGR")

        cap.release()

        # Remove the temporary directory and file
        os.remove(temp_file_path)
        os.rmdir(temp_dir)

if __name__ == "__main__":
    main()
