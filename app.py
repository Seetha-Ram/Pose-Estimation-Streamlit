import streamlit as st
import cv2 as cv2
import tempfile
import mediapipe as mp
import numpy as np

# Initialize Streamlit
st.title("Pose Estimation App")

# Upload a video file
video_file = st.file_uploader("Upload a video", type=["mp4", "avi"])

if video_file:
    # Convert the uploaded file to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_file.write(video_file.read())
    temp_file.close()
    video_path = temp_file.name

    # Display uploaded video
    st.video(video_path)

    # Initialize MediaPipe Pose solution
    mp_pose_holistic = mp.solutions.holistic

    def process_frame(frame):
        # Load the Holistic model from MediaPipe
        with mp_pose_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            # Convert frame to RGB format (required by MediaPipe)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Perform Pose estimation on the frame
            results = holistic.process(frame)

            # Convert frame back to RGB format for Streamlit display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Draw landmarks on the frame
            if results.pose_landmarks:
                # Create a blank image (black) to draw the landmarks on
                blank_image = np.zeros_like(frame)

                # Draw landmarks on the blank image
                mp.solutions.drawing_utils.draw_landmarks(blank_image, results.pose_landmarks)

                # Combine the original frame with the landmark-drawn image
                frame = cv2.addWeighted(frame, 1, blank_image, 0.5, 0)

        return frame  # Return the processed frame

    # Process the video and apply pose estimation
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        processed_frame = process_frame(frame)

        # Display the processed frame
        st.image(processed_frame, channels="RGB", use_column_width=True)

    cap.release()
