import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Initialize MediaPipe Pose solution
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Initialize video capture
cap = None

# Function to generate frames
def generate_frames():
    global cap
    while cap is not None:
        ret, img = cap.read()
        if not ret:
            break

        # Resize image/frame
        img = cv2.resize(img, (600, 400))

        # Do Pose detection
        results = pose.process(img)
        # Draw the detected pose on the video frame
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                               mp_draw.DrawingSpec((255, 0, 255), 2, 2))

        # Extract and draw pose on a plain white image
        h, w, c = img.shape
        opImg = np.zeros([h, w, c])
        opImg.fill(255)

        # Draw extracted pose on a black and white image
        mp_draw.draw_landmarks(opImg, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec((255, 0, 0), 2, 2),
                               mp_draw.DrawingSpec((255, 0, 255), 2, 2))

        # Display pose on the original video/live stream
        st.image(img, channels="BGR", use_column_width=True, caption="Pose Estimation")

        # Display extracted pose on a blank image
        st.image(opImg, channels="BGR", use_column_width=True, caption="Extracted Pose")

# Streamlit app
def main():
    global cap

    st.title("Pose Estimation App")

    st.sidebar.title("Upload Video")
    uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4"])
    
    if uploaded_file:
        cap = cv2.VideoCapture(uploaded_file)
    
    if st.sidebar.button("Start Pose Estimation"):
        st.write("Processing...")
        generate_frames()

if __name__ == '__main__':
    main()
