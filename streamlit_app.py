import os
import cv2
import numpy as np
import streamlit as st
from utils.input_handler import process_image, process_folder, process_video
from utils.webcam_detect import CameraStream
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, filename='streamlit_startup.log')
logger = logging.getLogger(__name__)

# Initialize session state for simple authentication
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'username' not in st.session_state:
    st.session_state.username = ""

# Simple authentication (replace with streamlit-authenticator for production)
def login():
    with st.sidebar:
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            # Dummy check (replace with your auth logic)
            if username == "admin" and password == "password":
                st.session_state.authenticated = True
                st.session_state.username = username
                st.success("Logged in as {}".format(username))
            else:
                st.error("Invalid credentials")
    if not st.session_state.authenticated:
        st.stop()

# Main app
def main():
    start_time = time.time()
    logger.info("Starting Streamlit app initialization")

    # Pre-create directories
    dir_start_time = time.time()
    for dir_path in ['static/uploads', 'static/results', 'static/uploads/batch', 'static/results/batch']:
        os.makedirs(dir_path, exist_ok=True)
    logger.info(f"Directories created in {time.time() - dir_start_time} seconds")

    login()  # Check authentication

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Dashboard", "Image Detection", "Batch Detection", "Video Detection", "Webcam"])

    if page == "Dashboard":
        st.title("Dashboard")
        st.write(f"Welcome, {st.session_state.username}!")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.username = ""
            st.experimental_rerun()

    elif page == "Image Detection":
        st.title("Image Detection")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Save file
            filepath = os.path.join("static/uploads", uploaded_file.name)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            # Process image
            annotated_img, analysis = process_image(filepath)
            if annotated_img is not None and 'error' not in analysis:
                output_path = os.path.join("static/results", uploaded_file.name)
                cv2.imwrite(output_path, annotated_img)
                st.image(output_path, caption="Detected Objects", use_column_width=True)
                st.json(analysis)
            else:
                st.error(analysis.get('error', 'Unknown error'))

    elif page == "Batch Detection":
        st.title("Batch Image Detection")
        uploaded_files = st.file_uploader("Choose multiple images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        if uploaded_files:
            folder_path = "static/uploads/batch"
            image_paths = []
            for file in uploaded_files:
                filepath = os.path.join(folder_path, file.name)
                with open(filepath, "wb") as f:
                    f.write(file.getbuffer())
                image_paths.append(filepath)
            images, analyses = process_folder(image_paths)
            output_dir = "static/results/batch"
            os.makedirs(output_dir, exist_ok=True)
            for filename, img in images:
                out_path = os.path.join(output_dir, filename)
                cv2.imwrite(out_path, img)
                st.image(out_path, caption=f"Detected Objects in {filename}", use_column_width=True)
            st.json(analyses)

    elif page == "Video Detection":
        st.title("Video Detection")
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi"])
        if uploaded_file is not None:
            filepath = os.path.join("static/uploads", uploaded_file.name)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())
            output_path, analysis = process_video(filepath)
            if output_path and 'error' not in analysis:
                st.video(output_path)
                st.json(analysis)
            else:
                st.error(analysis.get('error', 'Unknown error'))

    elif page == "Webcam":
        st.title("Webcam Detection")
        if st.button("Start Webcam"):
            camera_stream = CameraStream()
            if camera_stream.start():
                camera_stream.start_detection()
                st.write("Press 'q' in the video window to stop (for local display)")
                while camera_stream.running:
                    frame = camera_stream.get_frame()
                    if frame is not None:
                        st.image(frame, channels="BGR", caption="Webcam Feed", use_column_width=True)
                    time.sleep(0.1)  # Control frame rate
            else:
                st.error("Failed to start camera")

    logger.info(f"Streamlit app startup completed in {time.time() - start_time} seconds")

if __name__ == "__main__":
    main()