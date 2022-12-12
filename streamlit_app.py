from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import streamlit as st
import mediapipe as mp
import cv2
import av

import numpy as np

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

uploaded_video_file = st.file_uploader("請上傳影片", type=['mp4'])

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        smooth_landmarks=True,
                        enable_segmentation=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

def findPose(image):
    image.flags.writeable = False
    #image = cv2.resize(image, (640, 480))   
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    # Draw on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.pose_landmarks:
         mp_drawing.draw_landmarks(
             image,
             results.pose_landmarks,
             mp_pose.POSE_CONNECTIONS,
             landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
    return cv2.flip(image, 1)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = findPose(img)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
      
if __name__ == "__main__":
    col1, col2 = st.columns(2)
    with col1:
        st.header("User Pose")
        webrtc_ctx = webrtc_streamer(
            key = "PBV1",
            mode = WebRtcMode.SENDRECV,
            rtc_configuration = RTC_CONFIGURATION,
            media_stream_constraints = {"video": True, "audio": False},
            video_processor_factory = VideoProcessor,
            async_processing = True,
        )

    with col2:
        st.header("Bench Pose")
        if uploaded_video_file is not None:
            video_bytes = uploaded_video_file.read()
            st.video(video_bytes)
          
     
