# Video Recorder

import streamlit as st
import cv2
import numpy as np
import os

st.title("FER APP")

# Sidebar controls
st.sidebar.header("Controls")
start_record = st.sidebar.button("Start Recording")
stop_record = st.sidebar.button("Stop Recording")

# Placeholder for the video feed
frame_placeholder = st.empty()

# Initialize recording status and video writer
recording = False
video_writer = None

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"XVID")
output_file = "output.avi"

x = 0


def get_video_writer(output_file, frame_size):
    return cv2.VideoWriter(output_file, fourcc, 20.0, frame_size)


# Open the default camera (usually the first camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Error: Could not open camera.")
else:
    # Main loop
    while True:
        x += 1
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to capture image.")
            break

        # Write the frame to the output file if recording
        if recording and video_writer is not None:
            video_writer.write(frame)

        # Convert the frame to RGB format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the resulting frame
        frame_placeholder.image(frame, channels="RGB")

        # Start recording
        if start_record and not recording:
            recording = True
            video_writer = get_video_writer(
                "output.avi", (frame.shape[1], frame.shape[0])
            )

        # Stop recording
        if stop_record and recording:
            recording = False
            if video_writer is not None:
                video_writer.release()
                video_writer = None
                st.sidebar.success(f"Recording saved as {output_file}")

        # Break the loop on Streamlit stop
        if stop_record:
            # if st.sidebar.button("Stop Application", key=str(x)):
            break

    # Release everything when the job is finished
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    print(type(output_file))

    if os.path.exists(output_file):
        with open(output_file, "rb") as file:
            btn = st.download_button(
                label="Download Video",
                data=file,
                file_name=output_file,
                mime="video/x-msvideo",
            )


##################################### Video Processing ##################################################


import streamlit as st
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import timm
import torch.nn as nn
import mediapipe as mp
import time
import tempfile
import pandas as pd

# Initialize device
device = "cpu"
st.write(f"Using CUDA: {torch.cuda.is_available()}")

# Define the transformation to apply to the images
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
emotion_list = []
change_list = []
# Load the model
model = timm.create_model("tf_efficientnet_b0_ns", pretrained=False)
model.classifier = nn.Sequential(nn.Linear(in_features=1280, out_features=7))
model = torch.load(
    "C:/Users/jishn/OneDrive/Desktop/Emotion Detection Project/22.6_AffectNet_10K_part2.pt",
    map_location=device,
)
model.to(device)
model.eval()

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# # Streamlit interface
# st.title("Emotion Detection from Video")
# st.write("Upload a video file to detect emotions.")

frame_count = 0

# uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
# print(type(uploaded_file))

y = 1

if y == 1:
    # if uploaded_file is not None:
    # with tempfile.NamedTemporaryFile(delete=False) as temp_file:
    #     temp_file.write(uploaded_file.read())
    #     print(type(temp_file))
    #     video_path = temp_file.name

    cap = cv2.VideoCapture(output_file)
    st.write(f"Processing the video")
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_fps = cap.get(cv2.CAP_PROP_FPS)
    st.write(f"Total frames : {total_frames}")
    st.write(f"FPS : {total_fps}")

    # calculate duration of the video
    video_length = total_frames / total_fps
    st.write(f"Video Length = {video_length}")

    histogram = {i: 0 for i in range(7)}
    mat = [[0 for _ in range(7)] for _ in range(7)]
    prev_emotion = None
    current_emotion = None

    begin = time.time()
    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    ) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame and detect faces
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    # Get bounding box
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = (
                        int(bboxC.xmin * iw),
                        int(bboxC.ymin * ih),
                        int(bboxC.width * iw),
                        int(bboxC.height * ih),
                    )

                    # Extract the region of interest (the face)
                    face = frame[y : y + h, x : x + w]

                    # Convert the face to a PIL image
                    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

                    # Apply transformations
                    face_tensor = transform(face_pil).unsqueeze(0).to(device)

                    # Pass the face through the neural network
                    with torch.no_grad():
                        output = model(face_tensor)
                        _, predicted = torch.max(output, 1)

                    label_dict = {
                        0: "angry",
                        1: "disgust",
                        2: "fear",
                        3: "happy",
                        4: "neutral",
                        5: "sad",
                        6: "surprised",
                    }

                    # Draw a rectangle around the face and label it with the prediction
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    label = f"{label_dict[predicted.item()]}"
                    emotion_list.append(label)
                    frame_count += 1
                    # st.write(f"frame {frame_count} : {label}")
                    current_emotion = predicted.item()
                    if current_emotion != prev_emotion:
                        current_time = time.time() - begin
                        if prev_emotion != None:
                            # st.write(
                            #     f"Change detected: {label_dict[prev_emotion]} -> {label_dict[current_emotion]} at {current_time}"
                            # )
                            change_list.append(
                                f"Change detected: {label_dict[prev_emotion]} -> {label_dict[current_emotion]} at {current_time}"
                            )
                    if prev_emotion is not None:
                        mat[current_emotion][prev_emotion] += 1

                    prev_emotion = current_emotion
                    histogram[predicted.item()] += 1
                    cv2.putText(
                        frame,
                        label,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 0, 0),
                        2,
                    )

            # Display the resulting frame
            # st.image(frame, channels="BGR")

        # Release the capture and close the windows
        cap.release()

    end = time.time()
    # st.write(f"Total runtime of the program is {end - begin}")
    # st.write(f"Frame count : {frame_count}")
    frame_length = round(video_length / frame_count, 4)
    time_stamps = []
    for i in range(0, frame_count):
        time_stamps.append(round(i * frame_length, 2))

    # st.write(f"{time_stamps}")
    # st.write(f"{emotion_list}")

    # Plot histogram
    st.write("Emotion Distribution")
    x = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]
    y = list(histogram.values())
    total = sum(y)
    y_new = [(i / total) * 100 for i in y]

    st.bar_chart({"Emotions": x, "Percentage": y_new})

    print(mat)
    data = {
        "angry": mat[0],
        "disgust": mat[1],
        "fear": mat[2],
        "happy": mat[3],
        "neutral": mat[4],
        "sad": mat[5],
        "surprise": mat[6],
    }

    st.write("Change Matrix")
    st.write("Y - axis -> initial emotion")
    st.write("X - axis -> next emotion")
    df = pd.DataFrame(
        data,
        index=["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"],
    )
    st.table(df)
    # for i in mat:
    #     st.write(i[7], i[0:7])

    st.write("Change List")
    st.write(change_list)
    data = {"Time Stamp": time_stamps, "Emotion": emotion_list}
    df = pd.DataFrame(data)
    # Convert DataFrame to CSV
    csv_data = df.to_csv(index=False).encode("utf-8")

    # Define the download button
    st.download_button(
        label="Download CSV File",
        data=csv_data,
        file_name="emotion.csv",
        mime="text/csv",
    )

else:
    st.write("Please upload a video file to start emotion detection.")
