import streamlit as st
import cv2
import torch
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

from ultralytics import YOLO  # YOLOv7_Customised

# ------------------------------
# Add YOLObasic_v7 repo to sys.path
# ------------------------------
YOLObasic_v7_PATH = "D:\TA\M.Tech\\2 Sem\MINI PROJECT\Final_project\Final_project\yolobasic_v7"
YOLOv7_Customised_MODEL_PATH = "D:\TA\M.Tech\\2 Sem\MINI PROJECT\Final_project\Final_project\Trained_Models\Pothole_detection_yolov7customised.pt"
YOLObasic_v7_MODEL_PATH = "D:\TA\M.Tech\\2 Sem\MINI PROJECT\Final_project\Final_project\Trained_Models\pothole_detection_basicyolov7.pt"
NAMES_PATH = "D:\TA\M.Tech\\2 Sem\MINI PROJECT\Final_project\Final_project\Trained_Models\classes.names"

sys.path.append(YOLObasic_v7_PATH)
from yolobasic_v7.models.experimental import attempt_load
from yolobasic_v7.utils.general import non_max_suppression

# ------------------------------
# Letterbox (for YOLObasic_v7)
# ------------------------------
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114),
              auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img

# ------------------------------
# Load class names
# ------------------------------
def load_class_names(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f.readlines()]

# ------------------------------
# Load YOLObasic_v7 model
# ------------------------------
@st.cache_resource
def load_model(YOLObasic_v7_MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = attempt_load(YOLObasic_v7_MODEL_PATH)
    model.to(device).eval()
    return model, device

# ------------------------------
# Load YOLOv7_Customised model
# ------------------------------
@st.cache_resource
def load_yolov7_Customised_model(YOLOv7_Customised_MODEL_PATH):
    model = YOLO(YOLOv7_Customised_MODEL_PATH)
    return model

# ------------------------------
# Detection - YOLObasic_v7
# ------------------------------
def detect_yolobasic_v7(model, device, frame, class_names, conf_thres=0.25, iou_thres=0.45):
    img = letterbox(frame, new_shape=640)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)[0]

    annotated = frame.copy()
    if pred is not None and len(pred):
        for det in pred:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            label = f"{class_names[int(cls)]} {conf:.2f}"
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return annotated

# ------------------------------
# Detection - YOLOv7_Customised
# ------------------------------

def detect_yolobasic_v7(model, device, frame, class_names, conf_thres=0.25, iou_thres=0.45):
    img = letterbox(frame, new_shape=640)
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img_tensor = torch.from_numpy(img).to(device).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    with torch.no_grad():
        pred = model(img_tensor)[0]
        pred = non_max_suppression(pred, conf_thres, iou_thres)[0]
    annotated = frame.copy()
    if pred is not None and len(pred):
        for det in pred:
            x1, y1, x2, y2, conf, cls = det.cpu().numpy()
            label = f"{class_names[int(cls)]} {conf:.2f}"
            cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(annotated, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return annotated

def detect_yolov7_Customised(model, frame):
    results = model.predict(source=frame, conf=0.25, iou=0.45, verbose=False)
    annotated = frame.copy()
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                b = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf)
                cls = int(box.cls)
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(annotated, (b[0], b[1]), (b[2], b[3]), (255, 100, 0), 2)
                cv2.putText(annotated, label, (b[0], b[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    return annotated

# ------------------------------ Streamlit UI ------------------------------
st.set_page_config(page_title="YOLObasic_v7 vs YOLOv7_Customised Pothole Detection", layout="wide")
st.title("🛣️ Pothole Detection using YOLObasic_v7 and YOLOv7_Customised")

if not Path(YOLObasic_v7_MODEL_PATH).exists() or not Path(YOLOv7_Customised_MODEL_PATH).exists():
    st.error("Model weights not found.")
    st.stop()

class_names = load_class_names(NAMES_PATH)
model_basic_v7, device_basic_v7 = load_model(YOLObasic_v7_MODEL_PATH)
model_v7_Customised = load_yolov7_Customised_model(YOLOv7_Customised_MODEL_PATH)

uploaded_video = st.file_uploader("📂 Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress = st.progress(0)
    count = 0

    stframe1 = st.empty()
    stframe2 = st.empty()
    
    col1, col2 = st.columns(2)
    placeholder1 = col1.empty()
    placeholder2 = col2.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        frame_basic_v7 = detect_yolobasic_v7(model_basic_v7, device_basic_v7, frame, class_names)
        frame_v7_Customised = detect_yolov7_Customised(model_v7_Customised, frame)

        frame_basic_v7_rgb = cv2.cvtColor(frame_basic_v7, cv2.COLOR_BGR2RGB)
        frame_v7_Customised_rgb = cv2.cvtColor(frame_v7_Customised, cv2.COLOR_BGR2RGB)

        placeholder1.image(frame_basic_v7_rgb, caption=f"YOLObasic_v7 Output - Frame {count}", use_column_width=True, channels="RGB")
        placeholder2.image(frame_v7_Customised_rgb, caption=f"YOLOv7_Customised Output - Frame {count}", use_column_width=True, channels="RGB")
        progress.progress(min(int(count / total_frames * 100), 100))

    cap.release()
    st.success("✅ Real-time detection completed.")
