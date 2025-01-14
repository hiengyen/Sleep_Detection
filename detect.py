import numpy as np
import os
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import time
from threading import Thread, Lock
import dlib
import cv2
import pygame
import queue
from collections import deque
import subprocess
import database

rtsp_url = "rtsp://localhost:8554/mystream"
fps = 60  # FPS mong muốn
frame_delay = 1 / fps  # Khoảng thời gian giữa mỗi khung hình


def start_rtsp_stream(rtsp_url):
    command = [
        "ffmpeg",
        "-y",  # Ghi đè file đầu ra (nếu lưu file)
        "-f",
        "rawvideo",  # Định dạng đầu vào là raw video
        "-pix_fmt",
        "bgr24",  # FFmpeg nhận khung hình BGR (OpenCV sử dụng BGR)
        "-s",
        "640x480",  # Kích thước khung hình (phải khớp với VideoStream)
        "-r",
        str(fps),  # Tốc độ khung hình đầu vào
        "-i",
        "-",  # Nhận từ stdin
        "-c:v",
        "libx264",  # Bộ mã hóa video
        "-preset",
        "ultrafast",  # Tăng tốc mã hóa
        "-f",
        "rtsp",  # Định dạng đầu ra là RTSP
        rtsp_url,
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


rtsp_process = start_rtsp_stream(rtsp_url)


# Cấu hình đường dẫn file
wav_path = "alarm.wav"

# Các ngưỡng cho phát hiện trạng thái
EYE_RATIO_THRESHOLD = 0.3
MAX_SLEEP_FRAMES = 30
MAX_NO_FACE_FRAMES = 30
MAX_HEAD_TURN_FRAMES = 45
HEAD_ROTATION_THRESHOLD = 0.8

# Thêm tham số cho bộ lọc
ROLLING_WINDOW_SIZE = 5  # Kích thước cửa sổ cho rolling average

# Khởi tạo các biến đếm frame và trạng thái
sleep_frames = 0
no_face_frames = 0
head_turn_frames = 0
warning_displayed = ""

# Khởi tạo các biến điều khiển
lock = Lock()
playing_sound = False
command_queue = queue.Queue()  # Queue để điều khiển âm thanh

# Khởi tạo các bộ đệm cho smoothing
eye_ratio_buffer = deque(maxlen=ROLLING_WINDOW_SIZE)
head_rotation_buffer = deque(maxlen=ROLLING_WINDOW_SIZE)

# Khởi tạo pygame cho xử lý âm thanh
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound(wav_path)


def smooth_value(value, buffer):
    """Làm mịn giá trị bằng rolling average"""
    buffer.append(value)
    return np.mean(buffer)


def play_sound_thread():
    """Luồng xử lý âm thanh"""
    global playing_sound
    while True:
        try:
            command = command_queue.get(timeout=0.1)
            if command == "start":
                alarm_sound.play(-1)  # Phát âm thanh lặp lại
                playing_sound = True
            elif command == "stop":
                alarm_sound.stop()
                playing_sound = False
        except queue.Empty:
            continue


def start_warning(message, messageType):
    """Khởi động cảnh báo"""
    global warning_displayed
    with lock:
        warning_displayed = message
        if not playing_sound:
            command_queue.put("start")
    database.update_message(message, messageType)


def stop_all_warnings():
    """Dừng tất cả các cảnh báo"""
    global warning_displayed
    with lock:
        warning_displayed = ""
        command_queue.put("stop")
    database.update_message("TAI XE TINH TAO", "normal")


def e_dist(pA, pB):
    """Tính khoảng cách Euclidean giữa hai điểm"""
    return np.linalg.norm(pA - pB)


def eye_ratio(eye):
    """Tính tỷ lệ đóng/mở của mắt"""
    d_V1 = e_dist(eye[1], eye[5])
    d_V2 = e_dist(eye[2], eye[4])
    d_H = e_dist(eye[0], eye[3])
    return (d_V1 + d_V2) / (2.0 * d_H)


def check_head_rotation(shape):
    """Kiểm tra góc quay đầu dựa trên tỷ lệ bất đối xứng của khuôn mặt"""
    nose = shape[30]
    left_face = shape[0]
    right_face = shape[16]

    dist_left = e_dist(nose, left_face)
    dist_right = e_dist(nose, right_face)

    asymmetry_ratio = abs(dist_left - dist_right) / \
        ((dist_left + dist_right) / 2)
    return asymmetry_ratio > HEAD_ROTATION_THRESHOLD


def check_normal_state(faces, eye_avg_ratio, is_head_turned):
    """Kiểm tra xem tất cả trạng thái có bình thường không"""
    return (
        len(faces) > 0
        and not is_head_turned
        and eye_avg_ratio >= EYE_RATIO_THRESHOLD
        and no_face_frames == 0
        and head_turn_frames == 0
        and sleep_frames == 0
    )


# Khởi tạo các detector
face_detect = dlib.get_frontal_face_detector()
landmark_detect = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Lấy chỉ số cho landmark mắt
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start,
 right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Khởi động luồng xử lý âm thanh
sound_thread = Thread(target=play_sound_thread)
sound_thread.daemon = True
sound_thread.start()

# Khởi tạo camera
vs = VideoStream(src=0).start()
time.sleep(1.0)

try:
    while True:
        start_time = time.time()  # Ghi lại thời gian bắt đầu xử lý khung hình
        # Lấy khung hình từ VideoStream
        frame = vs.read()

        # Viết khung hình vào stdin của FFmpeg
        rtsp_process.stdin.write(frame.tobytes())

        # Đồng bộ hóa thời gian để tránh gửi khung hình quá nhanh
        elapsed_time = time.time() - start_time
        time.sleep(max(0, frame_delay - elapsed_time))

        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detect(gray)

        eye_avg_ratio = 1.0
        is_head_turned = False

        if len(faces) == 0:
            no_face_frames += 1
            if no_face_frames >= MAX_NO_FACE_FRAMES:
                start_warning("KHONG PHAT HIEN KHUON MAT!", "miss")
        else:
            no_face_frames = 0
            rect = faces[0]
            landmark = landmark_detect(gray, rect)
            landmark = face_utils.shape_to_np(landmark)

            # Áp dụng smoothing cho head rotation
            is_head_turned = check_head_rotation(landmark)
            is_head_turned = (
                smooth_value(float(is_head_turned), head_rotation_buffer) > 0.5
            )

            if is_head_turned:
                head_turn_frames += 1
                if head_turn_frames >= MAX_HEAD_TURN_FRAMES:
                    start_warning("TAI XE DANG MAT TAP TRUNG!", "distract")
            else:
                head_turn_frames = 0

            # Phân tích trạng thái mắt với smoothing
            leftEye = landmark[left_eye_start:left_eye_end]
            rightEye = landmark[right_eye_start:right_eye_end]
            left_eye_ratio = eye_ratio(leftEye)
            right_eye_ratio = eye_ratio(rightEye)
            eye_avg_ratio = smooth_value(
                (left_eye_ratio + right_eye_ratio) / 2.0, eye_ratio_buffer
            )

            # Vẽ đường bao quanh mắt với màu thay đổi theo trạng thái
            eye_color = (
                (0, 255, 0) if eye_avg_ratio >= EYE_RATIO_THRESHOLD else (0, 0, 255)
            )
            left_eye_bound = cv2.convexHull(leftEye)
            right_eye_bound = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [left_eye_bound], -1, eye_color, 1)
            cv2.drawContours(frame, [right_eye_bound], -1, eye_color, 1)

            if eye_avg_ratio < EYE_RATIO_THRESHOLD:
                sleep_frames += 1
                if sleep_frames >= MAX_SLEEP_FRAMES:
                    start_warning("TAI XE DANG BUON NGU!!!", "warning")
            else:
                sleep_frames = 0

            # Hiển thị tỷ lệ mắt
            cv2.putText(
                frame,
                "EYE RATIO: {:.3f}".format(eye_avg_ratio),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

        if check_normal_state(faces, eye_avg_ratio, is_head_turned):
            stop_all_warnings()

        if warning_displayed:
            cv2.putText(
                frame,
                warning_displayed,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    # Cleanup
    cv2.destroyAllWindows()
    vs.stop()
    pygame.mixer.quit()

