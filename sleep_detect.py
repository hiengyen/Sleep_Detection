from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import os
import imutils
import time
from threading import Thread, Lock
import dlib
import cv2
import subprocess
import firebase_admin
from firebase_admin import credentials

# Init  Firebase Admin SDK
cred = credentials.Certificate("./firebase-adminsdk.json")
firebase_admin.initialize_app(
    cred,
    {
        "storageBuckage": "sleepdetection-43473.firebasestorage.app",
        "databaseURL": "https://smart-school-firebase-default-rtdb.asia-southeast1.firebasedatabase.app/",
    },
)

# Cấu hình đường dẫn đến file alarm.wav
wav_path = "alarm.wav"

# Biến khóa để bảo vệ dữ liệu giữa các luồng
lock = Lock()

# Biến để kiểm soát trạng thái âm thanh
playing_sound = False

rtsp_url = "rtsp://localhost:8554/mystream"


# Hàm phát âm thanh liên tục
def play_continuous_sound(path):
    global playing_sound
    while playing_sound:
        os.system("aplay " + path)


# Hàm tính khoảng cách giữa 2 điểm
def e_dist(pA, pB):
    return np.linalg.norm(pA - pB)


# Hàm tính tỷ lệ mắt
def eye_ratio(eye):
    d_V1 = e_dist(eye[1], eye[5])
    d_V2 = e_dist(eye[2], eye[4])
    d_H = e_dist(eye[0], eye[3])
    return (d_V1 + d_V2) / (2.0 * d_H)


def start_rtsp_stream(rtsp_url):
    command = [
        "ffmpeg",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "450x350",
        "-r",
        "30",
        "-i",
        "-",
        "-vcodec",
        "libx264",
        "-preset",
        "ultrafast",
        "-tune",
        "zerolatency",
        "-f",
        "rtsp",
        rtsp_url,
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


rtsp_process = start_rtsp_stream(rtsp_url)


# Định nghĩa ngưỡng tỷ lệ mắt và số frame ngủ
eye_ratio_threshold = 0.19
max_sleep_frames = 8

# Khởi tạo biến đếm
sleep_frames = 0

# Khởi tạo bộ phát hiện khuôn mặt và landmark
face_detect = dlib.get_frontal_face_detector()
landmark_detect = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start,
 right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Đọc từ camera
vs = VideoStream(src=0).start()
time.sleep(1.0)


while True:
    # Đọc frame từ camera và resize
    frame = vs.read()
    frame = imutils.resize(frame, width=450)

    # publish rtsp process
    rtsp_process.stdin.write(frame.tobytes())

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt
    faces = face_detect(gray)

    # Nếu có khuôn mặt phát hiện, chỉ xử lý khuôn mặt đầu tiên
    if len(faces) > 0:
        rect = faces[0]  # Chọn khuôn mặt đầu tiên
        # Nhận diện các điểm landmark
        landmark = landmark_detect(gray, rect)
        landmark = face_utils.shape_to_np(landmark)

        # Tính toán tỷ lệ mắt trái và phải, sau đó lấy trung bình
        leftEye = landmark[left_eye_start:left_eye_end]
        rightEye = landmark[right_eye_start:right_eye_end]
        left_eye_ratio = eye_ratio(leftEye)
        right_eye_ratio = eye_ratio(rightEye)
        eye_avg_ratio = (left_eye_ratio + right_eye_ratio) / 2.0

        # Vẽ đường bao quanh mắt
        left_eye_bound = cv2.convexHull(leftEye)
        right_eye_bound = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [left_eye_bound], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_bound], -1, (0, 255, 0), 1)

        # Kiểm tra xem mắt có nhắm không
        if eye_avg_ratio < eye_ratio_threshold:
            sleep_frames += 1
            if sleep_frames >= max_sleep_frames and not playing_sound:
                playing_sound = True  # Bắt đầu phát âm thanh liên tục
                t = Thread(target=play_continuous_sound, args=(wav_path,))
                t.daemon = True
                t.start()
        else:
            # Reset sleep frames khi mở mắt và dừng cảnh báo
            sleep_frames = 0
            playing_sound = False  # Dừng phát âm thanh

        # Vẽ dòng chữ cảnh báo
        if playing_sound:
            cv2.putText(
                frame,
                "BUON NGU THI DI NGU DI ONG OI!!!",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        # Hiển thị giá trị tỷ lệ mắt trung bình
        cv2.putText(
            frame,
            "EYE AVG RATIO: {:.3f}".format(eye_avg_ratio),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    # Hiển thị lên màn hình
    cv2.imshow("Camera", frame)

    # Bấm Esc để thoát
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# Dừng camera và đóng tất cả cửa sổ
cv2.destroyAllWindows()
vs.stop()
# Đóng ffmpeg
rtsp_process.stdin.close()
rtsp_process.terminate()
rtsp_process.wait()
