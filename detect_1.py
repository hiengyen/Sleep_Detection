import numpy as np
import os
import imutils
from imutils.video import VideoStream
from imutils import face_utils
import time
from threading import Thread, Lock
import dlib
import cv2
import subprocess

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
EYE_RATIO_THRESHOLD = 0.25  # Ngưỡng tỷ lệ mắt để phát hiện ngủ gật
MAX_SLEEP_FRAMES = 30  # Số frame tối đa cho trạng thái ngủ gật
MAX_NO_FACE_FRAMES = 30  # Số frame tối đa không thấy khuôn mặt
MAX_HEAD_TURN_FRAMES = 30  # Số frame tối đa cho trạng thái quay đầu
HEAD_ROTATION_THRESHOLD = 0.8  # Ngưỡng tỷ lệ bất đối xứng để phát hiện quay đầu

# Khởi tạo các biến đếm frame và trạng thái
sleep_frames = 0  # Đếm frame ngủ gật
no_face_frames = 0  # Đếm frame không thấy mặt
head_turn_frames = 0  # Đếm frame quay đầu
warning_displayed = ""  # Lưu trạng thái cảnh báo hiện tại

# Khởi tạo biến điều khiển
lock = Lock()  # Khóa cho đồng bộ hóa luồng
playing_sound = False  # Trạng thái phát âm thanh

# Khởi tạo các detector
face_detect = dlib.get_frontal_face_detector()
landmark_detect = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Lấy chỉ số cho landmark mắt
(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Các hàm xử lý


def play_continuous_sound(path):
    """Phát âm thanh cảnh báo liên tục"""
    global playing_sound
    while playing_sound:
        os.system("aplay " + path)


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

    asymmetry_ratio = abs(dist_left - dist_right) / ((dist_left + dist_right) / 2)
    return asymmetry_ratio > HEAD_ROTATION_THRESHOLD


def start_warning(message):
    """Khởi động cảnh báo âm thanh và hiển thị"""
    global playing_sound, warning_displayed
    if not playing_sound:
        playing_sound = True
        t = Thread(target=play_continuous_sound, args=(wav_path,))
        t.daemon = True
        t.start()
    warning_displayed = message


def stop_all_warnings():
    """Dừng tất cả các cảnh báo"""
    global playing_sound, warning_displayed
    playing_sound = False
    warning_displayed = ""


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


# Khởi tạo camera
vs = VideoStream(src=0).start()


# Vòng lặp xử lý chính
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

    # Phát hiện khuôn mặt
    faces = face_detect(gray)

    eye_avg_ratio = 1.0  # Giá trị mặc định khi không thấy mặt
    is_head_turned = False  # Giá trị mặc định khi không thấy mặt

    # Xử lý trường hợp không thấy khuôn mặt
    if len(faces) == 0:
        no_face_frames += 1
        if no_face_frames >= MAX_NO_FACE_FRAMES:
            start_warning("KHONG PHAT HIEN KHUON MAT!")
    else:
        no_face_frames = 0
        rect = faces[0]
        landmark = landmark_detect(gray, rect)
        landmark = face_utils.shape_to_np(landmark)

        # Kiểm tra góc quay đầu
        is_head_turned = check_head_rotation(landmark)
        if is_head_turned:
            head_turn_frames += 1
            if head_turn_frames >= MAX_HEAD_TURN_FRAMES:
                start_warning("TAI XE DANG MAT TAP TRUNG!!")
        else:
            head_turn_frames = 0

        # Phân tích trạng thái mắt
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

        # Kiểm tra trạng thái ngủ gật
        if eye_avg_ratio < EYE_RATIO_THRESHOLD:
            sleep_frames += 1
            if sleep_frames >= MAX_SLEEP_FRAMES:
                start_warning("TAI XE DANG BUON NGU!!!")

        else:
            sleep_frames = 0

        # Hiển thị tỷ lệ mắt
        cv2.putText(
            frame,
            "EYE AVG RATIO: {:.3f}".format(eye_avg_ratio),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

    # Kiểm tra và reset cảnh báo khi trạng thái bình thường
    if check_normal_state(faces, eye_avg_ratio, is_head_turned):
        stop_all_warnings()

    # Hiển thị cảnh báo nếu có
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

    # Hiển thị frame
    cv2.imshow("Camera", frame)

    # Kiểm tra phím thoát
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Dừng camera và đóng cửa sổ
cv2.destroyAllWindows()
vs.stop()

# Đóng stdin và chờ FFmpeg kết thúc
rtsp_process.stdin.close()
rtsp_process.wait()
