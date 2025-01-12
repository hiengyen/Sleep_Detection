import cv2
import subprocess

# Địa chỉ RTSP của MediaMTX
rtsp_url = "rtsp://localhost:8554/mystream"

# Lệnh FFmpeg để xuất luồng video
ffmpeg_command = [
    "ffmpeg",
    "-y",  # Ghi đè file output nếu có
    "-f",
    "rawvideo",  # Định dạng video thô
    "-pix_fmt",
    "bgr24",  # OpenCV sử dụng định dạng BGR
    "-s",
    "640x480",  # Kích thước khung hình (phải khớp với camera)
    "-r",
    "30",  # Frame rate
    "-i",
    "-",  # Input từ stdin
    "-c:v",
    "libx264",  # Codec video
    "-preset",
    "ultrafast",  # Tốc độ mã hóa
    "-f",
    "rtsp",  # Giao thức output
    rtsp_url,
]

# Mở camera
cap = cv2.VideoCapture(0)

# Kiểm tra camera có mở được không
if not cap.isOpened():
    print("Không thể mở camera.")
    exit()

# Lấy kích thước khung hình từ camera
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print(f"Đang stream {width}x{height} @ {fps}fps tới {rtsp_url}")

# Tạo process FFmpeg
process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc frame từ camera.")
            break

        # Gửi frame tới FFmpeg qua stdin
        process.stdin.write(frame.tobytes())

except KeyboardInterrupt:
    print("Dừng streaming...")

finally:
    # Đóng camera và FFmpeg
    cap.release()
    process.stdin.close()
    process.wait()
