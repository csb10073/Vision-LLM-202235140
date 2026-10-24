import os
import platform

import cv2


# Windows와 Jetson 모두 USB 웹캠을 기본값으로 사용한다.
# CSI 카메라는 실행 전에 PIANO_CAMERA=csi로 선택할 수 있다.
IS_WINDOWS = platform.system() == "Windows"
CAMERA_TYPE = os.environ.get("PIANO_CAMERA", "usb").strip().lower()
CAMERA_INDEX = 0
CSI_SENSOR_ID = 0

# USB는 거울처럼 좌우 반전하고, CSI는 기존 장착 방향에 맞춰 상하 반전한다.
# 1: 좌우 반전, 0: 상하 반전, -1: 둘 다 반전, None: 반전 없음.
FLIP_DIRECTION = 1 if CAMERA_TYPE == "usb" else 0

# Jetson CSI 카메라 영상을 OpenCV가 사용할 BGR 형식으로 변환한다.
# 마지막 queue와 appsink 설정은 오래된 프레임을 버려 입력 지연이 쌓이는 것을 막는다.
JETSON_CAMERA_PIPELINE = (
    f"nvarguscamerasrc sensor-id={CSI_SENSOR_ID} ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "                     
    "video/x-raw, format=BGRx ! "      
    "videoconvert ! "                  
    "video/x-raw, format=BGR ! "       
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)


def open_camera():
    if CAMERA_TYPE not in ("usb", "csi"):
        raise ValueError("PIANO_CAMERA must be 'usb' or 'csi'")

    if CAMERA_TYPE == "usb":
        # Jetson USB 웹캠은 Linux의 V4L2로, Windows 웹캠은 DirectShow로 연다.
        camera_backend = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_V4L2
        camera_capture = cv2.VideoCapture(CAMERA_INDEX, camera_backend)

        # 낮은 해상도와 한 장짜리 버퍼로 영상 처리 지연을 줄인다.
        camera_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera_capture.set(cv2.CAP_PROP_FPS, 30)
        return camera_capture

    if IS_WINDOWS:
        raise ValueError("CSI cameras require Jetson Linux; use PIANO_CAMERA=usb")

    # CSI는 NVIDIA 카메라 플러그인과 GStreamer를 지원하는 OpenCV가 필요하다.
    return cv2.VideoCapture(JETSON_CAMERA_PIPELINE, cv2.CAP_GSTREAMER)


#화면 방향을 맞추고 건반 검출에 사용할 흐린 흑백 영상을 만든다.
def prepare_frame(camera_frame):
    display_frame = (
        camera_frame.copy()
        if FLIP_DIRECTION is None
        else cv2.flip(camera_frame, FLIP_DIRECTION)
    )

    # 건반의 검은 테두리를 찾기 쉽도록 흑백 변환 후 작은 노이즈를 부드럽게 만든다.
    grayscale_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
    blurred_grayscale = cv2.GaussianBlur(grayscale_frame, (5, 5), 0)

    # 컬러 화면은 표시와 손 검출에, 흐린 흑백 화면은 건반 검출에 사용한다.
    return display_frame, blurred_grayscale
