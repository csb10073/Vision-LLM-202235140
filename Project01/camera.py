
import platform

import cv2


# 실행 중인 운영체제에 맞춰 카메라 연결 방식을 선택한다.
IS_WINDOWS = platform.system() == "Windows"

# Jetson CSI 카메라 영상을 OpenCV가 사용할 BGR 형식으로 변환한다.
# 마지막 queue와 appsink 설정은 오래된 프레임을 버려 입력 지연이 쌓이는 것을 막는다.
JETSON_CAMERA_PIPELINE = (
    "nvarguscamerasrc sensor-id=0 ! "  
    "video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1 ! "
    "nvvidconv ! "                     
    "video/x-raw, format=BGRx ! "      
    "videoconvert ! "                  
    "video/x-raw, format=BGR ! "       
    "queue leaky=downstream max-size-buffers=1 ! "
    "appsink drop=true max-buffers=1 sync=false"
)


def open_camera():
    if IS_WINDOWS:
       #노트북의 기본 웹캠을 연다.
        camera_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # 낮은 해상도와 한 장짜리 버퍼로 영상 처리 지연을 줄인다.
        camera_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera_capture.set(cv2.CAP_PROP_FPS, 30)
        return camera_capture

    # Windows가 아니면 Jetson의 GStreamer CSI 카메라 파이프라인을 사용한다.
    return cv2.VideoCapture(JETSON_CAMERA_PIPELINE, cv2.CAP_GSTREAMER)


#화면 방향을 맞추고 건반 검출에 사용할 흐린 흑백 영상을 만든다.
def prepare_frame(camera_frame):
    # Windows는 좌우 반전하고, Jetson은 카메라 장착 방향에 맞춰 상하 반전한다.
    flip_direction = 1 if IS_WINDOWS else 0
    display_frame = cv2.flip(camera_frame, flip_direction)

    # 건반의 검은 테두리를 찾기 쉽도록 흑백 변환 후 작은 노이즈를 부드럽게 만든다.
    grayscale_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
    blurred_grayscale = cv2.GaussianBlur(grayscale_frame, (5, 5), 0)

    # 컬러 화면은 표시와 손 검출에, 흐린 흑백 화면은 건반 검출에 사용한다.
    return display_frame, blurred_grayscale
