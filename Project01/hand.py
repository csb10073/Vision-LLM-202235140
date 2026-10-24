#MediaPipe로 한 손의 관절을 검출하고 카메라 화면에 표시한다.

import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 손 관절 검출 모델의 경로와 검지 끝의 landmark 번호이다.
# 실행 폴더의 한 단계 위에 있는 src 폴더에서 모델을 불러온다.
HAND_MODEL_PATH = "src/models/MediaPipe/hand_landmarker.task"
INDEX_FINGERTIP = 8

# 한 손만 검출하며, 이전 프레임의 손 위치를 이어서 추적하는 VIDEO 모드를 사용한다.
_detector_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)

# 위 설정으로 검출기를 만들고 손 관절 사이를 연결할 선 정보를 가져온다.
_hand_detector = vision.HandLandmarker.create_from_options(_detector_options)
_hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS

# VIDEO 모드에는 프레임 순서가 계속 증가하는 millisecond 값이 필요하다.
_last_frame_timestamp_ms = -1

# 한손을 검출하고 화면에 그린 뒤 검출 결과를 반환한다.
def detect_and_draw_hand(display_frame):
    global _last_frame_timestamp_ms

    # 같은 millisecond에 여러 프레임이 처리돼도 이전 값보다 최소 1 크게 만든다.
    frame_timestamp_ms = max(
        time.monotonic_ns() // 1_000_000,
        _last_frame_timestamp_ms + 1,
    )
    _last_frame_timestamp_ms = frame_timestamp_ms

    # OpenCV의 BGR 영상을 MediaPipe가 사용하는 RGB 영상으로 변환한다.
    rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

    # 현재 프레임에서 손의 21개 landmark 좌표를 검출한다.
    hand_detection = _hand_detector.detect_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame),
        frame_timestamp_ms,
    )

    frame_height, frame_width = display_frame.shape[:2]
    for hand_landmarks in hand_detection.hand_landmarks:
        # 0~1 범위의 landmark 좌표를 실제 화면의 pixel 좌표로 변환한다.
        landmark_positions = [
            (int(landmark.x * frame_width), int(landmark.y * frame_height))
            for landmark in hand_landmarks
        ]

        # 서로 연결된 관절 사이에 초록색 선을 그려 손 모양을 표시한다.
        for connection in _hand_connections:
            cv2.line(
                display_frame,
                landmark_positions[connection.start],
                landmark_positions[connection.end],
                (0, 255, 0),
                2,
            )

        # 검지 끝은 큰 빨간 점으로, 나머지 관절은 파란 점으로 표시한다.
        for landmark_index, landmark_position in enumerate(landmark_positions):
            is_index_fingertip = landmark_index == INDEX_FINGERTIP
            landmark_color = (0, 0, 255) if is_index_fingertip else (255, 0, 0)
            landmark_radius = 6 if is_index_fingertip else 4
            cv2.circle(
                display_frame,
                landmark_position,
                landmark_radius,
                landmark_color,
                -1,
            )

    return hand_detection


def close_detector():
    #프로그램 종료 시 MediaPipe 검출기가 사용한 자원을 정리한다.
    _hand_detector.close()
