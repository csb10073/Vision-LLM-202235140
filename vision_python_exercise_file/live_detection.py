
# TODO: 배운 기술(이미지 전처리, 특징 추출)을 활용하여 실시간으로 객체를 검출



# TODO: 각 프레임 이미지를 전처리
# TODO: Color Segmentation으로 Binary Mask 생성
# TODO: Morphological Operation을 적용하여 객체 영역 추출
# TODO: 객체의 중심과 반지름을 구해 프레임 위에 overlay


import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.data









pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)




cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
        

    #LAB 색공간으로 변환
    ball = cv2.cvtColor(frame,cv2.COLOR_RGB2LAB)


    green_lower = np.array([30, 60, 90], dtype=np.uint8)
    green_upper = np.array([230, 115, 180], dtype=np.uint8)


    # split channel
    l, a, b = cv2.split(ball)
    #: CLAHE 생성 (clipLimit은 3.0, tileGridSize는 8x8로 설정)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # TODO: L 채널에 CLAHE 적용 후 강화된 이미지 생성
    l_clahe_enhanced = clahe.apply(l)
    ball_enhanced_lab = cv2.merge((l_clahe_enhanced,a,b))


    # TODO: 강화된 이미지로 binary mask 생성
    mask_enhanced = cv2.inRange( ball_enhanced_lab,green_lower ,green_upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ball_mask_closed1 = cv2.morphologyEx(mask_enhanced, cv2.MORPH_CLOSE, kernel)

    #cvt lab2RGB
    ball = cv2.cvtColor(ball,cv2.COLOR_LAB2RGB)

    # CONTOUR
    # 최종 binary mask
    final_ball_mask = ball_mask_closed1

    # Contour를 그릴 원본 이미지 copy
    final_img = ball.copy()

    # 컨투어 추출 (두 번째 반환값인 hierarchy는 바깥쪽 테두리만 검출하는 현재 방식상 무의미하기에 저장하지 않음)
    contours, _ = cv2.findContours(
        final_ball_mask,        # binary mask
        cv2.RETR_EXTERNAL,       # 구멍 뚫린 객체의 바깥쪽 테두리만 검출
        cv2.CHAIN_APPROX_SIMPLE  # 직선 경로 위에 있는 픽셀 좌표 전체 대신 꼭짓점 좌표만 저장
    )

    if len(contours) > 0:
        # 가장 큰 contour
        largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
        final_mask = np.zeros_like(final_ball_mask)
        cv2.drawContours(
            final_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

    # 가장 큰 contour만 채운 binary mask를 반투명 초록색으로 overlay
        mask_color = np.zeros_like(final_img)
        mask_color[:, :, 1] = final_mask
        alpha = 0.35
        final_img = cv2.addWeighted(final_img, 1.0, mask_color, alpha, 0)

    # Contour 그리기 (초록색 외곽선)
    
        cv2.imshow("VideoCapture with GStreamer", cv2.drawContours(final_img, [largest_contour], -1, (0,255,0), 2))

cap.release()
cv2.destroyAllWindows()