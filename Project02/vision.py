"""Ultralytics YOLO 탐지와 색 판별."""

import cv2
import numpy as np


HSV_BOUNDS = {
    "blue": (((95, 100, 50), (135, 255, 255)),),
    "red": (((0, 100, 50), (4, 255, 255)), ((175, 100, 50), (180, 255, 255))),
}
MASK_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


class Yolo:

    def __init__(self, model_path, class_ids, confidence):
        from ultralytics import YOLO

        self.model = YOLO(model_path, task="detect")
        self.class_ids = list(class_ids)
        self.confidence = confidence

        # TensorRT 엔진은 첫 추론 때 불러오므로, 시연 시간에 포함되지 않게 미리 한 번 실행한다.
        self.model.predict(source=np.zeros((576, 1024, 3), dtype=np.uint8), verbose=False)

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=0.5,
            classes=self.class_ids,
            verbose=False,
        )
        result = results[0]

        height, width = frame.shape[:2]
        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()

            # 노트북과 같이 이미지 크기에 맞게 좌표를 보정한다.
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(width, int(x2))
            y2 = min(height, int(y2))
            if x2 > x1 and y2 > y1:
                detections.append(([x1, y1, x2 - x1, y2 - y1], class_id, confidence))
        return detections


def mask_for_color(hsv, color):
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in HSV_BOUNDS[color]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
    return mask


def clean_mask(mask):
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MASK_KERNEL)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, MASK_KERNEL)


def external_contours(mask):
    return cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]


def enhance_lightness(frame, clahe):
    lightness, a, b = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB))
    return cv2.cvtColor(cv2.merge((clahe.apply(lightness), a, b)), cv2.COLOR_LAB2BGR)


def colors_in_crop(crop, min_fraction, clahe=None):
    if min(crop.shape[:2]) < 5:
        return set()
    blurred = cv2.GaussianBlur(crop, (5, 5), 0)
    if clahe is not None:
        blurred = enhance_lightness(blurred, clahe)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    minimum_area = max(25, crop.shape[0] * crop.shape[1] * min_fraction)
    colors = set()
    for color in HSV_BOUNDS:
        contours = external_contours(clean_mask(mask_for_color(hsv, color)))
        if any(cv2.contourArea(contour) >= minimum_area for contour in contours):
            colors.add(color)
    return colors


# 화면 가장자리를 상하좌우 10%씩 잘라내고 가운데 영역(ROI)만 돌려준다.
# 배경이나 화면 끝에 걸친 물체를 무시하기 위함.

def central_roi(frame):
    height, width = frame.shape[:2]
    left, top = width // 10, height // 10  # 1280x720이면 left=128, top=72
    return frame[top:height - top, left:width - left], left, top
