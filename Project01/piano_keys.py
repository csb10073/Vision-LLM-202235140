#카메라 영상에서 건반 영역을 찾고 검지 끝이 들어간 음을 판정한다.

import cv2
import numpy as np


def make_keyboard_mask(blurred_grayscale):
    """주변보다 어두운 건반 선을 흰색으로 표시한 이진 영상을 만든다."""
    # 위치별 주변 밝기를 기준으로 어두운 선은 흰색, 밝은 배경은 검은색으로 만든다.
    keyboard_mask = cv2.adaptiveThreshold(
        blurred_grayscale,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        5,
    )

    # 3×3 Closing으로 건반 선에 생긴 작은 끊김을 연결한다.
    gap_filling_kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(
        keyboard_mask, cv2.MORPH_CLOSE, gap_filling_kernel
    )


def detect_key_contours(blurred_grayscale):
    """선으로 둘러싸인 건반 칸을 찾아 왼쪽부터 최대 5개 반환한다."""
    keyboard_mask = make_keyboard_mask(blurred_grayscale)

    # RETR_TREE로 바깥 테두리와 그 안의 개별 건반 사이의 부모·자식 관계도 찾는다.
    # CHAIN_APPROX_SIMPLE은 직선 위의 중복 좌표를 줄여 계산량을 낮춘다.
    detected_contours, contour_hierarchy = cv2.findContours(
        keyboard_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if contour_hierarchy is None:
        return []

    # 화면 크기에 비례한 최소 면적을 사용해 해상도가 달라도 작은 잡음을 제외한다.
    frame_height, frame_width = blurred_grayscale.shape[:2]
    frame_area = frame_height * frame_width
    minimum_key_area = max(100, frame_area * 0.0005)
    rectangular_contours = {}

    for contour_index, contour in enumerate(detected_contours):
        # 너무 작으면 잡음이고, 화면 대부분을 차지하면 배경일 가능성이 높다.
        contour_area = cv2.contourArea(contour)
        if contour_area < minimum_key_area or contour_area > frame_area * 0.8:
            continue

        # boundingRect는 contour를 감싸는 사각형의 x, y, 너비, 높이를 반환한다.
        contour_x, contour_y, contour_width, contour_height = cv2.boundingRect(contour)
        touches_frame_edge = (
            contour_x == 0
            or contour_y == 0
            or contour_x + contour_width >= frame_width
            or contour_y + contour_height >= frame_height
        )
        if touches_frame_edge:
            continue

        # arcLength는 닫힌 contour 테두리의 전체 둘레 길이를 계산한다.
        contour_perimeter = cv2.arcLength(contour, True)

        # 둘레 길이에 비례한 허용 오차로 굴곡을 줄이고 꼭짓점 네 개인 도형인지 확인한다.
        for approximation_ratio in (0.02, 0.03, 0.04):
            corner_points = cv2.approxPolyDP(
                contour, approximation_ratio * contour_perimeter, True
            )
            if len(corner_points) == 4 and cv2.isContourConvex(corner_points):
                rectangular_contours[contour_index] = corner_points
                break

    # 붙여 그린 건반에서는 큰 외곽선 안에 개별 건반 contour가 들어간다.
    # 부모로 사용되는 큰 외곽선의 번호를 모아 최종 건반 후보에서 제외한다.
    outer_contour_indices = set()
    for contour_index in rectangular_contours:
        parent_index = contour_hierarchy[0][contour_index][3]
        while parent_index != -1:
            if parent_index in rectangular_contours:
                outer_contour_indices.add(parent_index)
            parent_index = contour_hierarchy[0][parent_index][3]

    key_contours = [
        contour
        for contour_index, contour in rectangular_contours.items()
        if contour_index not in outer_contour_indices
    ]

    # 면적이 큰 후보 다섯 개만 선택한 뒤 화면의 왼쪽부터 오른쪽으로 정렬한다.
    largest_contours = sorted(
        key_contours, key=cv2.contourArea, reverse=True
    )[:5]
    return sorted(
        largest_contours,
        key=lambda contour: cv2.boundingRect(contour)[0],
    )

#검지 끝 좌표가 들어간 건반의 음 이름을 반환한다.
def find_note_at_position(locked_keys, fingertip_position):
    
    for piano_key in locked_keys:
        # 결과가 0 이상이면 점이 contour의 내부 또는 테두리 위에 있다.
        if cv2.pointPolygonTest(
            piano_key["contour"], fingertip_position, False
        ) >= 0:
            return piano_key["note"]
    return None
