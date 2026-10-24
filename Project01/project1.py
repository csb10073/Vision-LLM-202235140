"""카메라로 건반과 검지 누름을 인식해 도·레·미·파·솔을 연주한다."""

import time
import cv2
import camera
import hand
import piano_keys
import scale
from press_gestures import FingerPressDetector, calculate_index_finger_angle


# 왼쪽부터 건반에 배정할 음 이름과 MediaPipe의 검지 끝 landmark 번호이다.
NOTE_NAMES = ("Do", "Re", "Mi", "Fa", "Sol")
INDEX_FINGERTIP = 8

# 재생한 음 이름을 화면에 보여주는 시간과 손가락의 직전 건반 위치를 기억하는 시간이다.
NOTE_LABEL_DURATION_SECONDS = 0.5
KEY_POSITION_MEMORY_SECONDS = 0.5


def main():
    #카메라 입력부터 건반 고정, 누름 판정, 소리 출력을 반복 실행한다.
    # 카메라와 프로그램에서 계속 사용할 상태를 준비한다.
    camera_capture = camera.open_camera()
    preview_key_contours = []
    locked_keys = []
    press_detector = FingerPressDetector()
    remembered_key = None
    last_played_note = None
    note_label_visible_until = 0.0

    try:
        # 카메라를 열지 못했으면 영상 처리를 시작하지 않는다.
        if not camera_capture.isOpened():
            raise RuntimeError("Camera could not be opened")

        while True:
            # 카메라에서 현재 프레임 한 장을 가져온다.
            frame_available, camera_frame = camera_capture.read()
            if not frame_available:
                break

            # 기존 화면은 contour 표시와 손 검출에, 흐린 흑백 화면은 건반 검출에 사용한다.
            display_frame, blurred_grayscale = camera.prepare_frame(camera_frame)
            frame_height, frame_width = display_frame.shape[:2]

            # k키로 건반을 고정하기 전에는 현재 영상에서 건반 후보를 계속 갱신한다.
            if not locked_keys:
                preview_key_contours = piano_keys.detect_key_contours(blurred_grayscale)

            # 한 손의 landmark를 검출하고 화면에 관절점과 연결선을 그린다.
            hand_detection = hand.detect_and_draw_hand(display_frame)
            current_time = time.monotonic()

            # 매 프레임 새로 계산할 현재 건반과 재생할 음을 비워 둔다.
            hovered_note = None
            note_to_play = None

            if hand_detection.hand_landmarks:
                # 화면 좌표는 검지 위치에, 3차원 world 좌표는 관절 각도 계산에 사용한다.
                image_landmarks = hand_detection.hand_landmarks[0]
                if hand_detection.hand_world_landmarks:
                    angle_landmarks = hand_detection.hand_world_landmarks[0]
                else:
                    # world 좌표가 없으면 화면 landmark로 각도를 계산한다.
                    angle_landmarks = image_landmarks

                # 0~1 범위의 검지 끝 좌표를 실제 화면 pixel 좌표로 변환한다.
                fingertip_position = (
                    float(image_landmarks[INDEX_FINGERTIP].x * frame_width),
                    float(image_landmarks[INDEX_FINGERTIP].y * frame_height),
                )

                # 검지 끝이 들어간 건반의 음과 현재 검지의 굽힘 각도를 구한다.
                hovered_note = piano_keys.find_note_at_position(
                    locked_keys, fingertip_position
                )
                index_finger_angle = calculate_index_finger_angle(angle_landmarks)

                if index_finger_angle is not None:
                    # 검지가 펴져 있을 때 들어간 건반을 누르기 직전 위치로 기억한다.
                    if hovered_note is not None and not press_detector.is_pressed:
                        remembered_key = (hovered_note, current_time)

                    # 펴진 검지가 굽혀지는 순간 기억한 건반의 음을 재생 대상으로 정한다.
                    if press_detector.update(index_finger_angle, current_time):
                        if remembered_key is not None:
                            remembered_note, remembered_at = remembered_key
                            if current_time - remembered_at <= KEY_POSITION_MEMORY_SECONDS:
                                note_to_play = remembered_note
                        remembered_key = None

            # 기억한 건반 위치가 0.5초보다 오래되면 잘못된 입력을 막기 위해 삭제한다.
            if (
                remembered_key is not None
                and current_time - remembered_key[1] > KEY_POSITION_MEMORY_SECONDS
            ):
                remembered_key = None

            # 누름이 확인되면 음을 재생하고 화면에 표시할 음과 종료 시간을 저장한다.
            if note_to_play is not None:
                scale.play_note(note_to_play)
                last_played_note = note_to_play
                note_label_visible_until = current_time + NOTE_LABEL_DURATION_SECONDS

            # 고정된 건반은 초록색으로 그리고, 검지가 들어간 건반만 빨간색으로 표시한다.
            if locked_keys:
                for piano_key in locked_keys:
                    key_contour = piano_key["contour"]
                    note_name = piano_key["note"]
                    contour_color = (
                        (0, 0, 255) if note_name == hovered_note else (0, 255, 0)
                    )
                    cv2.drawContours(display_frame, [key_contour], -1, contour_color, 2)
                    key_x, key_y, _, key_height = cv2.boundingRect(key_contour)
                    cv2.putText(
                        display_frame,
                        note_name,
                        (key_x + 5, key_y + key_height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        contour_color,
                        2,
                    )
            elif preview_key_contours:
                # 고정 전 건반 후보는 주황색 contour로 미리 보여준다.
                cv2.drawContours(
                    display_frame, preview_key_contours, -1, (0, 165, 255), 2
                )

            # 음을 재생한 뒤 0.5초 동안 현재 음 이름을 화면에 표시한다.
            if current_time < note_label_visible_until:
                cv2.putText(
                    display_frame,
                    f"Now Playing: {last_played_note}",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
            cv2.putText(
                display_frame,
                "Point with index finger and bend it to play",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

            # 모든 표시가 끝난 현재 프레임을 Virtual Piano 창에 출력한다.
            cv2.imshow("Virtual Piano", display_frame)

            # Q는 프로그램을 종료하고, K는 처음 한 번만 현재 건반 위치를 고정한다.
            pressed_key = cv2.waitKey(1) & 0xFF
            if pressed_key in (ord("q"), ord("Q")):
                break
            if pressed_key in (ord("k"), ord("K")) and not locked_keys:
                if preview_key_contours:
                    # 왼쪽부터 찾은 contour에 도·레·미·파·솔을 순서대로 배정한다.
                    locked_keys = [
                        {"contour": contour.copy(), "note": note_name}
                        for contour, note_name in zip(
                            preview_key_contours, NOTE_NAMES
                        )
                    ]
                    preview_key_contours = []
                    press_detector.reset()
                    remembered_key = None
    finally:
        # 정상 종료와 오류 발생 모두에서 검출기, 카메라, 화면 자원을 정리한다.
        hand.close_detector()
        camera_capture.release()
        cv2.destroyAllWindows()


# 이 파일을 직접 실행했을 때만 main 함수를 시작한다.
if __name__ == "__main__":
    main()
