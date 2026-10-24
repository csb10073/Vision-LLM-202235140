#검지의 3차원 관절 각도를 이용해 건반을 누르는 순간을 판정한다.

import math


# MediaPipe가 제공하는 21개 손 landmark 중 검지에 해당하는 번호이다.
# 5번은 검지 뿌리, 6번은 중간 관절, 8번은 손가락 끝이다.
INDEX_BASE_JOINT = 5
INDEX_MIDDLE_JOINT = 6
INDEX_FINGERTIP = 8


def calculate_joint_angle(first_joint, middle_joint, last_joint):
    #중간 관절을 꼭짓점으로 하는 3차원 각도를 degree 단위로 반환한다.
    # 중간 관절에서 첫 관절로 향하는 벡터를 만든다.
    first_vector = (
        first_joint.x - middle_joint.x,
        first_joint.y - middle_joint.y,
        first_joint.z - middle_joint.z,
    )

    # 중간 관절에서 마지막 관절로 향하는 벡터를 만든다.
    second_vector = (
        last_joint.x - middle_joint.x,
        last_joint.y - middle_joint.y,
        last_joint.z - middle_joint.z,
    )

    # 두 벡터의 길이는 내적을 이용해 각도를 계산할 때 필요하다.
    first_length = math.sqrt(sum(component**2 for component in first_vector))
    second_length = math.sqrt(sum(component**2 for component in second_vector))

    # 두 좌표가 같으면 길이가 0이 되어 나눗셈을 할 수 없다.
    if first_length == 0 or second_length == 0:
        return None

    # 두 벡터의 내적과 길이로 cos 값을 구한 뒤 degree 각도로 변환한다.
    # 부동소수점 오차로 인해 cos 범위를 벗어나지 않도록 -1~1 사이로 제한한다.
    cosine = sum(
        first * second for first, second in zip(first_vector, second_vector)
    ) / (first_length * second_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

#검지의 뿌리, 중간 관절, 끝 좌표로 검지 굽힘 각도를 계산한다.
def calculate_index_finger_angle(hand_landmarks):
    
    return calculate_joint_angle(
        hand_landmarks[INDEX_BASE_JOINT],
        hand_landmarks[INDEX_MIDDLE_JOINT],
        hand_landmarks[INDEX_FINGERTIP],
    )

#검지가 펴진 상태에서 굽혀지는 순간을 한 번의 누름으로 판정한다.
class FingerPressDetector:
    

    def __init__(
        self,
        press_angle=160,
        release_angle=165,
        tracking_reset_seconds=0.5,
    ):
        # 누름 각도보다 해제 각도를 크게 두어 경계에서 상태가 반복되는 것을 막는다.
        if not 0 < press_angle < release_angle < 180:
            raise ValueError("press_angle must be smaller than release_angle")

        self.press_angle = press_angle
        self.release_angle = release_angle
        self.tracking_reset_seconds = tracking_reset_seconds

        # 현재 검지가 눌린 상태인지와 마지막으로 검출된 시간을 저장한다.
        self.is_pressed = False
        self.last_detected_at = None
        
    #저장된 누름 상태와 마지막 검출 시간을 초기화한다.
    def reset(self):
        self.is_pressed = False
        self.last_detected_at = None
    #펴져 있던 검지가 누름 각도까지 굽혀지는 순간에만 True를 반환한다.
    def update(self, finger_angle, current_time):   
        # 처음 검출됐거나 0.5초 넘게 손을 놓쳤다면 이전 상태를 이어 사용하지 않는다.
        tracking_was_lost = (
            self.last_detected_at is None
            or current_time - self.last_detected_at > self.tracking_reset_seconds
        )
        self.last_detected_at = current_time

        # 새로 나타난 손이 이미 굽혀져 있으면 누름 소리를 내지 않고 상태만 기록한다.
        if tracking_was_lost:
            self.is_pressed = finger_angle <= self.press_angle
            return False

        # 눌린 상태에서는 검지가 충분히 펴져야 다음 누름을 받을 준비가 된다.
        if self.is_pressed:
            if finger_angle >= self.release_angle:
                self.is_pressed = False
            return False

        # 펴진 검지가 160도 이하로 굽혀지는 순간을 새로운 누름으로 판단한다.
        if finger_angle <= self.press_angle:
            self.is_pressed = True
            return True
        return False
