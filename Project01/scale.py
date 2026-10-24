

import threading
import winsound


# 각 음 이름에 대응하는 소리의 주파수이며 단위는 Hz이다.
# 주파수가 높을수록 더 높은 음으로 들린다.
NOTE_FREQUENCIES = {
    "Do": 262,
    "Re": 294,
    "Mi": 330,
    "Fa": 349,
    "Sol": 392,
}

# 한 음을 200 millisecond 동안 재생한다.
NOTE_DURATION_MS = 200


def play_note(note_name):
    #영상 처리를 멈추지 않도록 별도 스레드에서 음을 재생한다.
    # 전달받은 음 이름으로 주파수를 찾는다.
    note_frequency = NOTE_FREQUENCIES.get(note_name)

    # 등록되지 않은 음이면 소리를 재생하지 않고 함수를 끝낸다.
    if note_frequency is None:
        return

    # winsound.Beep는 재생이 끝날 때까지 호출한 코드의 진행을 멈춘다.
    # 별도 스레드에서 실행하면 소리가 나는 동안에도 카메라 영상을 계속 처리할 수 있다.
    threading.Thread(
        target=winsound.Beep,
        args=(note_frequency, NOTE_DURATION_MS),
        # 프로그램이 종료될 때 이 소리 스레드가 종료를 막지 않게 한다.
        daemon=True,
    ).start()
