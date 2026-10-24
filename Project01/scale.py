"""NumPy로 만든 전자음을 Windows와 Jetson에서 비동기로 재생한다."""

import numpy as np
import sounddevice as sd


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
SAMPLE_RATE = 44100
VOLUME = 0.25


def _make_note_waveform(frequency):
    # 소리를 이루는 작은 숫자들을 미리 만들어 연주 중 계산을 줄인다.
    sample_count = int(SAMPLE_RATE * NOTE_DURATION_MS / 1000)
    sample_times = np.arange(sample_count, dtype=np.float32) / SAMPLE_RATE
    waveform = (VOLUME * np.sin(2 * np.pi * frequency * sample_times)).astype(
        np.float32
    )

    # 시작과 끝의 음량을 5ms 동안 부드럽게 바꿔 딱딱거리는 소리를 줄인다.
    fade_samples = min(int(SAMPLE_RATE * 0.005), sample_count // 2)
    if fade_samples:
        fade = np.linspace(0, 1, fade_samples, dtype=np.float32)
        waveform[:fade_samples] *= fade
        waveform[-fade_samples:] *= fade[::-1]
    return waveform


_NOTE_WAVEFORMS = {
    note_name: _make_note_waveform(frequency)
    for note_name, frequency in NOTE_FREQUENCIES.items()
}


def play_note(note_name):
    # 소리가 끝날 때까지 기다리지 않아 카메라 영상을 계속 처리할 수 있다.
    waveform = _NOTE_WAVEFORMS.get(note_name)

    # 등록되지 않은 음이면 소리를 재생하지 않고 함수를 끝낸다.
    if waveform is None:
        return

    # 한 손가락 연주이므로 새 음이 오면 이전 음을 멈추고 최신 음을 재생한다.
    sd.play(waveform, samplerate=SAMPLE_RATE, blocking=False)
