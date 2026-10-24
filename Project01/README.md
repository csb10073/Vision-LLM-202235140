# 최수빈 202235140


# 손동작으로 연주하는 가상 피아노

카메라로 종이에 그린 건반을 찾고, 한 손의 검지를 구부려 도·레·미·파·솔을 연주하는 Python 프로젝트입니다. OpenCV로 건반 영역을 검출하고 MediaPipe로 검지 위치와 관절 각도를 계산합니다.

Windows PC와 **Jetson Orin Nano + USB 웹캠**에서 사용할 수 있도록 카메라 입력과 소리 출력을 구성했습니다. USB 웹캠을 기본값으로 사용하며, Jetson에서는 기존 CSI 카메라 방식도 선택할 수 있습니다. 실제 Jetson 장치에서의 카메라·소리 출력은 별도 확인이 필요합니다.

## 1. 준비 사항과 의존성

- Windows PC와 Python 3.11 64비트
- 사용할 수 있는 웹캠과 카메라 접근 권한
- 소리가 출력되는 스피커 또는 이어폰
- OpenCV 영상 창을 표시할 수 있는 데스크톱 환경
- 흰 종이에 검은 선으로 그린 사각형 건반. 다섯 칸을 가로로 붙여 그리는 구성을 권장합니다.
- 손 검출 모델: 저장소의 `src/models/MediaPipe/hand_landmarker.task`. Python 파일 폴더에서는 `../src/models/MediaPipe/hand_landmarker.task`로 접근합니다.

아래 버전은 이 프로젝트의 현재 로컬 가상환경에서 확인한 값입니다. 다른 PC에서의 실행 속도와 카메라 지원 여부는 별도로 확인해야 합니다.

| 항목 | 확인한 버전 | 역할 |
| --- | --- | --- |
| Python | 3.11.15 | 프로그램 실행 |
| `mediapipe` | 1.1.1 | 손의 관절 위치 검출과 추적 |
| `opencv-contrib-python` | 5.0.0.93 | 카메라 입력, 영상 처리, contour 검출, 화면 표시 |
| `numpy` | 2.4.6 | 영상 배열과 Closing 커널 생성 |

소리 출력에는 추가 패키지 `sounddevice`를 사용합니다. NumPy로 음의 파형을 만들므로 별도의 WAV 파일은 필요하지 않습니다. `time`, `math`, `platform`, `os`는 Python 표준 라이브러리이므로 별도 설치하지 않습니다.

OpenCV 패키지는 `cv2`라는 같은 모듈 이름을 사용하므로 **`opencv-contrib-python` 하나만 사용하는 새 가상환경**이 권장됩니다. `opencv-python`을 중복 설치하거나 화면 표시 기능이 빠진 `headless` 패키지를 사용하는 구성은 피하는 편이 좋습니다. MediaPipe의 추가 의존성은 `pip`가 함께 설치합니다. [OpenCV 패키지 안내](https://pypi.org/project/opencv-contrib-python/5.0.0.93/)

## 2. 파일 구성

GitHub 저장소에서는 **기존 `src` 폴더와 Python 파일을 모아 둔 폴더가 같은 단계에 위치**합니다. 아래의 `piano`는 Python 파일 폴더 이름의 예시이며, 실제 업로드한 폴더 이름으로 바꾸어 읽으면 됩니다.

```text
vision-llm-work/
├── src/
│   └── models/
│       └── MediaPipe/
│           └── hand_landmarker.task
└── piano/                  # Python 파일을 모아 둔 폴더 이름의 예시
    ├── README.md
    ├── project1.py          # 실행 시작점: 건반 고정, 누름 판정, 화면 표시
    ├── camera.py            # 카메라 열기, 화면 반전, 흑백 변환, Gaussian Blur
    ├── hand.py              # MediaPipe 모델 로딩, 한 손 추적, 관절 표시
    ├── piano_keys.py        # 건반 contour 검출과 검지 위치에 해당하는 음 판정
    ├── press_gestures.py    # 검지 관절 각도 계산과 누름/해제 상태 관리
    └── scale.py             # 음별 전자음을 생성해 sounddevice로 재생
```

모델이 있는 `src`도 포함해 저장소 전체를 내려받아야 합니다. Python 파일 폴더만 따로 내려받으면 위 상대 경로에서 모델을 찾을 수 없습니다.

## 3. 설치 방법

**Windows PowerShell을 저장소 최상위 폴더에서 연 상태**에서 시작합니다. 먼저 Python 파일 폴더로 이동합니다. 아래 `piano`는 실제 폴더 이름으로 대체하며, 이미 `project1.py`가 있는 폴더라면 이동은 필요하지 않습니다.

```powershell
cd .\piano
```

이후 설치와 실행 명령은 모두 **`project1.py`와 `hand.py`가 있는 폴더 안에서** 수행합니다. Python 3.11 설치 후 이 폴더에 새 가상환경을 만들고 필요한 패키지를 설치합니다.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install mediapipe==1.1.1 opencv-contrib-python==5.0.0.93 numpy==2.4.6
.\.venv\Scripts\python.exe -m pip install sounddevice
.\.venv\Scripts\python.exe -m pip check
```

`py` 명령이 없지만 Python 3.11이 설치되어 있다면 첫 줄은 `python -m venv .venv`로 대체할 수 있습니다. 다른 PC에서 복사한 `.venv`는 원래 Python 설치 경로를 참조할 수 있으므로, 전달받은 소스 폴더에서 가상환경을 새로 만드는 방식이 권장됩니다.

이 문서에서는 가상환경의 Python 실행 파일을 직접 지정하므로 PowerShell에서 가상환경을 활성화할 필요가 없습니다. VS Code에서도 Python 파일 폴더 안에 만든 `.venv\Scripts\python.exe`를 인터프리터로 선택하면 같은 환경을 사용합니다.



## 4. 실행 방법

설치가 끝나면 **Python 파일 폴더 안에서** 다음 명령으로 실행합니다. `project1.py`와 방금 만든 `.venv`가 현재 폴더에 있어야 합니다.

```powershell
.\.venv\Scripts\python.exe .\project1.py
```

`hand.py`의 모델 경로는 다음 값으로 고정되어 있습니다.

```python
HAND_MODEL_PATH = "../src/models/MediaPipe/hand_landmarker.task"
```


## 5. 사용 순서와 조작법

1. 건반 그림 전체가 웹캠에 보이도록 두고, 종이와 카메라를 움직이지 않게 배치합니다. 건반은 화면 가장자리에 닿지 않게 하고, 주변의 불필요한 사각형 물체를 줄이는 편이 좋습니다.
2. 프로그램 실행 후 `Virtual Piano` 창에서 건반의 **주황색 테두리**를 확인합니다. 처음에는 손으로 건반 그림을 가리지 않는 편이 좋습니다.
3. 원하는 건반(최대5개)가 모두 검출된 상태에서 영상 창을 선택하고 `K`를 한 번 누릅니다. 건반 위치가 저장되면 테두리가 **초록색**으로 바뀌며 왼쪽부터 음 이름이 표시됩니다.
4. 손바닥이 카메라를 향하도록 한 손을 보여 주고, 검지를 편 상태에서 화면 속 원하는 건반 안에 검지 끝의 빨간 점을 올립니다. 해당 건반 테두리가 **빨간색**으로 바뀝니다.
5. 검지를 구부리면 기억된 건반의 음이 재생됩니다. 손가락을 다시 편 뒤 구부려 다음 음을 연주합니다.
6. 영상 창에서 `Q`를 누르면 종료됩니다.

| 키/표시 | 의미 |
| --- | --- |
| `K` 또는 `k` | 고정 전 검출된 건반을 한 번에 고정. 하나 이상 검출됐을 때 동작 |
| 고정 후 `K` | 추가 동작 없음. 위치를 다시 잡으려면 프로그램을 재시작 |
| `Q` 또는 `q` | 프로그램 종료 |
| 주황색 테두리 | 아직 고정하지 않은 건반 후보 |
| 초록색 테두리 | 고정된 건반 영역 |
| 빨간색 테두리 | 현재 검지 끝이 건반 영역 안에 있음. 누름 판정과는 별개 |
| `Now Playing: ...` | 최근 누름으로 재생 요청한 음을 0.5초 동안 표시 |

음은 **반전 처리가 끝난 화면에서 왼쪽부터** `Do`, `Re`, `Mi`, `Fa`, `Sol` 순서로 배정됩니다. 최대 다섯 개까지 저장하며, 일부만 검출된 상태에서 고정하면 검출된 개수만큼 앞에서부터 음이 배정됩니다. 다섯 음을 모두 사용하려면 다섯 영역을 확인한 뒤 고정하는 것이 좋습니다.

## 6. 인식 방식과 조정할 값

```text
카메라 영상
├── 흑백 변환 → Gaussian Blur → Adaptive Threshold → Closing → 건반 검출
└── RGB 변환 → MediaPipe 한 손 추적 → 검지 위치와 관절 각도 계산
                                   ↓
             기억한 건반 위치 + 검지 굽힘 변화 → 음 재생
```

### Adaptive Threshold를 적용한 이유

웹캠으로 종이를 촬영하면 조명과 그림자 때문에 같은 종이 안에서도 밝기가 달라질 수 있습니다. 화면 전체에 하나의 고정된 기준값을 적용하는 이진화는 그림자가 진 배경을 선으로 판단하거나, 밝게 찍힌 건반 선을 놓칠 수 있습니다. 따라서 **각 위치 주변의 밝기에 맞춰 판정 기준을 계산하는 Adaptive Threshold를 사용해, 조명이 불균일한 상황에서도 건반 선을 추출하기 쉽게 했습니다.**

현재 코드는 각 픽셀 주변 `31×31` 영역에서 가까운 픽셀에 더 큰 가중치를 주는 `ADAPTIVE_THRESH_GAUSSIAN_C`를 사용합니다. 이 가중 평균에서 `5`를 뺀 값을 해당 위치의 이진화 기준으로 사용합니다. 화면의 밝기를 균일하게 바꾸는 처리가 아니라, 위치마다 다른 밝기에 맞춰 선과 배경을 나누는 기준을 정하는 방식입니다.

```python
keyboard_mask = cv2.adaptiveThreshold(
    blurred_grayscale,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    31,
    5,
)
```

`THRESH_BINARY_INV`는 기준보다 어둡거나 같은 픽셀을 흰색으로, 밝은 픽셀을 검은색으로 만듭니다. 건반의 어두운 선을 흰색 영역으로 표현해 이후 Closing과 contour 검출에 사용합니다.

전처리에서 Gaussian Blur는 작은 밝기 잡음을 줄이고, Adaptive Threshold는 지역별 밝기를 기준으로 이진화하며, Closing은 이진화 결과에 남은 작은 끊김을 연결합니다. 세 처리는 각각 다른 목적을 갖습니다. 다만 반사로 선이 완전히 하얗게 날아가거나 선과 종이의 밝기 차이가 사라진 경우까지 복구하지는 못하며, `31`과 `5`는 모든 촬영 환경에서 최적인 값으로 검증한 것은 아닙니다.

### Closing을 적용한 이유

이 프로젝트는 건반 테두리 안에 둘러싸인 개별 칸을 찾아야 합니다. 하지만 손으로 그린 선의 작은 틈이나 조명·반사에 따른 이진화 결과 때문에 테두리가 끊어지면, 칸 내부가 바깥 배경과 연결되어 개별 건반 영역을 찾기 어려워질 수 있습니다. 따라서 **이진화한 테두리의 작은 끊김을 연결해 건반 후보 검출을 안정시키기 위해 Closing을 적용**했습니다.

반전 이진화한 마스크에서는 건반 선이 흰색입니다. Closing은 이 흰색 영역을 먼저 **팽창**시켜 가까운 선 사이의 작은 틈을 연결하고, 이어서 **침식**해 팽창으로 늘어난 영역을 줄입니다. 단순 팽창만 사용할 때보다 선이 두꺼워지는 영향을 줄이면서 작은 끊김을 메울 수 있습니다. OpenCV의 `cv2.MORPH_CLOSE`가 이 연산을 의미합니다.

```python
gap_filling_kernel = np.ones((3, 3), dtype=np.uint8)
closed_mask = cv2.morphologyEx(
    keyboard_mask, cv2.MORPH_CLOSE, gap_filling_kernel
)
```

현재 `3×3` 커널은 주변의 작은 틈을 처리하기 위한 설정입니다. 커널을 크게 하면 더 큰 틈을 연결할 수 있지만 가까운 별도 선이나 영역까지 붙어 건반 모양이 달라질 수 있습니다. 따라서 작은 커널을 사용했으며, 모든 조명과 건반 크기에서 최적인 값으로 검증한 것은 아닙니다. 선이 이미 선명하고 잘 연결된 영상에서는 Closing의 효과가 작을 수 있고, 큰 끊김이나 반사로 사라진 선 전체를 복구하는 용도로는 적합하지 않습니다.

Closing은 **`K`로 고정하기 전 건반 검출**에 사용됩니다. 고정 후 손에 가려져도 영역이 유지되는 것은 저장한 contour 좌표를 다시 사용하기 때문입니다.

### 연주 판정과 설정값

- 건반 위치는 `K`를 누른 시점에 고정됩니다. 이후 손이 그림을 가려도 저장된 영역은 유지되지만, 종이나 카메라를 움직이면 실제 위치와 맞지 않게 됩니다.
- 한 손의 **검지 하나로 모든 음**을 연주합니다. 손끝이 영역에 들어가는 것만으로는 소리가 나지 않습니다.
- 검지의 뿌리·중간 관절·끝인 landmark `5`, `6`, `8`로 각도를 계산합니다. 화면 좌표는 건반 위치 판정에, 모델이 추정한 3차원 world 좌표는 각도 계산에 사용합니다.
- 누르기 직전에 검지가 들어가 있던 건반을 0.5초 동안 기억하므로, 구부리는 도중 검지 끝이 건반 밖으로 벗어나도 기억이 유효하면 해당 음을 냅니다.
- 처음부터 검지를 구부린 채 나타나거나 추적이 끊긴 뒤 구부린 채 다시 나타나면 상태만 기록합니다. 한 번 펴고 다시 구부릴 때 새 누름을 인식합니다.


| 설정 위치 | 값 | 역할 |
| --- | --- | --- |
| `press_gestures.py`의 `press_angle` | `160` | 펴진 검지의 각도가 160도 이하가 되면 누름으로 판정 |
| `press_gestures.py`의 `release_angle` | `165` | 눌린 검지가 165도 이상 펴져야 다음 누름 준비 |
| `press_gestures.py`의 `tracking_reset_seconds` | `0.5` | 마지막 검출 후 이 시간보다 긴 공백이 있으면 상태를 새로 설정 |
| `project1.py`의 `KEY_POSITION_MEMORY_SECONDS` | `0.5` | 직전 건반 위치의 유효 시간 |
| `project1.py`의 `NOTE_LABEL_DURATION_SECONDS` | `0.5` | 화면에 음 이름을 표시하는 시간 |
| `scale.py`의 `NOTE_DURATION_MS` | `200` | 한 음의 재생 시간. 화면 표시 시간과 별도 |
| `camera.py`의 USB 카메라 설정 | `640×480`, `30 FPS` | 카메라에 요청하는 해상도와 속도. 실제 적용은 장치에 따라 다름 |
| `camera.py`의 `CAMERA_INDEX` / `CSI_SENSOR_ID` | 각각 `0` | USB 카메라 번호 / CSI 카메라 번호 |
| `camera.py`의 `FLIP_DIRECTION` | USB는 `1`, CSI는 `0` | `1`: 좌우, `0`: 상하, `-1`: 둘 다 반전. `None`이면 반전 없음 |
| `scale.py`의 `VOLUME` | `0.25` | 생성하는 전자음의 크기. 출력 장치 음량과 별도 |

음은 실제 피아노 녹음 대신 `262`, `294`, `330`, `349`, `392 Hz`의 전자음으로 생성합니다. 소리가 끝날 때까지 영상 처리를 기다리지 않는 비동기 재생을 사용하며, 새 음을 재생하면 이전 음은 중단됩니다. 여러 음을 동시에 재생하지 않으며, 장치별 반응 시간은 별도 확인이 필요합니다.


## 8. Jetson Orin Nano 실행 설정

Jetson에서는 기존에 MediaPipe와 NumPy가 동작하는 Python 환경을 기준으로 다음 소리 출력 의존성을 추가할 수 있습니다. `python3`는 프로그램 실행에 사용하는 Python을 의미합니다. Linux의 `sounddevice`는 PortAudio가 필요할 수 있습니다. [sounddevice 설치 안내](https://python-sounddevice.readthedocs.io/en/latest/installation.html)

```bash
sudo apt-get install libportaudio2
python3 -m pip install sounddevice
```

다음 명령은 기존 프로젝트에서 모델을 찾을 수 있는 작업 폴더에서 실행하는 예시입니다. `project1.py`가 다른 폴더에 있으면 해당 Python 파일 경로를 사용합니다.

| 카메라 | 연결 방식 | 실행 예시 |
| --- | --- | --- |
| USB 웹캠 | USB 포트 | `python3 project1.py` |
| CSI | 리본 케이블 | `PIANO_CAMERA=csi python3 project1.py` |

`PIANO_CAMERA`를 생략하면 USB 웹캠을 사용합니다. `PIANO_CAMERA=usb`로 명시해도 같습니다. Jetson USB는 V4L2로 열고, CSI는 기존 `nvarguscamerasrc` 파이프라인으로 엽니다. `camera.py`의 `CAMERA_INDEX`와 `CSI_SENSOR_ID`로 각 카메라 번호를 조정할 수 있습니다.

- **소리:** 운영체제의 기본 출력 장치로 재생합니다. HDMI/DisplayPort 모니터나 USB 오디오 등 실제 연결된 출력 장치와 음량이 설정되어 있어야 합니다. `python3 -m sounddevice`로 장치 목록을 확인할 수 있습니다.
- **소리만 확인:** 프로젝트 파일 폴더에서 `python3 -c "import scale; import sounddevice as sd; scale.play_note('Do'); sd.wait()"`로 200ms 전자음을 확인할 수 있습니다. 이 명령은 손 인식 모델을 불러오지 않습니다.
- **OpenCV:** CSI 입력에는 GStreamer 지원과 NVIDIA 카메라 플러그인이 필요합니다. 실행에 사용할 Python에서 `python3 -c "import cv2; print(cv2.getBuildInformation())"`의 `GStreamer: YES`를 확인할 수 있습니다. Windows용 OpenCV 설치 명령을 Jetson에 그대로 적용하면 현재 사용 중인 OpenCV가 바뀔 수 있습니다.
- **화면:** 기존 `cv2.imshow` 창과 `K`·`Q` 입력을 사용하므로 모니터가 연결된 데스크톱 환경을 전제로 합니다.
- **방향과 성능:** CSI는 기존 `1280×720`, `30 FPS` 요청과 상하 반전을 유지합니다. USB는 `640×480`, `30 FPS` 요청과 좌우 반전을 사용합니다. 장착 방향이 다르면 `FLIP_DIRECTION`을 조정할 수 있습니다. GPU 가속이나 실시간 성능은 Jetson에서 검증되지 않았습니다.
- **설치 환경:** JetPack, Python과 ARM64에 맞는 MediaPipe·OpenCV·NumPy 조합이 필요합니다. Windows의 `.venv`는 Jetson으로 복사해서 사용할 수 없습니다.

## 참고 문서

- [MediaPipe Hand Landmarker Python 안내](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)
- [MediaPipe 1.1.1 배포 정보](https://pypi.org/project/mediapipe/1.1.1/)
- [OpenCV 설치 및 패키지 선택 안내](https://pypi.org/project/opencv-contrib-python/5.0.0.93/)
- [NumPy 2.4.6 배포 정보](https://pypi.org/project/numpy/2.4.6/)
- [sounddevice 설치 안내](https://python-sounddevice.readthedocs.io/en/latest/installation.html)
- [sounddevice 비동기 재생 안내](https://python-sounddevice.readthedocs.io/en/latest/api/convenience-functions.html)
