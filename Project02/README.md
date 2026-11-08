# 파랑 → 빨강 절차 검증 에이전트 (Jetson 온디바이스)

카메라로 사용자가 **파란색 물체 → 빨간색 물체** 순서로 시연하는지 확인하고, 시연하는 동안 녹음한 설명에서 사용자가 **말로 설명한 순서**도 함께 비교해 결과를 한국어 음성으로 알려주는 프로그램입니다.
YOLO 탐지, Whisper 음성 인식, Piper 음성 합성, Gemma 보고서 생성까지 모두 NVIDIA Jetson 위에서 온디바이스로 동작하도록 작성되었습니다.

> ⚠️ **현재 상태:** Jetson에서 **카메라 인식(물체·색 인식, 단계 확정)까지만 동작**하고, **마이크 녹음이 정상적으로 되지 않는 것으로 보입니다.** 녹음 이후 단계(음성 인식 → 음성 순서 판정)는 정상 동작을 확인하지 못했습니다. 자세한 내용은 [알려진 문제](#알려진-문제-현재-상태)를 참고하세요.

---

## 동작 흐름

시연 1회(Enter 한 번)는 다음 순서로 진행됩니다.

1. CSI 카메라를 열고 **15초 동안 카메라 처리와 마이크 녹음을 동시에** 진행합니다.
2. 각 프레임의 가운데 영역(상하좌우 10% 제외)에서 YOLO로 병·컵(COCO 클래스 39, 41)을 찾고, 박스 안의 HSV 색으로 파랑/빨강을 판별합니다. `--color-only`이면 YOLO 없이 가운데 영역 전체의 색만 봅니다.
3. 처음에 빈 화면이 2프레임 연속 보이면 "준비 완료", 이후 같은 색이 2프레임 연속 보이면 한 단계로 확정합니다. 단계를 확정한 뒤 1.5초 동안은 새 관찰을 무시합니다.
4. 15초가 끝나면 녹음 파일을 Whisper(한국어)로 텍스트로 바꾸고, 정규식으로 "먼저 파란색, 그다음 빨간색" 같은 문장에서 순서를 추출합니다.
5. **화면 판정이 1차 기준**이며, 화면이 맞았을 때만 음성 순서를 확인해 최종 판정을 냅니다.
6. Piper TTS로 "통과" 또는 "순서를 다시 확인하시오"를 스피커로 재생합니다. 화면과 음성의 순서가 다르면 "음성과 화면의 순서가 다릅니다."를 추가로 재생합니다.
7. 시연 기록을 `reports/trial_YYYYMMDD_HHMMSS.json`으로 저장합니다. Gemma를 사용하면 평가자용 요약 보고서(3문장 이내)를 기록에 함께 넣습니다. 이 보고서는 사용자에게 읽어주지 않습니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `agent.py` | 실행 진입점. 인자 파싱, 필요한 모델 파일 확인, 모델 로드, 시연 반복 루프, 판정 후 음성 안내와 기록 저장 |
| `devices.py` | 하드웨어·외부 프로그램 연결: CSI 카메라(GStreamer), `arecord` 녹음, `whisper-cli` 음성 인식, Piper TTS와 `aplay` 재생, 미리보기 창 |
| `vision.py` | YOLO(Ultralytics, TensorRT 엔진) 탐지, HSV 기반 파랑/빨강 판별, CLAHE 밝기 보정, 중앙 ROI 자르기 |
| `procedure.py` | 프레임별 색 관찰을 단계로 확정하는 상태 머신, 화면 판정, Whisper 문장에서 말한 순서 추출, 최종 판정 |
| `report.py` | 시연 기록 JSON 저장, 기록을 문장으로 바꿔 Gemma(llama-cpp-python)로 평가자용 보고서 생성 |

---

## 실행 환경 및 의존성

### 하드웨어

- NVIDIA Jetson (JetPack 설치, CUDA·TensorRT 사용)
- CSI 카메라: `nvarguscamerasrc sensor-id=0`, 1280×720 @ 30fps로 엽니다.
- USB 마이크 (ALSA 장치)
- 스피커 (ALSA 장치)

### 시스템 패키지

- GStreamer와 `nvarguscamerasrc` (JetPack 기본 포함)
- **GStreamer 지원으로 빌드된 OpenCV**: JetPack에 포함된 시스템 OpenCV를 사용해야 합니다. pip의 `opencv-python`은 GStreamer가 빠져 있어 CSI 카메라를 열지 못합니다. `ultralytics`를 설치할 때 pip OpenCV가 함께 설치되어 시스템 OpenCV를 가리는 경우가 있으니 주의하세요.
  확인: `python3 -c "import cv2; print(cv2.getBuildInformation())" | grep GStreamer` → `YES`
- `alsa-utils` (`arecord`, `aplay`)
- `pulseaudio-utils` (`pasuspender`): 녹음 명령이 `pasuspender -- arecord ...` 형태로 실행됩니다.

```bash
sudo apt install alsa-utils pulseaudio-utils
```

### Python 패키지 (메인 환경, JetPack 기본 Python 3)

| 패키지 | 용도 | 비고 |
|---|---|---|
| `numpy` | 영상 처리 | |
| `opencv` (시스템) | 카메라, 색 판별, 미리보기 | 위 GStreamer 주의사항 참고 |
| `ultralytics` | YOLO `.engine` 추론 | Jetson용 PyTorch와 JetPack의 TensorRT Python 바인딩 필요. `--color-only`이면 불필요 |
| `llama-cpp-python` | Gemma 보고서 | **CUDA 빌드** 권장(`n_gpu_layers=-1`로 전체 레이어를 GPU에 올림). Gemma 4 모델을 지원하는 버전이어야 함. `--no-gemma`이면 불필요 |

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python
```

### 외부 도구와 모델 파일

코드의 모든 경로는 **현재 실행 위치 기준 상대 경로**로 고정되어 있으므로 반드시 **프로젝트 루트에서** 실행해야 합니다.
`agent.py`는 시작할 때 아래 파일이 있는지 먼저 확인하고, 하나라도 없으면 `필요한 파일을 찾을 수 없습니다.` 오류로 멈춥니다.

```
Project02/
├── agent.py, devices.py, procedure.py, report.py, vision.py
├── whisper.cpp/
│   ├── build-cpu/bin/whisper-cli                 # whisper.cpp CPU 빌드
│   └── models/ggml-base.bin                      # Whisper base 모델
├── .piper_venv/bin/python                        # piper-tts가 설치된 별도 venv
├── src/models/
│   ├── YOLO/yolo11n_int8.engine                  # --color-only가 아니면 필요
│   ├── Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf  # --no-gemma가 아니면 필요
│   └── Piper/ko_KR-kss-medium.onnx               # .onnx.json 파일도 같은 폴더에 필요
└── reports/                                      # 실행 시 자동 생성
```

준비 방법(예시):

```bash
# whisper.cpp (CPU 빌드, build-cpu 폴더)
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
cmake -B build-cpu
cmake --build build-cpu -j --config Release
sh ./models/download-ggml-model.sh base
cd ..

# Piper TTS (메인 환경과 분리된 venv)
python3 -m venv .piper_venv
.piper_venv/bin/pip install piper-tts

# YOLO TensorRT 엔진: TensorRT 버전·GPU에 종속되므로 실행할 Jetson에서 직접 export
yolo export model=yolo11n.pt format=engine int8=True
mv yolo11n.engine src/models/YOLO/yolo11n_int8.engine
```

---

## 실행 방법

```bash
cd Project02
python3 agent.py --preview
```

1. 모델을 불러온 뒤 `Enter로 시연 시작, q로 종료 >` 프롬프트가 나옵니다.
2. Enter를 누르면 `녹음 시작...`이 출력됩니다. 카메라 가운데를 비워 두면 `준비 완료`가 출력됩니다.
3. 파란색 물체(병/컵) → 빨간색 물체 순서로 카메라 가운데에 하나씩 보여주면서 순서를 말로 설명합니다. **시연과 녹음은 Enter를 누른 뒤 15초 동안만** 진행됩니다.
4. 15초가 지나면 음성 인식 → 판정 → 스피커 안내 → 기록 저장이 진행됩니다. 다시 Enter를 누르면 다음 시연, `q`를 입력하면 종료합니다.

미리보기 창에서 `q`를 누르거나 Ctrl+C를 누르면 프로그램 전체가 종료됩니다.

### 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--preview` | 꺼짐 | 카메라 미리보기 창 표시(탐지 박스, 인식된 색, 진행 상태). 모니터가 연결된 데스크톱 세션이 필요하며 SSH만으로는 창을 띄울 수 없습니다. |
| `--clahe` | 꺼짐 | 색을 판별하기 전에 CLAHE로 밝기 대비 보정(어두운 조명용) |
| `--color-only` | 꺼짐 | YOLO 없이 가운데 영역의 색만으로 판정하는 진단 모드(YOLO 엔진 불필요) |
| `--no-gemma` | 꺼짐 | Gemma 보고서 생략(Gemma 모델·llama-cpp-python 불필요, 메모리 절약) |
| `--mic` | `plughw:3,0` | 녹음할 ALSA 장치. **장치마다 번호가 다르므로 `arecord -l`로 확인 후 지정하세요.** |
| `--speaker` | `default` | 재생할 ALSA 장치(`aplay -l`로 확인) |

```bash
# 가장 가벼운 진단 실행: YOLO·Gemma 없이 카메라 색 + 음성만 확인
python3 agent.py --color-only --no-gemma --preview

# 마이크 카드 번호를 직접 지정
python3 agent.py --preview --mic plughw:2,0
```

---

## 출력

### 터미널

```
[인식 완료   3.4초] 파란색 | 다음 단계까지 1.5초
[인식 완료   7.8초] 빨간색 | 다음 단계까지 1.5초
[카메라] 처리한 화면 312장
[사용자 음성] 먼저 파란색을 보여주고 그 다음 빨간색을 보여줍니다
[화면 순서] 파란색 → 빨간색
[음성 순서] 파란색 → 빨간색
[판정] 통과 (파란색 다음에 빨간색을 확인했습니다.)
[기록 저장] reports/trial_20261108_141502.json
```

음성 인식이나 스피커 출력에 실패하면 `[음성 인식 오류]`, `[스피커 출력 오류]`가 출력되고 다음 단계는 계속 진행됩니다.

### 기록 파일 (`reports/trial_*.json`)

```json
{
    "expected": ["blue", "red"],
    "verdict": "통과",
    "result": "pass",
    "detail": "파란색 다음에 빨간색을 확인했습니다.",
    "camera_steps": [
        {"color": "blue", "seconds": 3.42,
         "objects": [{"class_id": 39, "confidence": 0.71, "bbox": [512, 210, 640, 470], "colors": ["blue"]}]},
        {"color": "red", "seconds": 7.81,
         "objects": [{"class_id": 41, "confidence": 0.64, "bbox": [530, 260, 660, 430], "colors": ["red"]}]}
    ],
    "transcript": "먼저 파란색을 보여주고 그 다음 빨간색을 보여줍니다",
    "spoken_order": ["blue", "red"],
    "report": "(Gemma 평가자 보고서, --no-gemma이면 없음)"
}
```

### 판정 코드 (`result`)

`pass`이면 "통과", 나머지는 모두 "순서를 다시 확인하시오"로 안내합니다.

| 코드 | 의미 |
|---|---|
| `pass` | 화면에서 파란색 → 빨간색 확인, 음성 순서도 일치 |
| `not_ready` | 시작할 때 빈 화면을 확인하지 못함 |
| `no_steps` | 파란색·빨간색 모두 확인되지 않음 |
| `missing_red` | 파란색만 확인되고 빨간색 누락 |
| `wrong_order` | 첫 단계가 파란색이 아님 |
| `both_visible` | 두 색이 동시에 보여 순서를 확정할 수 없음 |
| `extra_step` | 두 단계 외에 추가 단계가 감지됨 |
| `speech_unknown` | 화면은 맞았지만 음성에서 순서를 확정하지 못함 |
| `speech_mismatch` | 화면은 맞았지만 음성 순서가 화면과 다름 |

### 조정 가능한 값

| 파일 | 상수 | 기본값 | 의미 |
|---|---|---|---|
| `devices.py` | `SECONDS` | `15` | 시연 + 녹음 시간(초) |
| `devices.py` | `STABLE_FRAMES` | `2` | 색/빈 화면을 확정하는 연속 프레임 수 |
| `devices.py` | `STEP_DELAY` | `1.5` | 단계 확정 후 다음 단계를 받지 않는 시간(초) |
| `devices.py` | `WHISPER_THREADS` | `4` | Whisper CPU 스레드 수 |
| `agent.py` | `CLASS_IDS` | `[39, 41]` | YOLO 탐지 대상(COCO 병, 컵) |
| `agent.py` | `CONFIDENCE` | `0.30` | YOLO 신뢰도 임계값 |
| `vision.py` | `HSV_BOUNDS` | 파랑 H 95–135, 빨강 H 0–4·175–180 (S≥100, V≥50) | 색 판별 범위 |
| `report.py` | `CONTEXT_WINDOW`, `MAX_TOKENS` | `2048`, `150` | Gemma 컨텍스트 길이, 최대 생성 토큰 |

---

## 알려진 문제 (현재 상태)

### 증상

- Jetson에서 실행하면 **카메라 쪽(물체·색 인식, `[인식 완료]` 단계 확정)은 동작**합니다.
- 하지만 **마이크 녹음이 제대로 되지 않는 것으로 보입니다.** 녹음이 비어 있으면 Whisper 결과도 비게 되어(`[사용자 음성] (인식된 문장 없음)`), 화면 순서가 맞더라도 최종 판정이 `speech_unknown`으로 끝나 음성 기반 검증이 정상적으로 이뤄지지 않습니다.
- 원인은 아직 확정하지 못했습니다.

### 코드 기준 의심 지점

1. **마이크 장치 번호 하드코딩**: `--mic` 기본값이 `plughw:3,0`(카드 3)으로 고정되어 있습니다. Jetson은 HDMI·APE 같은 내장 오디오 장치가 먼저 카드 번호를 차지하고, USB 마이크 번호는 연결 순서나 부팅에 따라 바뀔 수 있습니다. 다른 장치를 잡으면 무음이 녹음되거나 `arecord`가 실패합니다.
2. **`pasuspender` 의존**: 녹음은 `pasuspender -- arecord ...`로 실행됩니다(`devices.py`의 `record_command`). `pulseaudio-utils`가 없거나 오디오 서버 상태에 따라 suspend가 실패하면 `arecord`가 실행되지 않거나 장치를 잡지 못할 수 있습니다.
3. **녹음 파일이 바로 삭제됨**: `agent.py`는 녹음(`speech.wav`)과 TTS 출력(`reply.wav`)을 `TemporaryDirectory`에 저장하고 시연 한 번이 끝나면 폴더째 삭제합니다. 그래서 실제로 무엇이 녹음됐는지 사후에 확인할 수 없습니다. 디버깅할 때는 고정 경로에 저장하도록 바꾸는 것이 좋습니다.
4. **마이크 캡처 볼륨·음소거**: ALSA 믹서에서 Capture가 음소거되어 있거나 0이면 파일은 생기지만 무음입니다. `alsamixer -c <카드번호>` → F4(Capture)에서 확인하세요.
5. **녹음 실패 시 프로그램 전체 종료**: `arecord`가 오류로 끝나면 `capture_trial`이 `RuntimeError("마이크 녹음 실패...")`를 던지는데, `main()`은 `KeyboardInterrupt`만 잡기 때문에 다음 시연으로 넘어가지 않고 프로그램이 종료됩니다.
6. **녹음 시간 15초 고정**: Enter 직후부터 15초만 녹음됩니다. 빈 화면 확인과 시연을 하는 동안 말해야 하며, 15초가 지난 뒤의 말은 녹음되지 않습니다.

### 오디오 점검 방법

```bash
# 1) 녹음·재생 장치의 카드/디바이스 번호 확인
arecord -l
aplay -l

# 2) 프로그램과 같은 형식(16kHz, mono, S16_LE)으로 5초 녹음 후 들어보기
#    plughw:3,0 은 1)에서 확인한 마이크 번호로 바꾸세요
arecord -D plughw:3,0 -f S16_LE -r 16000 -c 1 -d 5 test.wav
aplay test.wav

# 3) 프로그램과 똑같이 pasuspender를 거쳐 녹음
pasuspender -- arecord -D plughw:3,0 -f S16_LE -r 16000 -c 1 -d 5 test2.wav
aplay test2.wav

# 4) Whisper 단독 실행
./whisper.cpp/build-cpu/bin/whisper-cli -m whisper.cpp/models/ggml-base.bin -f test.wav -l ko
```

- 2)부터 무음이면 → 장치 번호 또는 캡처 볼륨 문제
- 2)는 되는데 3)이 안 되면 → `pasuspender` 문제
- 녹음은 되는데 4)에서 문장이 안 나오면 → Whisper 쪽 문제

---

## 기타 참고 사항

- **실행 위치**: 경로가 모두 상대 경로이므로 프로젝트 루트에서 실행해야 합니다.
- **카메라 재시작**: 카메라를 시연마다 새로 엽니다. 이전 실행이 비정상 종료된 뒤 `Jetson CSI 카메라를 열 수 없습니다`가 나오면 `sudo systemctl restart nvargus-daemon` 후 다시 실행하세요.
- **YOLO 워밍업**: TensorRT 엔진은 첫 추론 때 로드되므로, 시연 시간에 포함되지 않도록 시작 시 빈 이미지로 한 번 미리 추론합니다. 그래서 시작에 시간이 걸립니다.
- **탐지 대상 제한**: YOLO 모드에서는 COCO의 병(39)·컵(41)으로 탐지된 물체의 색만 봅니다. 다른 물체는 색이 맞아도 무시되므로, 인식이 안 되면 `--color-only`로 색 판별 자체가 되는지 먼저 확인하세요.
- **색 판별은 조명에 민감**: HSV 범위가 고정되어 있어 조명에 따라 빨강이 주황으로 잡히는 등 인식이 흔들릴 수 있습니다. `--clahe`를 쓰거나 `vision.py`의 `HSV_BOUNDS`를 조정하세요.
- **음성 순서 추출은 규칙 기반**: Whisper base 모델의 한국어 인식 정확도에 영향을 받습니다. 색 이름이 정확히 두 번(파랑 1번, 빨강 1번) 나와야 하고, "아니", "말고", "동시에"처럼 순서를 흐리는 표현이 있으면 순서를 확정하지 않습니다.
- **메모리**: Gemma, YOLO TensorRT 엔진, Whisper가 함께 메모리를 사용합니다. 메모리가 부족하거나 로딩이 느리면 `--no-gemma`로 실행하세요.
- **Gemma 보고서**: 평가자용으로 JSON 기록에만 저장되고 스피커로는 읽지 않습니다. 사용자에게는 고정 문구만 들려줍니다.
