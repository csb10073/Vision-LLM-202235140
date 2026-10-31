# 20261024 과제: Convolution · Motion Trail · MNIST CNN

이 문서는 `vision-llm-work` 폴더에 있는 다음 세 스크립트의 실행 방법, 의존성, 참고 사항을 정리한 것이다.

| 파일 | 내용 | 실행 장치 | 필요한 패키지 |
| --- | --- | --- | --- |
| `HW_Convolution.py` | 3×3 엣지 필터로 2D convolution을 하고 CPU(NumPy)와 GPU(CuPy) 결과를 비교 | CPU + **NVIDIA GPU** | numpy, opencv-python, matplotlib, **cupy** |
| `HW_Motiontrail.py` | 이미지를 오른쪽으로 0~31픽셀 이동한 복사본을 가중합해 잔상(motion trail)을 만들고 CPU/GPU 시간을 비교 | CPU + **NVIDIA GPU** | numpy, opencv-python, matplotlib, **cupy** |
| `HW_MnistCNN.py` | MNIST로 간단한 CNN을 학습·평가하고 예측 결과와 feature map을 시각화 | CPU | torch, torchvision, numpy, matplotlib |



## 3. `HW_Convolution.py`

### 하는 일

1. `seagull.jpg`를 흑백으로 읽고 1280×720으로 줄인 뒤 0~1로 정규화한다.
2. 두 개의 3×3 필터를 적용한다 (padding=1, stride=1, bias=-0.2).
   - `dx_edge_filter`: 좌우 밝기 차이 → 세로 경계
   - `dy_edge_filter`: 위아래 밝기 차이 → 가로 경계
3. 같은 연산을 두 방식으로 계산한다.
   - `convolution2d` (CPU): 출력 픽셀마다 Python 반복문으로 계산
   - `convolution2d_cp` (GPU): 필터의 9개 위치만 반복하고, 위치마다 출력 전체를 GPU에서 한 번에 계산
4. 결과에 ReLU를 적용하고 입력/CPU/GPU 결과를 2×3 그림으로 그린다.


### 참고 사항

- CPU 버전은 1280×720 픽셀을 Python 반복문으로 두 번 계산하므로 수십 초 이상 걸릴 수 있다.
- GPU 시간은 워밍업(64×64 이미지)을 먼저 실행한 뒤 `cp.cuda.Stream.null.synchronize()`로 동기화해서 측정한다. CPU→GPU 전송 시간과 ReLU 계산은 측정에 포함되지 않는다.
- NVIDIA GPU와 CUDA 드라이버가 없으면 CuPy 단계에서 오류가 난다.

---

## 4. `HW_Motiontrail.py`

### 하는 일

1. `seagull.jpg`를 흑백 1280×720으로 읽고 0~1로 정규화한다.
2. 0~31픽셀만큼 오른쪽으로 이동한 32개 복사본을 겹친다 (`TRAIL_LENGTH = 32`).
   - 가중치는 1.0 → 0.05로 선형 감소하고 합이 1이 되도록 정규화한다. 멀리 이동한 복사본일수록 흐리게 보인다.
   - 왼쪽 경계 밖에서 들어오는 값은 0으로 처리한다.
3. 같은 효과를 두 방식으로 계산한다.
   - `motion_trail_np` (CPU): 출력 픽셀 × 이동 거리를 Python 3중 반복문으로 계산
   - `motion_trail_cp` (GPU): 이동 거리 32번만 반복하고, 매번 이미지 영역 전체를 GPU에서 더한다.
4. 시간을 단계별로 출력하고 원본/CPU/GPU 결과를 1×3 그림으로 그린다.


### 출력 예시 (형식)

```text
========== NumPy CPU ==========
Processing time: ... seconds

========== CuPy GPU ==========
CPU to GPU: ... seconds
GPU processing: ... seconds
GPU to CPU: ... seconds
Total GPU time: ... seconds

========== Comparison ==========
Compute-only speedup: ...x
Including transfers: ...x
```

- `Compute-only speedup`: 순수 계산 시간끼리 비교 (CPU 시간 / GPU 계산 시간)
- `Including transfers`: GPU 업로드 + 계산 + 다운로드를 모두 포함해 비교
- 두 값 모두 워밍업 시간은 포함하지 않는다. 1보다 크면 GPU가 더 빠르다.

### 참고 사항

- **CPU 계산이 매우 오래 걸린다.** 1280 × 720 × 32 ≈ 2,950만 번 반복하므로 몇 분이 걸릴 수 있다. 빠르게 확인하려면 `TARGET_WIDTH`, `TARGET_HEIGHT`를 줄여서 실행한다.
- NVIDIA GPU와 CUDA 드라이버가 필요하다.

---

## 5. `HW.MnistCNNpy`

### 하는 일

1. MNIST(학습 60,000장 / 테스트 10,000장)를 `src/datasets`에서 불러온다. 없으면 자동으로 다운로드한다(`download=True`).
2. seed 42로 `random`, `numpy`, `torch`를 고정한다.
3. `SimpleCNN`을 만든다.

   ```text
   입력 [B, 1, 28, 28]
   → Conv(1→8, 3×3, pad 1) → ReLU → MaxPool 2×2   → [B, 8, 14, 14]
   → Conv(8→16, 3×3, pad 1) → ReLU → MaxPool 2×2  → [B, 16, 7, 7]
   → Flatten [B, 784] → FC 64 → ReLU → FC 10
   ```

4. Adam(lr=0.001), CrossEntropyLoss, batch 64로 **3 epoch** 학습한다. epoch마다 테스트 세트(batch 1000)로 평가한다.
5. 다음을 차례로 보여준다.
   - 테스트 이미지 1장의 정답/예측
   - 테스트 이미지 40장의 정답/예측/신뢰도 (4×10)
   - epoch별 Loss 그래프, Accuracy 그래프
   - conv1(8채널), conv2(16채널) feature map
   - conv1, relu1, pool1, conv2, relu2, pool2 각 단계의 feature map (4채널씩)

### 참고 사항

- **장치는 CPU로 고정되어 있다** (`device = torch.device("cpu")`). 설치된 torch도 CPU 빌드(`2.14.1+cpu`)라서 GPU가 없어도 실행된다.
- 학습 시간은 epoch마다 출력되고, 마지막에 `Total CPU training time`이 출력된다.



