# 20261025 과제: Convolution · Motion Trail · MNIST CNN

| 파일 | 내용 | 실행 장치 | 필요한 패키지 |
| --- | --- | --- | --- |
| [HW_Convolution2D.py](HW_Convolution2D.py) | 3×3 경계 필터로 2D convolution을 계산하고 NumPy CPU와 CuPy GPU 결과를 나란히 표시 | CPU + **NVIDIA GPU** | numpy, opencv-python, matplotlib, **cupy** |
| [HW_Motiontrail.py](HW_Motiontrail.py) | 이미지를 오른쪽으로 0~31픽셀 이동한 복사본을 서로 다른 비율로 겹쳐 잔상을 만들고 CPU/GPU 시간을 비교 | CPU + **NVIDIA GPU** | numpy, opencv-python, matplotlib, **cupy** |
| [HW_MnistCNN_todo.py](HW_MnistCNN_todo.py) | MNIST로 간단한 CNN을 학습·평가하고 예측 결과와 feature map을 시각화 | CPU | torch, torchvision, numpy, matplotlib |

##  `HW_Convolution2D.py`

### 하는 일

1. 두 개의 3×3 필터를 적용한다 (padding=1, stride=1, bias=-0.2).
   - `dx_edge_filter`: 좌우 밝기 차이 → 세로 경계
   - `dy_edge_filter`: 위아래 밝기 차이 → 가로 경계
2. 같은 연산을 두 방식으로 계산한다.
   - `convolution2d` (CPU): 출력 픽셀마다 Python 반복문으로 계산
   - `convolution2d_cp` (GPU): 필터의 9개 위치만 반복하고, 위치마다 출력 전체를 GPU에서 한 번에 계산
3. 결과에 ReLU를 적용하고 입력/CPU/GPU 결과를 2×3 그림으로 그린다.



#### CPU에서 출력 픽셀을 하나씩 계산하는 이유

`convolution2d`는 출력 좌표를 선택하고, 그 좌표에 대응하는 3×3 입력 조각을 잘라 `np.sum(image_region * kernel)`로 더한다. 필터를 사진 위에 놓고 한 칸씩 옮기는 과정을 표현해, 출력값 하나가 어떻게 만들어지는지 따라가기 쉽다.


#### GPU에서 필터 위치만 반복하는 이유

같은 계산은 더하는 순서를 바꿔도 표현할 수 있다. CPU는 한 출력 픽셀의 필터 값 9개를 모아 더하지만, `convolution2d_cp`는 필터 값 하나를 고정한 뒤 모든 출력 픽셀에 대한 값을 한꺼번에 더한다. 이 구조에서는 반복문이 3×3 필터의 9개 위치만 돌고, 각 위치의 큰 배열 곱셈과 덧셈을 GPU가 병렬로 처리한다.

### 참고 사항
- NVIDIA GPU와 CUDA 드라이버가 없으면 CuPy 단계에서 오류가 난다.

---

## `HW_Motiontrail.py`

### 하는 일

1. 0~31픽셀만큼 오른쪽으로 이동한 32개 복사본을 겹친다 (`TRAIL_LENGTH = 32`).
   - 가중치를 처음에는 1.0 → 0.05로 일정하게 줄여 만들고, 이후 합이 1이 되도록 나눈다. 멀리 이동한 복사본일수록 흐릿하다.
   - 왼쪽 경계 밖에서 들어오는 값은 0으로 처리한다.
2. 같은 효과를 두 방식으로 계산한다.
   - `motion_trail_np` (CPU): 출력 픽셀 × 이동 거리를 반복문으로 계산
   - `motion_trail_cp` (GPU): 이동 거리 32번만 반복하고, 매번 이미지 영역 전체를 GPU에서 더한다.
3. 시간을 단계별로 출력하고 원본/CPU/GPU 결과를 그린다.

### 구현 로직과 이유

#### 이동한 복사본을 겹치는 이유

잔상은 사진을 조금씩 다른 위치에 놓고 겹쳐서 표현한다. 이 코드는 한 장의 정지 이미지를 오른쪽으로만 이동시켜 효과를 만든다. `TRAIL_LENGTH=32`를 통해 이동하지 않은 원본을 포함해 0~31픽셀 이동한 32개 복사본을 사용한다. 복사본을 같은 비율로 더하면 모두 비슷하게 보이므로, `np.linspace(1.0, 0.05, TRAIL_LENGTH)`로 멀리 이동한 복사본의 가중치를 작게 만든다. 

 가중치 전체를 합계로 나눠 합이 1이 되게 한다. 단순히 여러 장의 밝기를 더하면 전체가 지나치게 밝아지지만, 가중치의 합을 1로 맞추면 밝기의 가중평균이 된다. 여기서 1.0과 0.05는 나누기 전의 값이므로 실제 계산에 쓰는 첫 가중치가 1.0인 것은 아니다.


#### 경계 밖을 0으로 처리하는 이유

`input_x < 0`이면 사진 왼쪽 바깥에서 가져올 값이 없다. 이를 더하지 않는 것이 밝기 0을 더하는 것과 같다. 이 검사가 없으면 NumPy의 음수 인덱스가 사진 오른쪽 끝을 읽어서, 오른쪽 물체가 왼쪽에 나타나버리게 된다.

경계에서는 유효한 복사본이 줄어도 가중치를 다시 나누지 않는다. 그래서 왼쪽 가장자리는 상대적으로 어두워질 수 있다. 출력 크기는 원본과 같아서 오른쪽 밖으로 밀려난 부분도 잘린다. 

#### GPU가 두 영역을 잘라 더하는 이유

`motion_trail_cp`는 한 픽셀씩 원본 좌표를 찾는 대신, `image[:, :width - shift]`를 `output[:, shift:]`을 사용한다. 예를 들어 3칸 이동이면 원본의 마지막 3열을 제외한 부분을 출력의 3열 뒤부터 더한다. 두 영역의 크기가 같으므로 `output_region += input_region * weights[shift]`로 해당 영역 전체를 처리할 수 있다.

`output_region`은 출력 배열 일부를 가리키므로 `+=`의 결과가 실제 `output`에 누적된다. `shift=0`에서는 전체 원본을 겹치고, `shift >= width`이면 화면 안에 남는 부분이 없으므로 건너뛴다. 사진을 반대편으로 감아 붙이지 않고도 CPU와 같은 경계 처리를 구현할 수 있다.

CPU는 높이·너비·이동 거리의 3중으로 반복문을 사용하지만, GPU는 이동 거리 32번만 반복하고 각 영역의 곱셈·덧셈을 병렬로 처리한다. 하지만 필요한 픽셀 연산 자체가 32개로 줄어든 것은 아니고 많은 픽셀 연산을 큰 배열 단위로 GPU에 맡겨 반복을 줄인다. 



## `HW_MnistCNN_todo.py`

### 3번쨰 강의 자료의 todo 부분 assignment

## 'HW_CIFAR10.py', 'CIFAR10_CNN.pth'

### CIFAR-10 데이터셋 모델 학습 파트 , 저장한 모델 파일 assignment

## 'HW_Convolution2D.py'

### Convolution2D GPU 과속 assignment

## 'HW_IOS_NMS.py'

### IoU, NMS 구현 assignment

## 'HW_MNIST_main_gpu_function.py'

### MNIST 데이터셋 main_gpu() 함수 assignment

## 'HW_Motiontrail.py'

### 잔상효과 GPU가속 assignment



