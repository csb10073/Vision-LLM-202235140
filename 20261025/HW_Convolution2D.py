
import cv2
import time
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np


IMAGE_PATH = "../src/images/seagull.jpg"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

# 입력 이미지를 지정한 가로와 세로 크기로 맞춘다.
img_gray = cv2.resize(
    img_gray,
    (TARGET_WIDTH, TARGET_HEIGHT),
    interpolation=cv2.INTER_AREA
)

input_np = img_gray.astype(np.float32) / 255.0


#cpu의 convolution을 구현한다.
def convolution2d(image, kernel, bias=0.0, stride=1, padding=0):
    

    kernel_height, kernel_width = kernel.shape

    # zero padding을 해준다.
    padded_image = np.pad( 
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    # (H, W: 입력 이미지 높이, 너비, P: Padding 픽셀값, S: Stride 픽셀 값)
    # Convolution 출력 이미지 높이 = floor((H + 2P - K) / S) + 1
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1

    # Convolution 출력 이미지 너비 = floor((W + 2P - K) / S) + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    # 계산 결과는 소수가 될 수 있으므로 float32 배열에 저장한다.
    output = np.zeros((output_height, output_width), dtype=np.float32)

    # 출력 위치를 한 칸씩 선택한다. 이 두 반복문은 CPU에서 순서대로 실행된다.
    for output_y in range(output_height):
        for output_x in range(output_width):
            # 출력 좌표에 stride를 곱하면 입력에서 필터를 놓을 시작 위치가 된다.
            start_y = output_y * stride
            start_x = output_x * stride

            # 현재 시작 위치에서 필터와 같은 크기의 입력 영역만 잘라낸다.
            image_region = padded_image[
                start_y:start_y + kernel_height,
                start_x:start_x + kernel_width
            ]

            # 같은 위치끼리 곱한 값을 모두 더해 출력 픽셀 하나의 값을 구한다.
            weighted_sum = np.sum(image_region * kernel)

            # 주변 픽셀의 가중합에 bias를 한 번 더해 최종 값을 저장한다.
            output[output_y, output_x] = weighted_sum + bias

    return output

# 왼쪽 밝기는 빼고 오른쪽 밝기는 더해, 좌우 밝기가 다른 세로 경계를 찾는다.
dx_edge_filter = np.array(
    [
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ],
    dtype=np.float32
)

# 위쪽 밝기는 빼고 아래쪽 밝기는 더해, 위아래 밝기가 다른 가로 경계를 찾는다.
dy_edge_filter = np.array(
    [
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ],
    dtype=np.float32
)

# 응답에서 0.2를 뺀 뒤 ReLU를 적용해, 원래 응답이 0.2를 넘는 부분만 남긴다.
dx_bias = -0.2
dy_bias = -0.2


# GPU 방식: 필터의 숫자를 하나씩 선택하고, 모든 출력 위치를 한꺼번에 계산한다.
# image와 kernel에는 GPU 메모리에 있는 CuPy 배열을 전달한다.
def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """
    CuPy를 이용한 2차원 convolution.

    커널의 위치는 Python 반복문으로 순차 처리하고,
    각 위치에 해당하는 출력 배열 전체 연산은 GPU에서 처리한다.
    """

    # 출력 픽셀 대신 필터 안의 위치를 반복하기 위해 필터 크기를 구한다.
    kernel_height, kernel_width = kernel.shape
    
    
    padded_image = cp.pad(image,
                          ((padding,padding), (padding, padding)),
                          mode="constant",
                          constant_values=0)
    # CPU 함수와 같은 공식으로 출력 배열의 높이와 너비를 계산한다.
    output_height = (
        padded_image.shape[0] - kernel_height
    ) // stride + 1
    output_width = (
        padded_image.shape[1] - kernel_width
    ) // stride + 1

    # 필터의 각 숫자가 기여하는 값을 더할 수 있도록 GPU에서 0으로 초기화한다.
    output = cp.zeros(
        (output_height, output_width),
        dtype=cp.float32
    )

    #  필터 위치를 순서대로 선택하고, 각 배열 연산은 GPU가 처리한다.
    for kernel_y in range(kernel_height):
        for kernel_x in range(kernel_width):
            # 현재 필터 위치에 있는 입력값을 모든 출력 위치에서 모은다.
            
            input_region = padded_image[
                kernel_y:kernel_y + output_height * stride:stride,
                kernel_x:kernel_x + output_width * stride:stride
            ]

            # 현재 필터 숫자 하나의 값을 출력 배열 전체에 동시에 더한다.
            output += input_region * kernel[kernel_y, kernel_x]

    # 모든 필터 위치의 값을 더한 뒤, 각 출력 픽셀에 bias를 한 번 적용한다.
    output += bias
    return output

# 두 방향의 CPU convolution만 측정한다. 아래 ReLU 계산은 이 시간에서 제외된다.
cpu_start = time.perf_counter()

# 첫 인자 input_np는 함수 내부의 image, 둘째 인자는 kernel로 전달된다.
output_dx_np = convolution2d(
    input_np, dx_edge_filter,
    bias=dx_bias, stride=1, padding=1
)
output_dy_np = convolution2d(
    input_np, dy_edge_filter,
    bias=dy_bias, stride=1, padding=1
)
# 계산이 끝난 시각에서 시작 시각을 빼면 걸린 시간을 초 단위로 얻는다.
cpu_time = time.perf_counter() - cpu_start
# 반대 방향의 밝기 변화로 생긴 음수 값도 이 단계에서 0이 된다.
relu_dx_np = np.maximum(0, output_dx_np)
relu_dy_np = np.maximum(0, output_dy_np)

# CPU의 NumPy 배열을 GPU 메모리의 CuPy 배열로 복사한다.
input_cp = cp.asarray(input_np)
dx_filter_cp = cp.asarray(dx_edge_filter)
dy_filter_cp = cp.asarray(dy_edge_filter)

# 작은 이미지로 같은 함수를 미리 실행해 첫 GPU 계산에 필요한 준비를 한다.
# 워밍업 시간은 실제 이미지의 계산 시간에 포함하지 않는다.
warmup_image = cp.zeros((64, 64), dtype=cp.float32)

# 워밍업 결과는 사용하지 않으므로  _에 저장한다.
_ = convolution2d_cp(
    image=warmup_image,
    kernel=dx_filter_cp,
    bias=dx_bias,
    stride=1,
    padding=1
)

# 입력 전송과 워밍업이 실제로 완료된 뒤 본 계산을 시작한다.
cp.cuda.Stream.null.synchronize()

# CPU와 같은 두 방향의 convolution만 측정한다.
gpu_start = time.perf_counter()

output_dx_cp = convolution2d_cp(
    image=input_cp,
    kernel=dx_filter_cp,
    bias=dx_bias,
    stride=1,
    padding=1
)
output_dy_cp = convolution2d_cp(
    image=input_cp,
    kernel=dy_filter_cp,
    bias=dy_bias,
    stride=1,
    padding=1
)
# GPU는 CPU 코드와 별도로 작업하므로, 계산이 끝날 때까지 기다려야 한다.
# 이 대기 없이 종료 시각을 읽으면 실제 계산 시간보다 짧게 측정될 수 있다.
cp.cuda.Stream.null.synchronize()

gpu_time = time.perf_counter() - gpu_start

# GPU에서 ReLU를 적용하고, 화면 표시를 위해 결과를 NumPy 배열로 복사한다.
relu_dx_cp_np = cp.asnumpy(cp.maximum(0, output_dx_cp))
relu_dy_cp_np = cp.asnumpy(cp.maximum(0, output_dy_cp))

# TODO: CPU/GPU 결과 비교 및 출력

# 첫째 행은 세로 경계, 둘째 행은 가로 경계의 처리 결과를 보여 준다.
plt.figure(figsize=(18,8))

plt.subplot(2,3,1), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,2), plt.imshow(relu_dx_np, cmap="gray"), plt.title("Vertical Convolution (CPU)")
plt.subplot(2,3,3), plt.imshow(relu_dx_cp_np, cmap="gray"), plt.title("Vertical Convolution (GPU)")
plt.subplot(2,3,4), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,5), plt.imshow(relu_dy_np, cmap="gray"), plt.title("Horizontal Convolution (CPU)")
plt.subplot(2,3,6), plt.imshow(relu_dy_cp_np, cmap="gray"), plt.title("Horizontal Convolution (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")
