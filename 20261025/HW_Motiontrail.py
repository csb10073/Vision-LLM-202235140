
import cv2
import time
import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

IMAGE_PATH = "../src/images/seagull.jpg"

# CPU와 GPU에 같은 크기의 이미지를 입력해 처리 시간을 비교한다.
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# 이동하지 않은 원본을 포함해, 0~31픽셀 이동한 총 32개 복사본을 겹친다.
TRAIL_LENGTH = 32


img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

# 입력 이미지의 가로 세로 크기를 넣는다.
img_gray = cv2.resize(
    img_gray,
    (TARGET_WIDTH, TARGET_HEIGHT),
    interpolation=cv2.INTER_AREA
)

# 가중치를 곱할 수 있도록 실수 배열로 바꾸고, 밝기를 0~255에서 0~1로 맞춘다.
input_np = img_gray.astype(np.float32) / 255.0

# 잔상 효과를 적용하기 전의 원본 이미지 이다.
plt.imshow(input_np, cmap="gray")
plt.axis("off");

# CPU 방식: 출력 이미지의 픽셀을 하나씩 선택해 복사본들의 밝기를 더한다.
# weights[shift]는 shift만큼 이동한 복사본을 얼마나 겹칠지 정한다.
# 두 함수 모두 TRAIL_LENGTH만큼 반복하므로 가중치도 그 개수 이상 필요하다.


def motion_trail_np(image, weights):
    """
    NumPy를 이용해 이미지 잔상 효과를 생성한다.

    출력 픽셀을 Python 반복문으로 하나씩 계산한다.
    """

    height,width = image.shape
    
    # 입력과 같은 배열을 만든다.
    output = np.zeros_like(image)
    
    # 각 출력 픽셀마다 모든 이동 거리의 값을 계산한다.
    for output_y in range(height):
        for output_x in range(width):
            # 앞서 계산한 픽셀의 값이 섞이지 않도록 매번 누적값을 초기화한다.
            pixel_sum = 0.0
            
            for shift in range(TRAIL_LENGTH):
                # 오른쪽으로 shift만큼 옮겼을 때, 현재 출력 위치에 올 원본의 x좌표.
                input_x = output_x - shift
                # 왼쪽 경계를 벗어난 부분은 밝기 0으로 한다.
                # 이 확인이 없으면 음수 인덱스가 이미지 오른쪽 끝을 읽게 된다.
                if input_x >= 0:
                    # 원본 밝기에 복사본의 가중치를 곱한 값을 더한다.
                    pixel_sum += weights[shift] * image[output_y,input_x]
                
            # 모든 복사본을 더한 뒤, 현재 출력 픽셀에 최종 밝기를 저장한다.
            output[output_y,output_x] = pixel_sum

    return output


# GPU 방식: 이동 거리만 반복하고, 한 번에 이미지 영역 전체를 계산한다.
# image와 weights는 GPU 메모리에 저장된 CuPy 배열을 전달받는다.

def motion_trail_cp(image, weights):
    """
    CuPy를 이용해 이미지 잔상 효과를 생성한다.

    이동 거리는 Python에서 순차적으로 반복하지만,
    각 이동 거리의 이미지 전체 계산은 GPU에서 병렬 처리한다.
    """

    # 세로 방향은 이동하지 않으므로 영역을 자르는 데 필요한 너비만 사용한다.
    _, width = image.shape
    
    # 누적 결과도 GPU 메모리에 만든다.
    output = cp.zeros_like(image)
    
    for shift in range(TRAIL_LENGTH):
        # 이동 거리가 이미지 너비 이상이면 화면 안에 남는 부분이 없다.
        if shift >= width:
            continue
        # 입력의 왼쪽 영역을 출력의 오른쪽 영역에 대응시켜 오른쪽 이동을 표현한다.
        # 두 영역의 크기는 같으며, shift=0일 때는 이미지 전체를 선택한다.
        input_region = image[:, :width - shift]
        output_region = output[:, shift:]

        # 배열 전체의 곱셈과 덧셈은 GPU에서 병렬로 처리된다.
        
        output_region += input_region * weights[shift]

    return output

# 멀리 이동한 복사본일수록 가중치를 작게 해, 오른쪽으로 잔상이 옅어지게 한다.
# 두 함수가 사용하는 이동 거리마다 하나씩 가중치를 준비한다.
weights_np = np.linspace(
    1.0,
    0.05,
    TRAIL_LENGTH,
    dtype=np.float32
)

# 복사본을 여러 장 더해도 밝기가 너무 커지지 않도록 가중치 합을 1로 맞춘다.
weights_np /= np.sum(weights_np)

# CPU 함수의 실행 직전과 직후 시각을 기록해 순수 계산 시간을 구한다.
cpu_start = time.perf_counter()

output_np = motion_trail_np(
    image=input_np,
    weights=weights_np
)

cpu_end = time.perf_counter()

cpu_time = cpu_end - cpu_start

# CPU에 있는 NumPy 배열을 GPU에 복사하고, 전송 시간을 별도로 측정한다.

upload_start = time.perf_counter()

input_cp = cp.asarray(input_np)
weights_cp = cp.asarray(weights_np)

# GPU 작업은 CPU 코드와 별도로 진행되므로, 전송이 끝난 뒤 종료 시각을 읽는다.
cp.cuda.Stream.null.synchronize()

upload_end = time.perf_counter()

upload_time = upload_end - upload_start


# 작은 이미지로 같은 함수를 먼저 실행해 GPU 계산에 필요한 준비 작업을 한다.
# 이 워밍업 시간은 아래의 실제 GPU 계산 시간에 포함하지 않는다.
warmup_image = cp.zeros((64, 64), dtype=cp.float32)
# 함수가 TRAIL_LENGTH번 반복하므로 워밍업에도 전체 가중치를 전달한다.
warmup_weights = cp.asarray(weights_np)

# 워밍업의 출력은 사용하지 않으므로  _에 저장한다.
_ = motion_trail_cp(
    image=warmup_image,
    weights=warmup_weights
)

# 워밍업 계산이 실제로 끝날 때까지 기다린다.
cp.cuda.Stream.null.synchronize()

# 앞의 GPU 작업이 끝난 상태에서 실제 이미지의 계산 시간을 측정한다.
cp.cuda.Stream.null.synchronize()

gpu_start = time.perf_counter()

output_cp = motion_trail_cp(
    image=input_cp,
    weights=weights_cp
)

# 함수 호출만 끝난 시점이 아니라 GPU 계산까지 끝난 시점을 측정한다.
cp.cuda.Stream.null.synchronize()

gpu_end = time.perf_counter()

gpu_time = gpu_end - gpu_start

# GPU 계산 결과를 시각화할 수 있도록 CPU의 NumPy 배열로 복사한다.
# 이 복사 시간도 순수 GPU 계산 시간과 구분해 기록한다.

download_start = time.perf_counter()

output_cp_np = cp.asnumpy(output_cp)

download_end = time.perf_counter()

download_time = download_end - download_start


# CPU 시간 / GPU 시간으로 속도 향상을 계산한다. 1보다 크면 GPU가 더 빠르다.
# 먼저 데이터 전송을 제외한 순수 계산 시간끼리 비교한다.
compute_speedup = cpu_time / gpu_time

# 실제 데이터 이동 비용도 확인하기 위해 업로드, 계산, 다운로드 시간을 더한다.
# 이 합계에는 워밍업 시간이 포함되지 않는다.
gpu_total_time = upload_time + gpu_time + download_time
total_speedup = cpu_time / gpu_total_time


# CPU 계산 시간과 GPU의 단계별 시간, 두 기준의 속도 향상을 출력한다.
print("========== NumPy CPU ==========")
print(f"Processing time: {cpu_time:.4f} seconds")
print()

print("========== CuPy GPU ==========")
print(f"CPU to GPU: {upload_time:.4f} seconds")
print(f"GPU processing: {gpu_time:.4f} seconds")
print(f"GPU to CPU: {download_time:.4f} seconds")
print(f"Total GPU time: {gpu_total_time:.4f} seconds")
print()

print("========== Comparison ==========")
print(f"Compute-only speedup: {compute_speedup:.2f}x")
print(f"Including transfers: {total_speedup:.2f}x")

plt.figure(figsize=(18,12))

plt.subplot(1,3,1), plt.imshow(input_np, cmap="gray"), plt.title("Original Image")
plt.subplot(1,3,2), plt.imshow(output_np, cmap="gray"), plt.title("Motion Trail (CPU)")
plt.subplot(1,3,3), plt.imshow(output_cp_np, cmap="gray"), plt.title("Motion Trail (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")
    
    
