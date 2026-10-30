import torch
import torchvision

print("PyTorch version:", torch.__version__)
print("Torchvision version:", torchvision.__version__)

import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 같은 조건의 실험을 비교하기 위해 고정한다.
seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="src/datasets",
    train=True,
    transform=transform,
    download=True,
)

test_dataset = datasets.MNIST(
    root="src/datasets",
    train=False,
    transform=transform,
    download=True,
)


print(
    "Train data:",
    len(train_dataset)
)

print(
    "Test data:",
    len(test_dataset)
)


image, label = train_dataset[0]

print("Image type:", type(image))
print("Image shape:", image.shape)
print("Label:", label)
print("Minimum:", image.min().item())
print("Maximum:", image.max().item())

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1000,
    shuffle=False,
    num_workers=0,
)
 
images, labels = next(
    iter(train_loader)
)

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

images, labels = next(
    iter(test_loader)
)

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)


class SimpleCNN(nn.Module):
    def __init__(self):
        
        super().__init__()

        
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.fc1 = nn.Linear(
            16 * 7 * 7,
            64,
        )

        self.fc2 = nn.Linear(
            64,
            10,
        )

    def forward(self, x, return_features=False):
        conv1_output = self.conv1(x)
        relu1_output = F.relu(conv1_output)
        # [B, 8, 28, 28] → [B, 8, 14, 14].
        pool1_output = self.pool(relu1_output)
        # [B, 8, 14, 14] → [B, 16, 14, 14].
        conv2_output = self.conv2(pool1_output)
        relu2_output = F.relu(conv2_output)
        # [B, 16, 14, 14] → [B, 16, 7, 7].
        pool2_output = self.pool(relu2_output)
        flattened = torch.flatten(pool2_output, start_dim=1)
        # [B, 784] → [B, 64] → [B, 10].
        fc1_output = F.relu(self.fc1(flattened))
        logits = self.fc2(fc1_output)

        if return_features:
            features = {
                "conv1": conv1_output,
                "relu1": relu1_output,
                "pool1": pool1_output,
                "conv2": conv2_output,
                "relu2": relu2_output,
                "pool2": pool2_output,
            }

            return logits, features

        return logits
    
    
    
device = torch.device("cpu")

torch.manual_seed(seed)

model = SimpleCNN()
model = model.to(device)

print("Device:", device)
print(model)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)



def train_one_epoch(model, data_loader, criterion, optimizer, device):
    
    model.train()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    for images, labels in data_loader:
        
        images = images.to(device)
        labels = labels.to(device)

        
        optimizer.zero_grad()

    
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        
        batch_size = images.size(0)

        total_loss += loss.item() * batch_size                 
        predictions = logits.argmax(dim=1)                    
        correct_count += (predictions == labels).sum().item()  
        sample_count += batch_size                            

    average_loss = total_loss / sample_count
    accuracy = correct_count / sample_count

    return average_loss, accuracy

# 테스트 데이터에 대한 예측 성능만 측정하고, optimizer로 가중치를 수정하지 않는다.
def evaluate(model, data_loader, criterion, device):
    
    model.eval()

    total_loss = 0.0
    correct_count = 0
    sample_count = 0

    with torch.no_grad():
        
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            
            loss = criterion(logits, labels)

            batch_size = images.size(0)

           
            total_loss += loss.item() * batch_size               
            predictions = logits.argmax(dim=1)                    
            correct_count += (predictions == labels).sum().item()  
            sample_count += batch_size                            


        average_loss = total_loss / sample_count
        accuracy = correct_count / sample_count

        return average_loss, accuracy
    
    
epochs = 3

train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []


training_start_time = time.perf_counter()

for epoch in range(epochs):
    epoch_start_time = time.perf_counter()

    
    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
    )

    
    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)
    test_loss_history.append(test_loss)
    test_accuracy_history.append(test_accuracy)

    epoch_time = time.perf_counter() - epoch_start_time

    # 정확도는 0~1 비율이므로 100을 곱해 백분율로 표시한다.
    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: "
        f"{train_accuracy * 100:.2f}% | "
        f"Test loss: {test_loss:.4f} | "
        f"Test accuracy: "
        f"{test_accuracy * 100:.2f}% | "
        f"Time: {epoch_time:.2f}s"
    )

# 모든 epoch의 학습과 테스트 평가를 마치는 데 걸린 시간을 출력한다.
total_training_time = time.perf_counter() - training_start_time
print(f"Total CPU training time: {total_training_time:.2f}s")

# 테스트 이미지 한 장을 골라 정답과 예측을 비교한다.
image, true_label = test_dataset[0]          
input_batch = image.unsqueeze(0).to(device) 

model.eval()

with torch.no_grad():
    
    logits = model(input_batch)                          
    probabilities = torch.softmax(logits, dim=1)          
    predicted_label = probabilities.argmax(dim=1).item()  
    
print("True label:", true_label)
print("Predicted label:", predicted_label)
print("Confidence:", probabilities[0, predicted_label].item())

plt.figure(figsize=(4,4))
plt.imshow(image.squeeze(0), cmap="gray")
plt.title(f"True: {true_label}, " f"Prediction: {predicted_label}")
plt.axis("off")
plt.show()

plt.figure(figsize=(15,10))

num_images = 40

# 여러 이미지를 확인할 때도 평가 모드와 gradient 기록 중단을 함께 사용한다.
model.eval()

with torch.no_grad():
    for i in range(num_images):
        image, true_label = test_dataset[i]
        input_batch = image.unsqueeze(0).to(device)

        logits = model(input_batch)
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = probabilities.argmax(dim=1).item()
        # 첫 번째 이미지에서 예측한 숫자의 확률을 꺼내 숫자로 바꾼다.
        confidence = probabilities[0, predicted_label].item()

        plt.subplot(4, 10, i+1)
        plt.imshow(image.squeeze(0), cmap="gray")
        plt.title(
            f"True: {true_label}\n"
            f"Pred: {predicted_label}\n"
            f"Conf: {confidence*100:.1f}%"
        )
        plt.axis("off")

plt.tight_layout()
plt.show()


# 학습과 테스트의 Loss 변화를 같은 그래프에 표시해 과정을 비교한다.
# x 데이터를 생략했기 때문에 가로축은 기록 순서인 0, 1, 2로 표시된다.
plt.figure(figsize=(8,5))
# 각 epoch에서 얻은 Loss를 연결한다.
plt.plot(train_loss_history, marker = "o", label="train_loss")
# 테스트 Loss를 함께 그린다.
plt.plot(test_loss_history, marker = "o", label="test_loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN Loss")

plt.grid()
plt.legend()
plt.show()

 
# accuaracy를 통해 학습과 테스트에서 어떻게 달라지는지 비교한다.

plt.figure(figsize=(8,5))

# 저장된 0~1 정확도에 100을 곱해 백분율로 표시한다.
# 백분율로 표시하기 위해 저장된 0~1 accuaracy에 100을 곱한다.
plt.plot([accuracy * 100 for accuracy in train_accuracy_history], marker="o", label="train_accuracy")
# 테스트 정확도도 같은 단위로 바꿔 train accuracy와 함께 그린다.
plt.plot([test_accuracy * 100 for test_accuracy in test_accuracy_history],marker="o",
label = "test_accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("CNN Accuracy")
plt.grid()
plt.legend()
plt.show()


# 이 함수는 입력 batch의 첫 번째 이미지에 대한 채널별 결과를 보여 준다.


def show_feature_maps(feature_tensor, title, max_maps=8, cmap="viridis"):
    
    if feature_tensor.ndim != 4:
        raise ValueError("Feature Tensor는 [Batch, Channel, Height, Width] 형태여야 합니다.")

    # 첫 이미지의 결과만 골라 train 기록과 분리하고 CPU로 가져온다.
    feature_maps = feature_tensor[0].detach().cpu()

    # 실제 채널 수와 요청한 최대 개수 중 작은 값까지만 표시한다.
    channel_count = min(feature_maps.shape[0], max_maps)
    column_count = 4
    # 한 행에 4개씩 배치하고, 남는 채널이 있어도 다음 행을 만들도록 올림한다.
    row_count = int(np.ceil(channel_count / column_count))

    plt.figure(figsize=(12, 3 * row_count))

  
    for channel_index in range(channel_count):
        plt.subplot(row_count, column_count, channel_index + 1)
        plt.imshow(feature_maps[channel_index], cmap=cmap)
        plt.title(f"Channel {channel_index}")
        plt.axis("off")
    
    # 그림 전체에 해당 layer의 이름을 붙인다.
    plt.suptitle(title)    
    plt.tight_layout
    plt.show()
    
    
# 테스트 이미지 한 장을 선택한다.
image, true_label = test_dataset[0]

model.eval()

with torch.no_grad():
        input_batch = image.unsqueeze(0).to(device)

        # return_features=True면 최종 점수와 여섯 단계의 결과를 함께 반환한다.
        logits,features = model(input_batch,return_features=True)
        

# 첫 번째 convolution의 채널 8개를 보여 준다. 아직 ReLU를 적용하기 전의 값이다.
show_feature_maps( features["conv1"],
    title="Conv1 Feature Maps",
    max_maps=8,
    cmap="viridis" 
)

# 두 번째 convolution은 채널 16개이므로 네 행에 걸쳐 모두 표시한다.
show_feature_maps(features["conv2"],
    title="Conv2 Feature Maps",
    max_maps=16,
    cmap="viridis"                 
                  )

layer_names = [
    "conv1",
    "relu1",
    "pool1",
    "conv2",
    "relu2",
    "pool2"
]

for layer in layer_names : 
    show_feature_maps(
    features[layer],
    title=f"{layer} Feature Maps",
    max_maps=4,
    cmap="viridis"
    )
    
    
