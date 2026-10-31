
from random import randint, gauss

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.data


#박스의 위치와 점수를 이미지에 그린다
def draw_bbox(img, coord1, coord2, color, thickness=3, txt=""):
   
    _, txt_h = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.rectangle(img, coord1, coord2, color, thickness)
    cv2.putText(img, txt, (coord1[0], coord1[1]-txt_h+15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# 박스를 IoU 계산에 사용할 두 모서리 좌표로 바꾼다.
def convert_bbox_coord(bbox):
    x, y, w, h = bbox
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    
    return x1, y1, x2, y2
    
    

# 두 박스가 얼마나 겹치는지 IoU로 계산해 NMS가 중복 여부를 판단하게 한다.
def calculate_iou(bbox_a, bbox_b):
    
    xa_1, ya_1, xa_2, ya_2 = bbox_a
    xb_1, yb_1, xb_2, yb_2 = bbox_b
    
    intersection_x1 = max(xa_1, xb_1)
    intersection_y1 = max(ya_1, yb_1)
    intersection_x2 = min(xa_2, xb_2)
    intersection_y2 = min(ya_2, yb_2)
    
    # 합집합을 구하기 위해 원래 박스의 넓이를 구한다.
    area_a = (xa_2 - xa_1) * (ya_2 - ya_1)
    area_b = (xb_2 - xb_1) * (yb_2 - yb_1)
    
    w = max(0, intersection_x2 - intersection_x1)
    h = max(0, intersection_y2 - intersection_y1)
    
    # 교집합의 넓이를 구한다.
    intersection_area = w * h
    
    # 합집합을 구한다.
    union_area = area_a + area_b - intersection_area
    
   
    return intersection_area / union_area if union_area > 0 else 0.0



def random_bbox(img, min_w=100, min_h=400, max_w=600, max_h=600):
  
    img_h, img_w = img.shape[:2]

    w = randint(min_w, min(max_w, img_w))
    h = randint(min_h, min(max_h, img_h))

    x1 = randint(0, img_w - w)
    y1 = randint(0, img_h - h)

    return x1, y1, x1 + w, y1 + h

def random_conf():
    return min(1.0, max(0.0, gauss(0.6, 0.1)))



hubble = skimage.data.hubble_deep_field()

N = 20

boxes = []
scores = []

for i in range(N):
    boxes.append(random_bbox(hubble))
    scores.append(random_conf())


for i, box in enumerate(boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )


plt.figure(figsize=(10,8))
plt.imshow(hubble)
plt.axis("off");



# 점수가 기준보다 낮은 박스를 제거해 신뢰도가 낮은 후보를 먼저 줄인다.
def prune_low_conf(boxes, scores, conf_threshold=0.5):

    pruned_boxes = []       # 기준을 통과한 박스 좌표
    pruned_scores = []     # 통과한 박스에 대응하는 점수
    
    # zip으로 같은 위치의 두 항목을 묶는다.
    for box, score in zip(boxes, scores):
        # 기준보다 낮은 점수만 제외한다.
        if score >= conf_threshold:
            
            pruned_boxes.append(box)
            pruned_scores.append(score)
    
    return pruned_boxes, pruned_scores
            
    
# 중복 박스 중 점수가 높은 박스를 남긴다.
def apply_nms(boxes, scores, iou_threshold=0.5):
    # 가장 높은 점수의 박스를 먼저 남겨야 겹치는 낮은 점수의 박스를 제외할 수 있다.
    # 점수만 정렬하면 해당 박스를 찾기 어렵기 때문에 argsort로 원래 위치를 정렬한다.
    # [::-1]로 높은 점수 순서로 뒤집고,리스트로 바꾼다.
    order = np.argsort(scores)[::-1].tolist()

    # 선택된 결과만 따로 저장하기 위한 리스트이다.
    nms_boxes = []
    nms_scores = []
    
    # 제거 후 남는 후보 수가 매번 달라지므로, 후보가 없어질 때까지 반복한다.
    while order:
        
        current = order[0]
        nms_boxes.append(boxes[current])
        nms_scores.append(scores[current])
        
        # 점수순으로 이웃하지 않은 박스도 겹칠 수 있으므로 나머지 모든 후보를 비교한다.
        # order[1:]을 통해 방금 확정한 대표 박스를 제외하여 자기 자신과 비교하는 일을 막는다.
        remaining = []
        for index in order[1:]:
            iou = calculate_iou(boxes[current], boxes[index])
            
            # 기준 이하면 대표 박스와 다르다고 보고 이후 비교 대상으로 남긴다.
            if iou <= iou_threshold:
                remaining.append(index)
        
        order = remaining
    
    return nms_boxes, nms_scores
                

print(len(boxes), len(scores))

new_boxes, new_scores = prune_low_conf(boxes,scores)
new_boxes, new_scores = apply_nms(new_boxes, new_scores, iou_threshold=0.2)

print(len(new_boxes), len(new_scores))

hubble = skimage.data.hubble_deep_field()

for i, box in enumerate(new_boxes):
    draw_bbox(
        hubble,
        (box[0], box[1]),
        (box[2], box[3]),
        (randint(0,255), randint(0,255), randint(0,255)),
        txt=f"{scores[i]:.2f}"
    )

plt.figure(figsize=(10,8))
plt.imshow(hubble)
plt.axis("off")
plt.show()

