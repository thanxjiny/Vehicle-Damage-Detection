## 1. YOLO v1 (You Only Look Once)
> *"Unified, Real-Time Object Detection"*

### 💡 핵심 아이디어
기존의 2-Stage Detector(R-CNN 등)가 '후보 영역 추출 -> 분류'의 느린 과정을 거쳤다면, YOLO v1은 **이미지를 한 번만 보고(One-stage)** 바로 박스와 클래스를 예측합니다.

### ⚙️ 동작 원리 (Grid System)
1.  입력 이미지를 **S x S 그리드(Grid)**로 나눈다.
2.  각 그리드 셀은 **B개의 Bounding Box**와 **Confidence Score**를 예측한다.
3.  동시에 해당 그리드의 **Class Probability**를 예측한다.
4.  이 모든 것을 하나의 CNN 망으로 처리하여 속도가 매우 빠르다.

#### [YOLO 이해]

![Darknet](./images/darknet.jpg)

#### [YOLO ver1.0]
 
![yolov1_1](./images/yolo_v1_1.jpg)

#### [YOLO v1 Architecture]

![yolo v1 Architecture](./images/yolo_v1_2.jpg)

---

## 2. YOLO v2 (YOLO9000)
> *"Better, Faster, Stronger"*

v1의 단점(낮은 재현율, 부정확한 위치)을 보완하기 위해 나온 버전입니다.

### 🚀 주요 개선점
1.  **Anchor Boxes 도입:**
    * v1은 박스 크기를 처음부터 무작위로 예측해서 학습이 불안정했음.
    * v2는 미리 정의된 '앵커 박스(Anchor Box)'를 기준으로 **오프셋(Offset)**만 예측하여 학습 안정화.
2.  **Batch Normalization:** 모든 레이어에 BN을 추가하여 mAP 2% 향상.
3.  **High Resolution Classifier:** 학습 시 입력 해상도를 높여 작은 물체 탐지 성능 개선.

 ### [YOLO v2]

![yolo_v2](./images/yolo_v2_1.jpg)

#### [YOLO bounding box + confidence socre]

![yolov1_bounding box](./images/yolo_v2_2.jpg)

## 3. YOLO ver3.0
 -'작은 물체 탐지(Small Object Detection)' 성능 저하를 해결하기 위해 설걔. ResNet의 Residual 구조와 FPN(Feature Pyramid Network) 개념을 도입하여 성능과 속도 개선

 ### 🚀 주요 개선점
 1. Backbone: **Darknet-53구조**
    - 기존 Darknet-19에서 층을 대폭 늘려 53개의 Convolutional Layer를 사용
    - Skip Connection (Shortcut) 개념을 도입
    - 효과: 층이 깊어져도 학습이 원활하며(Gradient 소실 방지), 이미지의 추상적인 특징을 더 정교하게 추출함
 
![yolo_v3_1](./images/yolo_v3_1.jpg)
    
 2. Multi-Scale Prediction (FPN)YOLOv3의 가장 큰 혁신은 3가지 서로 다른 스케일(Scale)에서 물체를 탐지
    - Large Scale (13x13): 큰 물체 탐지 > Medium Scale (26x26): 중간 크기 물체 탐지 > Small Scale (52x52): 작은 물체 탐지
    - 결과: Feature Map을 Upsampling하여 이전 단계의 특징과 합치는(Concatenate) 방식으로, 작은 파손 부위나 멀리 있는 객체 탐지 성능이 비약적으로 상승함.
 
3. Class Classification (Softmax -> Sigmoid)변경: 기존의 Softmax(하나만 선택) 대신, 각 클래스별로 **Binary Cross Entropy (Sigmoid)**를 사용

![yolo_v3_2](./images/yolo_v3_2.jpg)

![yolo_v3_3](./images/yolo_v3_3.jpg)

4. Bounding Box Predictionv2의 Anchor Box 개념을 계승하되, 각 스케일(3개)마다 3개의 앵커를 할당하여 총 9개의 앵커 박스를 사용 (K-Means Clustering)

----

#### [YOLO v1~v3]

![yolo_history](./images/yolo_v1_3.jpg)
