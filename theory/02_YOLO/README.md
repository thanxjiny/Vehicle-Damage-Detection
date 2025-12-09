## 2. YOLO v1 (You Only Look Once)
> *"Unified, Real-Time Object Detection"*

### 💡 핵심 아이디어
기존의 2-Stage Detector(R-CNN 등)가 '후보 영역 추출 -> 분류'의 느린 과정을 거쳤다면, YOLO v1은 **이미지를 한 번만 보고(One-stage)** 바로 박스와 클래스를 예측합니다.

### ⚙️ 동작 원리 (Grid System)
1.  입력 이미지를 **S x S 그리드(Grid)**로 나눈다.
2.  각 그리드 셀은 **B개의 Bounding Box**와 **Confidence Score**를 예측한다.
3.  동시에 해당 그리드의 **Class Probability**를 예측한다.
4.  이 모든 것을 하나의 CNN 망으로 처리하여 속도가 매우 빠르다.

#### [YOLO 이해]
 
![yolov1_1](./images/yolo_v1_1.jpg)

![yolov1_bounding box](./images/yolo_v2_2.jpg)

![yolov1](./images/yolo_v1_2.jpg)

![yolo_history](./images/yolo_v1_3.jpg)

---

## 3. YOLO v2 (YOLO9000)
> *"Better, Faster, Stronger"*

v1의 단점(낮은 재현율, 부정확한 위치)을 보완하기 위해 나온 버전입니다.

### 🚀 주요 개선점
1.  **Anchor Boxes 도입:**
    * v1은 박스 크기를 처음부터 무작위로 예측해서 학습이 불안정했음.
    * v2는 미리 정의된 '앵커 박스(Anchor Box)'를 기준으로 **오프셋(Offset)**만 예측하여 학습 안정화.
2.  **Batch Normalization:** 모든 레이어에 BN을 추가하여 mAP 2% 향상.
3.  **High Resolution Classifier:** 학습 시 입력 해상도를 높여 작은 물체 탐지 성능 개선.
