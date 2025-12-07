# 🚀 Step 2. YOLOv8x Fine-tuning (1st Attempt)

베이스라인(Pre-trained) 성능을 넘어서기 위해, 커스텀 데이터셋(AI-Hub 파손 차량 + COCO)을 **YOLOv8x 모델을 이용해 Fine-tuning** 실행

## 🎯 Objective (실험 목표)
1.  **Domain Adaptation:** 일반적인 COCO 데이터셋뿐만 아니라, **심하게 파손된 차량(Damaged Car)** 데이터 분포에 모델을 적응시킴
2.  **Performance Boost:** 베이스라인 대비 **mAP(평균 정밀도)**와 **Recall(재현율)**을 얼마나 향상시킬 수 있는지 확인합니다.
3.  **Optimization:** 우리 데이터셋에 맞는 최적의 하이퍼파라미터(Epochs, Batch size 등)를 탐색

## 🛠 Experiment Setup (학습 환경)
* **Model:** YOLOv8x (Load weights from `yolov8x.pt`)
* **Environment:** Google Colab Pro (A100 / T4 GPU)

## Dataset
1. 데이터셋 구축 및 표준화 (Dataset Construction & Standardization)
 - 서로 다른 형식을 가진 데이터를 YOLO 학습 포맷(.txt)으로 통일하고 정답(Ground Truth)을 생성
 - 수행 방법
    - Damaged: 기존 JSON 라벨의 bbox 좌표를 YOLO 포맷으로 변환 (파손 부위도 '차량'으로 학습).
    - Normal: Pre-trained 모델(YOLOv8x)을 이용해 오토 라벨링(Auto-labeling) 수행.
    - Background: 빈 텍스트 파일 생성 (Negative Sample, "차량 없음"을 명시).
    - 모든 객체의 클래스 ID를 0 (Vehicle) 하나로 통합.

2. 데이터 분할 및 격리
 - 모델의 암기(Memorizing)를 방지하고 객관적인 성능 검증을 위한 데이터 분리.
 - 비율 (Ratio): 전체 데이터를 7 : 2 : 1 비율로 랜덤 분할.
    - Train (70%): 모델 가중치 업데이트용 (학습).
    - Val (20%): 학습 중 성능 모니터링 및 조기 종료(Early Stopping) 결정용.
    - Test (10%): 완전 격리(Isolation). 학습 과정에 절대 관여하지 않으며, 최종 성능 평가에만 사용
  
| class | count | ratio | 
| :---: | :---: | :---: | 
| Train | 1369 | 0.7 |
| Valid | 392 | 0.2 | 
| Test | 196 | 0.1 |  
| total | 1957 | 1.0 | 







### ⚙️ Hyperparameters
| Parameter | Value | Note |
| :--- | :--- | :--- |
| **Epochs** | 50 (예시) | 조기 종료(Early Stopping) 적용 여부 확인 필요 |
| **Batch Size** | 16 | GPU 메모리에 맞춰 조정 |
| **Img Size** | 640 | YOLOv8 기본 입력 크기 |
| **Optimizer** | SGD / AdamW | (자동 선택됨) |
| **Lr0** | 0.01 | Initial Learning Rate |

## 📊 Training Results (학습 결과)
학습 완료 후 `model.val()`을 통해 얻은 최종 성능 지표입니다.

### 1. Metrics Comparison (베이스라인 vs 파인튜닝)
| Model | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **Baseline (Step 1)** | 0.XX | 0.XX | 0.XX | 0.XX |
| **Fine-tuned (Step 2)** | **0.XX** | **0.XX** | **0.XX** | **0.XX** |
> **Analysis:** Fine-tuning 결과 mAP@50이 약 **+0.XX** 상승했습니다. 특히 (Recall/Precision) 측면에서 개선이 두드러졌습니다.

### 2. Training Curves (학습 로그)
학습 진행에 따른 Loss 감소와 mAP 상승 추이입니다.
![Results Graph](runs/detect/train/results.png)
*(위 경로는 학습 후 생성된 `runs/detect/train/results.png` 파일을 `results/` 폴더로 옮긴 후 연결하세요)*

### 3. Confusion Matrix
모델이 배경(Background)과 차량(Car)을 얼마나 잘 구분하는지 보여줍니다.
![Confusion Matrix](runs/detect/train/confusion_matrix.png)

## 🖼 Validation Examples
실제 학습된 모델이 검증 데이터(Validation Set)를 추론한 결과입니다.

| Ground Truth (정답) | Prediction (예측) |
| :---: | :---: |
| ![GT](runs/detect/train/val_batch0_labels.jpg) | ![Pred](runs/detect/train/val_batch0_pred.jpg) |

## 📝 Conclusion & Next Step
* **결론:** Fine-tuning을 통해 파손된 차량에 대한 검출 능력이 강화되었습니다. 특히 베이스라인에서 놓치던 (심한 파손/특이 각도) 차량들을 더 안정적으로 잡아냅니다.
* **Next Step:**
    * 차량 탐지(Object Detection) 성능은 충분히 확보되었습니다.
    * 이제 검출된 차량 영역(Crop) 내에서 **파손의 종류(Scratch, Dent)를 분류**하거나 **파손 부위를 세그멘테이션(Segmentation)** 하는 모델 개발로 넘어갑니다.
