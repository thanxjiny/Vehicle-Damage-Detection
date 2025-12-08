# 🚀 Step 3. YOLOv8x Fine-tuning (2nd Attempt)

베이스라인(Pre-trained) 성능을 넘어서기 위해, 커스텀 데이터셋(AI-Hub 파손 차량 + COCO)을 **YOLOv8x 모델을 이용해 Fine-tuning** 실행

## 🎯 Objective (실험 목표)
1.  **Domain Adaptation:** 일반적인 COCO 데이터셋뿐만 아니라, **심하게 파손된 차량(Damaged Car)** 데이터 분포에 모델을 적응시킴
2.  **Performance Boost:** 베이스라인 대비 **mAP(평균 정밀도)**와 **Recall(재현율)**을 얼마나 향상시킬 수 있는지 확인
3.  **Optimization:** 우리 데이터셋에 맞는 최적의 하이퍼파라미터(Epochs, Batch size 등)를 탐색

## 🛠 Experiment Setup (학습 환경)
* **Model:** YOLOv8x (Load weights from `yolov8x.pt`)
* **Environment:** Google Colab Pro (A100 / T4 GPU)

## Dataset
1. 데이터셋 구축 및 표준화 (Dataset Construction & Standardization)
 - **하이브리드 라벨링 전략** (Hybrid Labeling Strategy)
    - 1단계 (우선순위): Auto-Labeling (YOLOv8x)
        - 일단 Pre-trained 모델로 "차량 전체 형상"을 찾기
        - 성공 시: Normal 데이터와 기준이 같아지므로 베스트
    - 2단계 (Fallback): JSON 라벨 활용
        - 만약 모델이 너무 확대된(Zoom-in) 이미지라 차량을 못 찾으면(Empty), 그때 JSON의 파손 부위 좌표를 가져옴
    - 이유: 1단계에서 탐지 못 해도, 빈 파일(Background)로 두는 게 아니라 **Damaged의 labeling**을 활용해 파손 부분만이라도 '차량'으로 학습

2. 데이터 분할 및 격리
 - 모델의 암기(Memorizing)를 방지하고 객관적인 성능 검증을 위한 데이터 분리.
 - 비율 (Ratio): 전체 데이터를 랜덤으로 섞은  7 : 2 : 1 비율로 랜덤 분할.
    - Train (70%): 모델 가중치 업데이트용 (학습).
    - Val (20%): 학습 중 성능 모니터링 및 조기 종료(Early Stopping) 결정용.
    - Test (10%): 학습 과정에 절대 관여하지 않으며, 최종 성능 평가에만 사용
  
| class | count | ratio | 
| :---: | :---: | :---: | 
| Train | 1369 | 0.7 |
| Valid | 392 | 0.2 | 
| Test | 196 | 0.1 |  
| total | 1957 | 1.0 | 

### ⚙️ Hyperparameters
| Parameter | Value | Note |
| :--- | :--- | :--- |
| **Epochs** | 50 | 조기 종료(Early Stopping) 적용 여부 확인 필요 |
| **Batch Size** | 16 | GPU 메모리에 맞춰 조정 |
| **Img Size** | 640 | YOLOv8 기본 입력 크기 |
| **freeze** | 10 | pre-trained 모델의 back-bone 유지 |
| **Lr0** | 1e-5 | Initial Learning Rate.초기학습률. 이미 학습이 잘 된 모델이니 조금씩 수 |
| **Optimizer** | SGD / AdamW |학습 속도가 빠르고 설정에 덜 민감(Yolov8 기본) |
| **patience** | 15 |early-stopping 조절. 성능이 더 이상 좋아지지 않을때, epoch반복 |

## 📊 Training Results (학습 결과)
학습 완료 후 `model.val()`을 통해 얻은 최종 성능 지표입니다.

### 1. Metrics Comparison (베이스라인 vs 파인튜닝 1st vs 파인튜닝 2nd)

| Model | Accuracy | average inference speed | FPS | GPU | test | fail |비고 |
| :---: | :---: | :---: | :---: |:---: | :---: |:---: |:---: |
| **Baseline (pre-trained)** |88.71%| 48.23 ms/장 | 20.73 FPS |T4|1957 | 221 |no-tuning |
| **Fine-tuned. ver1.0** |88.27%| 20.60 ms/장 | 48.55 FPS |L4|196 | 23 | freeze10 + epoch 50 |
| **Fine-tuned. ver2.0** |97.45%| 20.12 ms/장 | 49.70 FPS |L4|196 | 5 | ver1.0 + hybrid labeling |

| **Baseline (pre-trained)** | **Fine-tuned. ver1.0** | **Fine-tuned. ver2.0** |
| :---: | :---: | :---: |
| ![Baseline](./results/01_detection/confusion_matrix_010.png) | ![Fine-tuned](./results/01_detection/confusion_matrix_fine_tuning_1st.png) | ![Fine-tuned2](./results/01_detection/confusion_matrix_fine_tuning_2nd.png) |

| Model | Class | Precision | Recall | f1 | 비고 |
| :---: | :---: | :---: | :---: | :--- | :--- |
| **Baseline (pre-trained)** |Non-Vehicle| 0.74 | 0.96 | 0.84 |  |
| **Baseline (pre-trained)** |Vehicle| 0.98 | 0.85 | 0.91 | |
| **Fine-tuned. ver1.0** |Non-Vehicle| 0.73 | 0.98 | 0.84 |  |
| **Fine-tuned. ver1.0** |Vehicle| 0.99 | 0.84 | 0.91 | |
| **Fine-tuned. ver2.0** |Non-Vehicle| 0.97 | 0.95 | 0.96 |  |
| **Fine-tuned. ver2.0** |Vehicle| 0.98 | 0.99 | 0.98 | |

| **model results** | 
| :---: | 
| ![Baseline](./results/02_train_results/results.png) | 

| **valid sample** | 
| :---: | 
| ![Baseline](./results/02_train_results/val_batch0_pred.png) | 


### 💡 Findings
* fine-tuning을 통해 Accuracy는 비약적으로 상승(88.71% > 97.45%)하였고, 특히 FN는 줄고, TP가 상승하였다.







## 원인 추정(1st 문제점 해결 여부)
- fine-tuning 1st 모델의 성능이 향상하지 못 했던 원인은 **학습 데이터 간의 "정답 기준"이 서로 다르기 때문**일 가능성으로 추정
 - Damaged 데이터 (JSON 기반):
    - JSON에 있던 bbox가 **차량 전체**가 아니라 **스크래치 등 파손된 부위**만 감싸고 있었을 확률이 높음.
 - Normal 데이터 (Auto-labeling):
    - YOLO가 자동으로 라벨링했으므로 **차량 전체**를 잡았음
   
|  damaged1 | damaged2 |
| :---: | :---: |
| ![FN_1](./results/01_detection/sample_damaged.png) | ![FN2](./results/01_detection/sample_damaged2.png) |

### 💡 Findings
- 학습 데이터 간의 정답 기준이 서로 다른 문제점을 **하이브리드 라벨링 전략**으로 개선확인
  - GT : 파손된 일부만 labeling
  - Predicted : 전체 차량 향상을 찾음

## 📝 Conclusion 
* **결론:** Fine-tuning을 진행했지만 성능의 차이가 없음. 그 이유는 damaged와 normal의 labeling 데이터의 불일치로 추정됨. damaged의 라벨의 bbox 좌표를 YOLO 포맷으로 변환했고, normal은 YOLO-AUTO LABELING으로 차량 전체를 학습했기 때문
