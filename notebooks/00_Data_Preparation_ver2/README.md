### Step 1. Dataset Preparation

딥러닝 모델 학습을 위한 **차량 이미지 Dataset**을 구축하는 단계  
AI-Hub의 파손 차량 데이터와 COCO 데이터셋의 정상 차량/배경 이미지를 결합하여, **파손 탐지**뿐만 아니라 **오탐지(False Positive) 방지**까지 고려한 데이터셋을 설계

#### Objective (목표)
1.  **High Quality Data:** 원본 해상도를 유지하기 위해 리사이징 없이 수집
2.  **Balanced Composition:**  **파손 차량:** 파손 유형 학습.
    * **정상 차량:** 파손되지 않은 일반 차량을 구분하기 위해 추가.
    * **배경 이미지:** 차량이 아닌 물체(벤치, 신호등 등)를 차량으로 오인하지 않도록 학습(Negative Sample).
3.  **Quality Control:** 객체가 너무 작으면 학습에 방해가 되므로, 전체 이미지 면적의 **5% 이상**인 차량만 필터링하여 수집.

#### 1차. coco2017과 kaggle 데이터를 활용해 normal과 backgroud 데이터 추출

`00_dataset_preparation.ipynb` 코드를 수행하여 coco2017 데이터 추출

| Class (Category) | Source | Images | Labels | Note |
| :--- | :--- | :--- | :--- | :--- |
| **1. Damaged** | AI-Hub | **1,200** | 1,200 | AI-HUB 샘플 데이터 |
| **2. Normal** | COCO 2017 | **157** | - | 정상 차량 (면적 5% 이상 필터링 적용됨) |
| **3. Background** | COCO 2017 | **600** | - | 차량 없음 (Negative Samples) |
| **Total** | | **1,957** | | **✅ 구축 완료** |

> **Note:** COCO 2017 데이터 중 고품질 학습을 위해 **BBox Area Threshold(5%)** 를 엄격하게 적용하여 조건에 부합하는 157장만 선별

#### 2차. Dataset 추가
normal 데이터의 부족분을 kaggle 데이터셋에서 총 920개 추가  
`5_study1_yolov8x_fine_tuning_4th_kaggle_dataset.ipynb` 코드 수행

| Class (Category) | Source | Images | Labels | Note |
| :--- | :--- | :--- | :--- | :--- |
| **1. Damaged** | AI-Hub | **1,200** | 1,200 | 차량 파손 이미지 (Training Target) |
| **2. Normal** | COCO 2017+kaggle(920) | **1,077** | - | 정상 차량 (면적 5% 이상 필터링 적용됨) |
| **3. Background** | COCO 2017 | **600** | - | 차량 없음 (Negative Samples) |
| **Total** | | **2,877** | | **✅ 구축 완료** |

```text
Dataset_No_Resizing/
├── damaged/
│   ├── images/  # .jpg
│   └── labels/  # .txt (JSON)
├── normal/
│   └── images/  # .jpg (Labels not required for background usage)
├── normal(kaggle_dataset)/
│   └── images/  # .jpg (Labels not required for background usage)
└── background/
    └── images/  # .jpg
```

#### 3차. AI-HUB 데이터 샘플링(damaged)

* damage의 클래스별(4) 밸런스를 맞춰 데이터셋을 재구성
  - 샘플 데이터(1,200장)은 데이터가 부족하고, 클래스가 불균형한 상태
* 전체 AI-HUB 데이터(약50만장) 중 Train 폴더에서 10,000장, Valid 2,000장 랜덤으로 추출
* 전체 AI-HUB의 데이터를 Colab으로 옮길 수 없어 로컬pc에서 vs code로 수행  
`data_prep.py` 코드 수행

| ID | Class | Count |
| :--- | :--- | :--- |
| **0** | Scratched| 3,000 |
| **1** | Separated |  3,000 |
| **2** | Breakage |  3,000 |
| **3** | Crushed  | 3,000 |

```text
AI_HUB_DAMAGE_DATASET/
├── images/
│   ├── train/ (10,000장)
│   └── val/   (2,000장)
└── labels/
    ├── train/ (10,000개 .txt)
    └── val/   (2,000개 .txt)
```

#### 4차. 최종 데이터셋 정리
* 파손 데이터(AI-Hub)와 정상 차량(Normal), 그리고 오탐율을 줄이기 위한 배경 데이터(Background)를 결합하여 통합 차량 탐지 모델용 데이터셋을 생성
* YOLO v8x 모델이 자동 라벨링 시도 시, confidence threshold 0.1로 적용하여 최대한 차량을 인식하도록 유도

**하이브리드 라벨링 전략**
| 데이터소스 |  적용 전략 |수 | 설명 |
| :--- | :--- | :--- | :--- |
| **Damaged (AI-Hub)** | Hybrid| 12000| 1차로 YOLO v8x 모델이 자동 라벨링 시도, 실패 시 기존 원본 TXT 라벨을 차량(0) 클래스로 변환하여 보완 |
| **Normal (COCO2017 + Kaggle)** |Auto | 1072| 사전 학습된 YOLO v8x 모델을 활용하여 이미지 내 차량 객체를 자동으로 탐지 및 라벨링 |
| **Background(COCO2017)** | Empty | 605|  차량이 없는 배경 이미지를 'Empty Label'로 처리 |
|**전체**|-| 13,677 |-|  
 * Normal에서 Yolo v8x를 통해 차량으로 인식되지 못한 5개 Backgroud 적용
 * Normal = 157 + 920 - 5
 * Background 605(4.4%)

`6_study1_yolov8_fine_tuning_aihub.ipynb` 코드 수행

```text
CAR_DETECTION_AIHUB_KAGGLE_CONF01/ (전체/Background)
├── images/
│   ├── train/ (9,473 / 447)
│   └── val/   (2,807 /  98)
│   └── test/  (1,397 /  60) 
└── labels/
│   ├── train/ (9,473 txt 파일)
│   └── val/   (2,807 txt 파일)
│   └── test/  (1,397 txt 파일)
```


