### Step 0. Dataset Preparation

딥러닝 모델 학습을 위한 **차량 이미지 데이터셋(Dataset)**을 구축하는 단계  
AI-Hub의 파손 차량 데이터와 COCO 데이터셋의 정상 차량/배경 이미지를 결합하여, **파손 탐지(Damage Detection)**뿐만 아니라 **오탐지(False Positive) 방지**까지 고려한 데이터셋을 설계

#### Objective (목표)
1.  **High Quality Data:** 원본 해상도를 유지하기 위해 리사이징 없이 수집 (`Dataset_No_Resizing`).
2.  **Balanced Composition:** * **파손 차량:** 파손 유형 학습.
    * **정상 차량:** 파손되지 않은 일반 차량을 구분하기 위해 추가.
    * **배경 이미지:** 차량이 아닌 물체(벤치, 신호등 등)를 차량으로 오인하지 않도록 학습(Negative Sample).
3.  **Quality Control:** 객체가 너무 작으면 학습에 방해가 되므로, 전체 이미지 면적의 **5% 이상**인 차량만 필터링하여 수집.

#### Dataset Statistics (구축 결과)
`00_dataset_preparation.ipynb` 코드를 수행하여 생성된 최종 데이터셋 현황

| Class (Category) | Source | Images | Labels | Note |
| :--- | :--- | :--- | :--- | :--- |
| **1. Damaged** | AI-Hub | **1,200** | 1,200 | 차량 파손 이미지 (Training Target) |
| **2. Normal** | COCO 2017 | **157** | - | 정상 차량 (면적 5% 이상 필터링 적용됨) |
| **3. Background** | COCO 2017 | **600** | - | 차량 없음 (Negative Samples) |
| **Total** | | **1,957** | | **✅ 구축 완료** |

> **Note:** Normal 차량의 목표는 600장이었으나, 고품질 학습을 위해 **BBox Area Threshold(5%)**를 엄격하게 적용하여 조건에 부합하는 157장만 선별

#### Directory Structure
데이터셋은 아래와 같은 폴더 구조로 저장됩니다.

```text
Dataset_No_Resizing/
├── damaged/
│   ├── images/  # .jpg
│   └── labels/  # .txt (JSON)
├── normal/
│   └── images/  # .jpg (Labels not required for background usage)
└── background/
    └── images/  # .jpg
