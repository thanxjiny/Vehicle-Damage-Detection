# Vehicle Classification & Damage Detection Project

## Project Overview

* **Task:** Multi-class Image Classification
* **Classes:**
  1. `normal_car` (정상 차량)   
  2. `damaged_car` (파손 차량)  
  3. `non_car` (차량 아님 / 배경)  

---

## Dataset Creation

다양한 환경(조명, 각도, 차종, 배경)에서도 강건한(Robust) 모델을 만들기 위해 총 6개의 자체 구축 데이터셋을 병합하여 사용

클래스 불균형을 방지하고 정확한 평가를 진행하기 위해 전체 데이터를 **Train 8 : Validation 1 : Test 1** 비율로 층화 추출(Stratified Split)하여 구축

### 데이터셋 구성표

| 데이터셋명 (Dataset) | 출처 (Source) | 구분 (Class) | 이미지 수 (Images) |비고|
| :--- | :--- | :--- | :--- |:--- |
| **SAMPLE_aihub_damaged_car_data** | AI Hub 160.차량파손 이미지 데이터 | 파손 차량 (`damaged_car`) | 12,200장 |valid의 50,445개 중 샘플링|
| **SAMPLE_aihub_normal_car** | AI Hub 091.차량량 외관 영상 데이터 | 정상 차량 (`normal_car`) | 12,000장 |차종 및 트림에 따라 동일한 이미지수 샘플링|
| **SAMPLE_kaggle_normal_car** | Kaggle | 정상 차량 (`normal_car`) | 920장 |
| **SAMPLE_aihub_ocr_nocar** | AI Hub 공공행정문서 OCR | 차량 아님 (`non_car`) | 200장 |공공문서 OCR 샘플|
| **SAMPLE_auto_crop_parking_lot** | AI Hub 주차 공간 탐색을 위한 차량 관점 복합 데이터| 차량 아님 (`non_car`) | 308장 |Yolo_v8l 차량부분 크롭| 
| **SAMPLE_coco2017_nocar** | COCO 2017 | 차량 아님 (`non_car`) | 1000장 |10개 객체별 100장씩 샘플링|
| **Total** | - | **3개 클래스** | **26,628장** |

> *Tip: 모든 이미지는 학습 전 `224x224` (Swin V2는 `256x256`) 해상도로 리사이즈 및 정규화
---

### 데이터 분할 결과 (Train / Valid / Test)

| 데이터셋명 | Train (80%) | Valid (10%) | Test (10%) | 합계 |
| :--- | :--- | :--- | :--- | :--- |
| **aihub_damaged_car** | 9,760 | 1,220 | 1,220 | 12,200 |
| **aihub_normal_car** | 9,600 | 1,200 | 1,200 | 12,000 |
| **kaggle_normal_car** | 736 | 92 | 92 | 920 |
| **aihub_ocr_nocar** | 160 | 20 | 20 | 200 |
| **auto_crop_parking_lot** | 246 | 30 | 32 | 308 |
| **coco2017_nocar** | 800 | 100 | 100 | 1,000 |
| **최종 데이터셋 구성** | **21,302** | **2,662** | **2,664** | **26,628** |

## Models Introduced

1. **ResNet50 (Baseline)**
   * 딥러닝 비전의 표준이자 기준점(Baseline)이 되는 전통적인 CNN 아키텍터로 안정적인 성능을 보장
2. **ConvNeXt V2 (Tiny)**
   * 최신 트랜스포머의 설계 철학을 CNN에 이식하여 CNN의 한계를 돌파한 퓨어 CNN 모델
3. **Vision Transformer (ViT - Tiny)**
   * 이미지를 여러 개의 패치(Patch)로 분할하여 자연어 처리의 Self-Attention 메커니즘을 비전 분야에 최초로 적용한 순수 트랜스포머 모델. 이미지 전체의 전역적(Global) 문맥을 파악하는 데 우수 
4. **Swin Transformer V2 (Tiny)**
   * 이미지를 윈도우(Window) 단위로 쪼개어 계층적으로 분석하는 비전 트랜스포머 모델로, 차량 전체의 모습과 미세한 파손 부위를 동시에 캐치하는 데 탁월
5. **YOLOv8 (Classification)**
   * 초고속 객체 탐지로 유명한 YOLO의 분류 전용 모델. 파라미터가 매우 가벼워 실시간 모바일/엣지 디바이스 환경에 최적화
---

## Performance Comparison (test data 2,664장)

| Model | Accuracy (%) | F1-Score | FPS (추론 속도) | Params (M) |Fail|
| :--- | :--- | :--- | :--- | :--- |:--- |
| **ResNet50** | 99.81 | 99.81 | 149.68 | 23.51 |5|
| **ConvNeXt V2 (Tiny)** | 99.89 | 99.89 | 84.15 |  |3|
| **ViT (Tiny)** | 99.85|99.85| 69.93 | 5.52 |4|
| **Swin V2 (Tiny)** | 99.96 | 99.96 | 53.56 | 27.58 |1|
| **YOLOv8n-cls** | 99.36 | 99.36 | 38.05 | 1.44 |17|

## Conclusion (결론 요약)
테스트 결과, 모든 모델이 99% 이상의 뛰어난 정확도를 달성하여 구축된 데이터셋의 높은 품질을 증명

* **최고 정확도 모델 (Swin V2):** 2,664장의 테스트 이미지 중 **단 1건의 오탐(Accuracy 99.96%)**만을 기록하며 압도적인 1위를 차지. 차량 전체의 문맥(Global)과 미세한 파손 부위(Local)를 동시에 파악하는 계층적 트랜스포머 구조가 가장 효과적임을 입증
* **최고 속도 모델 (ResNet50):** 딥러닝 비전의 표준 모델답게 **149.68 FPS**라는 가장 빠른 추론 속도를 보여주었으며, 오탐 역시 5건으로 훌륭하게 방어해 내어 대규모 서버 환경의 실시간 처리에 가장 유리
* **초경량화 모델 (YOLOv8-cls):** 파라미터 수가 **1.44M**으로 다른 모델 대비 압도적으로 가벼우나, 상대적으로 오탐(17건)이 다소 발생했으나, 스마트폰이나 엣지(Edge) 디바이스 등 컴퓨팅 자원이 극도로 제한된 환경에서는 최고의 선택지가 될 수 있음

