# Vehicle Classification & Damage Detection Project

## Project Overview

* **Task:** Multi-class Image Classification
* **Classes:** 1. `normal_car` (정상 차량)
  2. `damaged_car` (파손 차량)
  3. `non_car` (차량 아님 / 배경)

---

## Dataset Creation

다양한 환경(조명, 각도, 차종, 배경)에서도 강건한(Robust) 모델을 만들기 위해 총 6개의 자체 구축 데이터셋을 병합하여 사용

클래스 불균형을 방지하고 정확한 평가를 진행하기 위해 전체 데이터를 **Train 8 : Validation 1 : Test 1** 비율로 층화 추출(Stratified Split)하여 구축

### 📊 데이터셋 구성표

| 데이터셋명 (Dataset) | 출처 (Source) | 구분 (Class) | 이미지 수 (Images) |비고|code|
| :--- | :--- | :--- | :--- |:--- |:--- |
| **SAMPLE_aihub_damaged_car_data** | AI Hub 160.차량파손 이미지 데이터 | 파손 차량 (`damaged_car`) | 12,200장 |valid의 50,445개 중 샘플링|01_aihub_damaged_car_data_sampler.py |
| **SAMPLE_aihub_normal_car** | AI Hub 091.차량량 외관 영상 데이터 | 정상 차량 (`normal_car`) | 12,000장 |차종 및 트림에 따라 동일한 이미지수 샘플링|03_aihub_normal_car_data_sampler.py |
| **SAMPLE_kaggle_normal_car** | Kaggle | 정상 차량 (`normal_car`) | 920장 |04_kaggle_normal_cars.py 
| **SAMPLE_aihub_ocr_nocar** | AI Hub 공공행정문서 OCR | 차량 아님 (`non_car`) | 200장 |공공문서 OCR 샘플|05_aihub_ocr_noncar.py |
| **SAMPLE_auto_crop_parking_lot** | AI Hub 주차 공간 탐색을 위한 차량 관점 복합 데이터| 차량 아님 (`non_car`) | 308장 |Yolo_v8l 차량부분 크롭| 
| **SAMPLE_coco2017_nocar** | COCO 2017 | 차량 아님 (`non_car`) | 1000장 |10개 객체별 100장씩 샘플링|
| **Total** | - | **3개 클래스** | **총 [26,000]장** |

> *Tip: 모든 이미지는 학습 전 `224x224` (Swin V2는 `256x256`) 해상도로 리사이즈 및 정규화되었으며, 과적합(Overfitting) 방지를 위해 ColorJitter, HorizontalFlip 등 증강(Augmentation) 기법이 적용되었습니다.*

---

## 🤖 Models Introduced

본 프로젝트에서는 AI 비전 분야의 발전을 주도한 대표적인 아키텍처 4가지를 선정하여 동일한 조건에서 성능을 겨루었습니다.



1. **ResNet50 (Baseline)**
   * 딥러닝 비전의 표준이자 기준점(Baseline)이 되는 전통적인 CNN 아키텍처입니다. 안정적인 성능을 보장합니다.
2. **ConvNeXt V2 (Tiny)**
   * 최신 트랜스포머의 설계 철학을 CNN에 이식하여 CNN의 한계를 돌파한 퓨어 CNN 모델입니다. 
3. **Swin Transformer V2 (Tiny)**
   * 이미지를 윈도우(Window) 단위로 쪼개어 계층적으로 분석하는 비전 트랜스포머 모델로, 차량 전체의 모습과 미세한 파손 부위를 동시에 캐치하는 데 탁월합니다.
4. **YOLOv8 (Classification)**
   * 초고속 객체 탐지로 유명한 YOLO의 분류 전용 모델입니다. 파라미터가 매우 가벼워 실시간 모바일/엣지 디바이스 환경에 최적화되어 있습니다.

---

## 🏆 Performance Comparison

동일한 Test Dataset으로 최종 검증을 진행한 결과입니다. (정확도와 F1-Score는 높을수록, FPS는 높을수록, Params는 낮을수록 좋습니다.)

| Model | Accuracy (%) | F1-Score | FPS (추론 속도) | Params (M) |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet50** | [90.50]% | [0.9010] | [85.2] | 23.5M |
| **ConvNeXt V2 (Tiny)** | [92.10]% | [0.9190] | [65.4] | 28.6M |
| **Swin V2 (Tiny)** | [93.40]% | [0.9320] | [55.8] | 28.3M |
| **YOLOv8 (Classification)** | [91.80]% | [0.9150] | **[145.0]** | **2.7M** |

### 💡 Conclusion (결론 요약)
* **최고 정확도 모델:** `[가장 정확도가 높은 모델명 기입, 예: Swin V2]`가 파손 차량의 미세한 특징을 가장 잘 잡아내어 가장 높은 정확도를 기록했습니다.
* **최고의 실용성 모델:** `YOLOv8-cls`는 파라미터 수가 가장 적으면서도 압도적인 추론 속도(FPS)를 보여주어, 실제 서비스 도입 시 비용 대비 효율(ROI)이 가장 뛰어날 것으로 판단됩니다.