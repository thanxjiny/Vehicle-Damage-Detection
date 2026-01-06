# Vehicle Damage Inspector

딥러닝을 활용한 차량 파손 탐지 및 분류 프로젝트 스터디

## theory

| Topic | Status | Report |
| :---| :--- | :--- |
| object detection basic | ✅ Done | [상세보기](./theory/01_Object_Detection/README.md) |
| YOLO| ✅ Done | [상세보기](./theory/02_YOLO/README.md) |



## Project Roadmap & Study Log

| Subject| Stage | Topic | Model | Status | Report |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Data Set** | **Step 0** | 데이터셋 구축 | AI-Hub(1200) + COCO + Kaggle| ✅ Done | [상세보기](./notebooks/00_Data_Preparation/README.md) |
| **Data Set** | **Step 1** | 데이터셋 구축 | AI-Hub(12000) + COCO + Kaggle| ✅ Done | [상세보기](./notebooks/00_Data_Preparation_ver2/README.md) |
| **Car Detection** | **Step 1** | 차량 인식 베이스라인 | YOLOv8x (Pre-trained) | ✅ Done | [상세보기](./notebooks/01_Baseline_Inference/README.md) |
| **Car Detection** | **Step 2** | 차량 인식 1st | YOLOv8 | ✅ Done |  [상세보기](./notebooks/02_Car_Detection_FineTuning_1st/README.md) | |
| **Car Detection** | **Step 3** | 차량 인식 2nd | YOLOv8 (하이브리드) | ✅ Done | [상세보기](./notebooks/03_Car_Detection_FineTuning_2nd/README.md) | |
| **Car Detection** | **Step 4** | 차량 인식 3rd | YOLOv8 (+normal 920)| ✅ Done | [상세보기](./notebooks/04_Car_Detection_FineTuning_3rd/README.md) | |
| **Car Detection** | **Step 5** | 차량 인식 4th | YOLOv8 | - 삭제 - |  | |
| **Car Detection** | **Step 6** | 차량 인식 5th | YOLOv8 (12000) | ✅ Done | [상세보기](./notebooks/06_Car_Detection_FineTuning_5th/README.md) | |
| **Damage Detection** | **Step 1** | 파손 인식 1st | YOLOv8 | ✅ Done | [상세보기](./notebooks/05_Damage_Detection_FineTuning_1st/README.md) | |
| **Damage Detection** | **Step 2** | 파손 인식 2nd | YOLOv8 (+normal 920) | ✅ Done | [상세보기](./notebooks/06_Damage_Detection_FineTuning_2nd/README.md) | |
| **Damage Detection** | **Step 3** | 파손 인식 3rd | YOLOv8 (12000) | ✅ Done | [상세보기](./notebooks/07_Damage_Detection_FineTuning_3rd/README.md) | |
| **Damage Classification** | **Step 1** | 파손 유형 1st | YOLOv8 (12000) | ✅ Done | [상세보기](./notebooks/08_Damage_Classification_FineTuning_1st/README.md) | |

## Tech Stack
* Python 3.10
* PyTorch
* Ultralytics YOLOv8
