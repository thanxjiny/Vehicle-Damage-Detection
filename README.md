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
| **Data Set** | **Step 0** | 데이터셋 구축 | AI-Hub + COCO + Kaggle| ✅ Done | [상세보기](./notebooks/00_Data_Preparation/README.md) |
| **Car Detection** | **DJ.Step 1** | 차량 인식 베이스라인 | YOLOv8x (Pre-trained) | ✅ Done | [상세보기](./notebooks/01_Baseline_Inference/README.md) |
| **Car Detection** | **DJ.Step 2** | 차량 인식 파인튜닝_1st | YOLOv8 Custom | ✅ Done |  [상세보기](./notebooks/02_Car_Detection_FineTuning_1st/README.md) | |
| **Car Detection** | **DJ.Step 3** | 차량 인식 파인튜닝_2nd | YOLOv8 Custom | ✅ Done | [상세보기](./notebooks/03_Car_Detection_FineTuning_2nd/README.md) | |
| **Car Detection** | **DJ.Step 4** | 차량 인식 파인튜닝_3rd | YOLOv8 Custom | ✅ Done | [상세보기](./notebooks/04_Car_Detection_FineTuning_3rd/README.md) | |
| **Damage Detection** | **DJ.Step 1** | 파손 인식 파인튜닝_1st | YOLOv8 Custom | ✅ Done | [상세보기](./notebooks/05_Damage_Detection_FineTuning_1st/README.md) | |
| **Damage Detection** | **DJ.Step 2** | 파손 인식 파인튜닝_2nd | YOLOv8 Custom | ✅ Done | [상세보기](./notebooks/06_Damage_Detection_FineTuning_2nd/README.md) | |



## Tech Stack
* Python 3.10
* PyTorch
* Ultralytics YOLOv8
