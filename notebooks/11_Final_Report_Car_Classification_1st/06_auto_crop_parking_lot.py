import os
import glob
import random
import cv2
import json
from tqdm import tqdm
from ultralytics import YOLO

# ==============================================================================
# ⚙️ [설정] 사용자 로컬 경로 설정
# ==============================================================================

# 1. 주차장 원본 폴더 (라벨링 없음, 차가 섞여있음)
INPUT_DIR = r"D:\AIHUB\주차 공간 탐색을 위한 차량 관점 복합 데이터_차량복수\New_Sample (1)\원천데이터\TS1_실내_대형주차장\실내\대형주차장\대형주차장_004\Camera"

# 2. 결과물이 저장될 SAMPLE 폴더
OUTPUT_DIR = r"D:\AIHUB\주차 공간 탐색을 위한 차량 관점 복합 데이터_차량복수\SAMPLE_auto_crop_parking_lot"

# 3. 크롭할 배경의 고정 크기 (해상도 저하 방지를 위해 크게 설정)
CROP_SIZE = 600 # 600x600 픽셀

# 4. 이미지당 최대 몇 장의 배경을 뽑아낼 것인가?
MAX_CROPS_PER_IMAGE = 2 

# ==============================================================================

def auto_crop_backgrounds():
    print("🤖 YOLOv8 모델을 불러오는 중... (최초 실행 시 다운로드될 수 있습니다)")
    # 가장 가볍고 빠른 YOLOv8 nano 모델 사용
    model = YOLO("yolov8l.pt") 
    
    img_out_dir = os.path.join(OUTPUT_DIR, 'images')
    lbl_out_dir = os.path.join(OUTPUT_DIR, 'labels')
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    # 유효한 이미지 파일 수집
    search_pattern = os.path.join(INPUT_DIR, "*.*")
    image_files = [f for f in glob.glob(search_pattern) if f.lower().endswith(('.jpg', '.png'))]
    
    print(f"👉 총 {len(image_files)}장의 주차장 사진을 스캔합니다.")
    
    success_crops = 0
    pbar = tqdm(total=len(image_files))

    for img_path in image_files:
        pbar.update(1)
        
        try:
            # OpenCV로 이미지 읽기
            img = cv2.imread(img_path)
            if img is None: continue
            
            h, w, _ = img.shape
            
            # 이미지가 크롭 사이즈보다 작으면 패스
            if h < CROP_SIZE or w < CROP_SIZE:
                continue

            # 1. YOLO로 이미지 내의 차량(자동차, 버스, 트럭 등) 탐지
            results = model(img, verbose=False, classes=[2, 5, 7]) # 2:car, 5:bus, 7:truck
            
            # 탐지된 차량들의 바운딩 박스 리스트 (x1, y1, x2, y2)
            car_boxes = []
            if len(results) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                for box in boxes:
                    car_boxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])

            crops_found = 0
            attempts = 0
            
            # 2. 안전한 빈 공간 찾기 루프 (최대 50번 던져봄)
            while crops_found < MAX_CROPS_PER_IMAGE and attempts < 50:
                attempts += 1
                
                # 랜덤 좌표로 600x600 창틀을 던짐
                x1 = random.randint(0, w - CROP_SIZE)
                y1 = random.randint(0, h - CROP_SIZE)
                x2 = x1 + CROP_SIZE
                y2 = y1 + CROP_SIZE
                
                # 3. 차량 박스와 겹치는지(충돌) 검사
                overlap = False
                for cb in car_boxes:
                    # 두 박스가 겹치는지 확인하는 로직
                    if not (x2 <= cb[0] or x1 >= cb[2] or y2 <= cb[1] or y1 >= cb[3]):
                        overlap = True
                        break # 하나라도 겹치면 이 창틀은 실패!
                
                if not overlap:
                    # 4. 겹치지 않는 순수 배경 발견! 크롭하여 저장
                    cropped_img = img[y1:y2, x1:x2]
                    
                    base_name = os.path.splitext(os.path.basename(img_path))[0]
                    new_filename = f"parking_bg_{base_name}_crop{crops_found}.jpg"
                    new_json_name = f"parking_bg_{base_name}_crop{crops_found}.json"
                    
                    dst_img_path = os.path.join(img_out_dir, new_filename)
                    cv2.imwrite(dst_img_path, cropped_img)
                    
                    # 5. Non-Car JSON 생성
                    noncar_json_data = {
                        "info": {"name": "parking_background"},
                        "images": {"id": 1, "width": CROP_SIZE, "height": CROP_SIZE, "file_name": new_filename},
                        "annotations": [], # 파손/차량 없음
                        "categories": {"id": "non_car", "supercategory_name": "NonVehicle"}
                    }
                    
                    dst_json_path = os.path.join(lbl_out_dir, new_json_name)
                    with open(dst_json_path, 'w', encoding='utf-8') as jf:
                        json.dump(noncar_json_data, jf, ensure_ascii=False, indent=4)
                    
                    crops_found += 1
                    success_crops += 1

        except Exception as e:
            continue
            
    pbar.close()

    print("\n" + "="*50)
    print("🎉 주차장 배경 안전 크롭(Safe Cropping) 완료!")
    print(f"✅ 추출해낸 고해상도(600x600) 배경 이미지: 총 {success_crops} 장")
    print(f"📁 저장 위치: {OUTPUT_DIR}/ (images, labels)")
    print("="*50)
    print("💡 이제 화질 저하 없이 AI가 '배경' 그 자체의 질감을 학습할 수 있습니다.")

if __name__ == "__main__":
    auto_crop_backgrounds()