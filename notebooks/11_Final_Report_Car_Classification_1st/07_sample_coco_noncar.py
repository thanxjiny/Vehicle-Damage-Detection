import os
import json
import shutil
from PIL import Image
from tqdm import tqdm

# ==============================================================================
# 🌟 [매우 중요] FiftyOne 다운로드 경로 D드라이브 강제 할당
# 반드시 import fiftyone 이전에 os.environ으로 설정해야 합니다!
# ==============================================================================
D_DRIVE_FIFTYONE_PATH = r"D:\AIHUB\fiftyone_datasets"
os.environ["FIFTYONE_DATASET_ZOO_DIR"] = os.path.join(D_DRIVE_FIFTYONE_PATH, "zoo")
os.environ["FIFTYONE_DEFAULT_DATASET_DIR"] = os.path.join(D_DRIVE_FIFTYONE_PATH, "default")

import fiftyone as fo
import fiftyone.zoo as foz

# ==============================================================================
# ⚙️ [설정] 사용자 로컬 경로 설정
# ==============================================================================

# 1. 최종 데이터셋이 모이는 SAMPLE 폴더 경로
OUTPUT_DIR = r"D:\AIHUB\coco2017_nocar\SAMPLE_coco2017_nocar"

# 2. 추출할 10개의 명확한 Non-Car 객체 클래스 선정
TARGET_CLASSES = [
    "person", "dog", "cat", "bench", "umbrella", 
    "traffic light", "stop sign", "backpack", "suitcase", "potted plant"
]

# 3. 클래스당 목표 추출 수량
SAMPLES_PER_CLASS = 100

# 4. 절대로 사진에 포함되어서는 안 되는 '금지 클래스' (차량 관련)
BANNED_CLASSES = {"car", "bus", "truck", "motorcycle"}

# ==============================================================================

def extract_coco_noncar_images():
    print("🚀 COCO 2017 데이터셋에서 완벽한 Non-Car 이미지 추출을 시작합니다.")
    # 에러가 발생했던 fo.config 출력 부분을 안전하게 직접 출력으로 변경했습니다.
    print(f"💾 데이터 다운로드 경로가 D 드라이브로 설정되었습니다: {D_DRIVE_FIFTYONE_PATH}")
    
    img_out_dir = os.path.join(OUTPUT_DIR, 'images')
    lbl_out_dir = os.path.join(OUTPUT_DIR, 'labels')
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)
    
    total_saved = 0
    
    for cls in TARGET_CLASSES:
        print(f"\n🔍 '{cls}' 클래스 수집 및 차량 필터링 중...")
        
        try:
            # 1. 특정 클래스가 포함된 데이터 로드 
            dataset = foz.load_zoo_dataset(
                "coco-2017",
                split="train",
                classes=[cls],
                max_samples=SAMPLES_PER_CLASS * 3,
                shuffle=True
            )
            
            saved_for_cls = 0
            
            # 2. 다운로드된 샘플들을 하나씩 검사
            for sample in dataset:
                if saved_for_cls >= SAMPLES_PER_CLASS:
                    break 
                
                has_vehicle = False
                
                # 이미지 내에 금지된 차량 객체가 있는지 검사
                if sample.ground_truth:
                    for det in sample.ground_truth.detections:
                        if det.label in BANNED_CLASSES:
                            has_vehicle = True
                            break
                
                # 차량이 전혀 없는 깨끗한 사진인 경우만 저장
                if not has_vehicle:
                    orig_filepath = sample.filepath
                    ext = os.path.splitext(orig_filepath)[1]
                    
                    new_filename = f"coco_{cls.replace(' ', '_')}_{sample.id}{ext}"
                    dst_img_path = os.path.join(img_out_dir, new_filename)
                    
                    # [Step A] 이미지 복사
                    if not os.path.exists(dst_img_path):
                        shutil.copy2(orig_filepath, dst_img_path)
                    
                    # [Step B] 사이즈 확인
                    with Image.open(dst_img_path) as img:
                        w, h = img.size
                    
                    # [Step C] Non-Car 용 JSON 생성 (파손 없음)
                    base_name = os.path.splitext(new_filename)[0]
                    dst_json_path = os.path.join(lbl_out_dir, base_name + ".json")
                    
                    noncar_json_data = {
                        "info": {"name": f"coco_{cls}", "date_created": "auto_generated"},
                        "images": {"id": 1, "width": w, "height": h, "file_name": new_filename},
                        "annotations": [], 
                        "categories": {"id": "non_car", "supercategory_name": "NonVehicle"}
                    }
                    
                    # [Step D] JSON 저장
                    with open(dst_json_path, 'w', encoding='utf-8') as jf:
                        json.dump(noncar_json_data, jf, ensure_ascii=False, indent=4)
                    
                    saved_for_cls += 1
                    total_saved += 1
            
            print(f"   ✅ '{cls}' 클래스 {saved_for_cls}장 추출 완료.")
            
            # 메모리 최적화를 위해 데이터셋 삭제
            dataset.delete()
            
        except Exception as e:
            print(f"   ❌ '{cls}' 처리 중 오류 발생: {e}")
            
    print("\n" + "="*50)
    print("🎉 COCO 2017 Non-Car 객체 데이터 추출 및 통합 완료!")
    print(f"✅ 추출된 총 이미지 수: {total_saved} 장")
    print(f"📁 저장 위치: {OUTPUT_DIR}/ (images, labels)")
    print("="*50)

if __name__ == "__main__":
    extract_coco_noncar_images()