import os
import glob
import shutil
import json
from tqdm import tqdm
from PIL import Image

# ==============================================================================
# ⚙️ [설정] 사용자 로컬 경로 설정
# ==============================================================================

# 1. 캐글 정상 차량 이미지가 있는 원본 폴더
INPUT_DIR = r"D:\AIHUB\kaggle_normal_car\images"

# 2. 최종 데이터셋이 모이는 SAMPLE 폴더
OUTPUT_DIR = r"D:\AIHUB\kaggle_normal_car\SAMPLE_kaggle_normal_car"

# 3. 파일명 앞에 붙일 접두사 (Prefix)
PREFIX = "kaggle_normal_car_"

# ==============================================================================

def process_kaggle_images():
    print(f"🔍 [{INPUT_DIR}] 경로에서 이미지 스캔 중...")
    
    # 1. 출력 폴더 세팅
    img_out_dir = os.path.join(OUTPUT_DIR, 'images')
    lbl_out_dir = os.path.join(OUTPUT_DIR, 'labels')
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)
    
    # 2. 유효한 이미지 파일 수집
    search_pattern = os.path.join(INPUT_DIR, "*.*")
    all_files = glob.glob(search_pattern)
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in all_files if f.lower().endswith(valid_exts)]
    
    total_images = len(image_files)
    print(f"👉 총 {total_images}장의 캐글 이미지를 찾았습니다.")
    
    if total_images == 0:
        print("❌ 처리할 이미지가 없습니다. 경로를 다시 확인해주세요.")
        return

    print("🚀 파일명 변경, 복사 및 JSON 라벨 자동 생성을 시작합니다...")
    
    success_count = 0
    pbar = tqdm(total=total_images)
    
    # 3. 일괄 처리 루프
    for img_path in image_files:
        try:
            # 원본 파일명 추출 (예: "01.jpg")
            orig_filename = os.path.basename(img_path)
            orig_basename, orig_ext = os.path.splitext(orig_filename)
            
            # 새 파일명 생성 (예: "kaggle_normal_car_01.jpg")
            new_filename = f"{PREFIX}{orig_basename}{orig_ext}"
            new_json_name = f"{PREFIX}{orig_basename}.json"
            
            dst_img_path = os.path.join(img_out_dir, new_filename)
            dst_json_path = os.path.join(lbl_out_dir, new_json_name)
            
            # [Step A] 이미지 파일 복사 (새 이름으로)
            if not os.path.exists(dst_img_path):
                shutil.copy2(img_path, dst_img_path)
                
            # [Step B] 이미지 크기(width, height) 추출
            with Image.open(img_path) as img:
                w, h = img.size
                
            # [Step C] 정상 차량용 JSON 데이터 구조체 생성 (파손 빈 리스트)
            normal_json_data = {
                "info": {
                    "name": "kaggle_normal_vehicle", 
                    "date_created": "auto_generated"
                },
                "images": {
                    "id": 1, 
                    "width": w, 
                    "height": h, 
                    "file_name": new_filename  # 변경된 파일명 기입
                },
                "annotations": [], # 파손 부위 없음
                "categories": {
                    "id": "normal", 
                    "supercategory_name": "Vehicle"
                }
            }
            
            # [Step D] JSON 파일 저장
            with open(dst_json_path, 'w', encoding='utf-8') as jf:
                json.dump(normal_json_data, jf, ensure_ascii=False, indent=4)
                
            success_count += 1
            pbar.update(1)
            
        except Exception as e:
            # 예기치 못한 에러(손상된 이미지 등) 발생 시 스킵
            pbar.update(1)
            continue
            
    pbar.close()

    print("\n" + "="*50)
    print("🎉 Kaggle 정상 차량 데이터 통합 완료!")
    print(f"✅ 성공적으로 처리된 수량: {success_count} / {total_images} 장")
    print(f"📁 저장 위치: {OUTPUT_DIR}/ (images, labels)")
    print("="*50)
    print("💡 이제 파일명 충돌 걱정 없이 안전하게 SAMPLE 데이터셋에 추가되었습니다.")

if __name__ == "__main__":
    process_kaggle_images()