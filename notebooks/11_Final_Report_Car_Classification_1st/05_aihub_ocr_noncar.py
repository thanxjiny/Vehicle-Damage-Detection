import os
import glob
import shutil
import json
from tqdm import tqdm
from PIL import Image

# ==============================================================================
# ⚙️ [설정] 사용자 로컬 경로 설정
# ==============================================================================

# 1. 공공행정문서 OCR 원본 폴더 (Non-Car 데이터)
INPUT_DIR = r"D:\AIHUB\공공행정문서 OCR\New_sample\원천데이터\인.허가\5350109\2001"

# 2. 최종 데이터셋이 모이는 SAMPLE 폴더
OUTPUT_DIR = r"D:\AIHUB\공공행정문서 OCR\SAMPLE_aihub_ocr_nocar"

# ==============================================================================

def process_noncar_ocr_images():
    print(f"🔍 [{INPUT_DIR}] 경로에서 문서 이미지 스캔 중...")
    
    # 1. 출력 폴더 세팅 (기존 SAMPLE 폴더 재사용)
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
    print(f"👉 총 {total_images}장의 OCR 문서 이미지를 찾았습니다.")
    
    if total_images == 0:
        print("❌ 처리할 이미지가 없습니다. 경로를 다시 확인해주세요.")
        return

    print("🚀 파일 복사 및 Non-Car JSON 라벨 자동 생성을 시작합니다...")
    
    success_count = 0
    pbar = tqdm(total=total_images)
    
    # 3. 일괄 처리 루프
    for img_path in image_files:
        try:
            # 원본 파일명 그대로 사용
            file_name = os.path.basename(img_path)
            base_name, _ = os.path.splitext(file_name)
            
            dst_img_path = os.path.join(img_out_dir, file_name)
            dst_json_path = os.path.join(lbl_out_dir, base_name + ".json")
            
            # [Step A] 이미지 파일 복사
            if not os.path.exists(dst_img_path):
                shutil.copy2(img_path, dst_img_path)
                
            # [Step B] 이미지 크기(width, height) 추출
            with Image.open(img_path) as img:
                w, h = img.size
                
            # [Step C] Non-Car 용 JSON 데이터 구조체 생성
            # 차량이 아니므로 category를 'non_car'로 명확히 지정하고 파손 부위는 비워둡니다.
            noncar_json_data = {
                "info": {
                    "name": "public_ocr_document", 
                    "date_created": "auto_generated"
                },
                "images": {
                    "id": 1, 
                    "width": w, 
                    "height": h, 
                    "file_name": file_name
                },
                "annotations": [], # 파손이나 차량 부품이 없으므로 빈 리스트
                "categories": {
                    "id": "non_car", 
                    "supercategory_name": "NonVehicle"
                }
            }
            
            # [Step D] JSON 파일 저장
            with open(dst_json_path, 'w', encoding='utf-8') as jf:
                json.dump(noncar_json_data, jf, ensure_ascii=False, indent=4)
                
            success_count += 1
            
        except Exception as e:
            # 예기치 못한 에러(손상된 이미지 등) 발생 시 스킵
            pass
        finally:
            pbar.update(1)
            
    pbar.close()

    print("\n" + "="*50)
    print("🎉 차량이 아닌 이미지(Non-Car 문서) 통합 완료!")
    print(f"✅ 성공적으로 처리된 수량: {success_count} / {total_images} 장")
    print(f"📁 저장 위치: {OUTPUT_DIR}/ (images, labels)")
    print("="*50)
    print("💡 이제 1단계 '차량 유무 판별 모델'이 똑똑하게 서류를 걸러낼 수 있는 기반이 마련되었습니다.")

if __name__ == "__main__":
    process_noncar_ocr_images()