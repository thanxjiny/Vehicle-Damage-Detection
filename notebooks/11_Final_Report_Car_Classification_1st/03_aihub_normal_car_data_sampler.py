import os
import csv
import json
import shutil
import glob
import random
from tqdm import tqdm
from PIL import Image

# ==============================================================================
# ⚙️ [설정] 사용자 로컬 경로 설정
# ==============================================================================

# 1. 파일 및 폴더 경로 설정
CSV_PATH = r"D:\AIHUB\091.차량 외관 영상 데이터\validation_structure_report_merged.csv"
SRC_BASE_DIR = r"D:\AIHUB\091.차량 외관 영상 데이터\01.데이터\2.Validation\원천데이터"
OUTPUT_DIR = r"D:\AIHUB\091.차량 외관 영상 데이터\SAMPLE_aihub_normal_car"

# 2. 목표 샘플링 수량
TARGET_TOTAL = 12000

# ==============================================================================

def stratified_sample_normal_vehicles():
    print("🔍 1단계: CSV 데이터를 분석하여 층화 추출 할당량(Quota)을 계산합니다...")
    
    segments = []
    total_available_images = 0
    
    # 1. 데이터 파악
    with open(CSV_PATH, mode='r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_count = int(row["이미지_수"])
            if img_count > 0:
                folder_path = os.path.join(SRC_BASE_DIR, row["제조사"], row["차종"], row["세부정보"])
                segments.append({
                    "path": folder_path,
                    "available": img_count,
                    "quota": 0,
                    "name": f"{row['제조사']}_{row['차종']}"
                })
                total_available_images += img_count

    print(f"👉 전체 정상 이미지 풀(Pool): {total_available_images}장")
    
    if total_available_images == 0:
        print("❌ 추출할 이미지가 없습니다. 경로를 확인해주세요.")
        return

    # 2. 비례 할당(Proportional Allocation) 로직
    # 만약 원본이 12000장 이하라면 전부 다 가져옴
    actual_target = min(TARGET_TOTAL, total_available_images)
    
    current_quota_sum = 0
    for seg in segments:
        # 비율에 따른 할당량 계산 (반올림)
        proportion = seg["available"] / total_available_images
        quota = int(round(proportion * actual_target))
        
        # 할당량이 원본 보유량보다 많아지는 경우 방지
        quota = min(quota, seg["available"])
        
        seg["quota"] = quota
        current_quota_sum += quota

    # 3. 반올림 오차 보정 (정확히 12,000장 맞추기)
    difference = actual_target - current_quota_sum
    
    # 부족하면 여유가 있는 그룹에 1장씩 더해주고, 넘치면 1장씩 뺌
    if difference != 0:
        # 가용량에 여유가 많은 순서대로 정렬해서 보정
        segments.sort(key=lambda x: x["available"] - x["quota"], reverse=(difference > 0))
        
        for i in range(abs(difference)):
            if difference > 0:
                # 부족한 경우 (여유 있는 곳에 1장씩 추가)
                segments[i % len(segments)]["quota"] += 1
            else:
                # 넘치는 경우 (1장씩 회수)
                segments[i % len(segments)]["quota"] -= 1

    print(f"🎯 층화 추출 할당 완료! (총 {sum(s['quota'] for s in segments)}장 추출 예정)")

    # 4. 실제 이미지 샘플링 및 폴더 복사
    print("\n🔍 2단계: 할당된 수량에 맞춰 폴더별로 이미지를 추출하고 JSON을 생성합니다...")
    
    img_out_dir = os.path.join(OUTPUT_DIR, 'images')
    lbl_out_dir = os.path.join(OUTPUT_DIR, 'labels')
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    success_count = 0
    pbar = tqdm(total=actual_target)

    for seg in segments:
        quota = seg["quota"]
        if quota <= 0:
            continue
            
        if not os.path.exists(seg["path"]):
            pbar.update(quota) # 폴더가 없으면 게이지라도 올리고 스킵
            continue
            
        # 해당 세그먼트의 전체 이미지 스캔
        search_pattern = os.path.join(seg["path"], "*.*")
        all_imgs = [f for f in glob.glob(search_pattern) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not all_imgs:
            pbar.update(quota)
            continue
            
        # 무작위 샘플링 (할당량만큼)
        random.shuffle(all_imgs)
        sampled_imgs = all_imgs[:quota]
        
        # 파일 복사 및 라벨 생성
        for img_path in sampled_imgs:
            try:
                file_name = os.path.basename(img_path)
                base_name = os.path.splitext(file_name)[0]
                
                # 이미지 복사
                dst_img_path = os.path.join(img_out_dir, file_name)
                if not os.path.exists(dst_img_path):
                    shutil.copy2(img_path, dst_img_path)
                
                # 사이즈 확인 및 JSON 생성
                with Image.open(img_path) as img:
                    w, h = img.size
                    
                normal_json_data = {
                    "info": {
                        "name": "normal_vehicle", 
                        "date_created": "auto_generated"
                    },
                    "images": {
                        "id": 1, 
                        "width": w, 
                        "height": h, 
                        "file_name": file_name
                    },
                    "annotations": [], # 파손 없음 (빈 리스트)
                    "categories": {
                        "id": "normal", 
                        "supercategory_name": "Vehicle"
                    }
                }
                
                # JSON 저장
                dst_json_path = os.path.join(lbl_out_dir, base_name + ".json")
                with open(dst_json_path, 'w', encoding='utf-8') as jf:
                    json.dump(normal_json_data, jf, ensure_ascii=False, indent=4)
                    
                success_count += 1
                pbar.update(1)
                
            except Exception as e:
                pbar.update(1)
                continue

    pbar.close()

    print("\n" + "="*50)
    print("🎉 층화 추출(Stratified Sampling) 기반 데이터 수집 완료!")
    print(f"✅ 성공적으로 복사 및 생성된 수량: {success_count} / {actual_target} 장")
    print(f"📁 저장 위치: {OUTPUT_DIR}/ (images, labels)")
    print("="*50)

if __name__ == "__main__":
    stratified_sample_normal_vehicles()