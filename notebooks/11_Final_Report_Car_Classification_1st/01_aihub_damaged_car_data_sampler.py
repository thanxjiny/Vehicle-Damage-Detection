import os
import json
import shutil
import glob
import random
from tqdm import tqdm
from pathlib import Path

# ==============================================================================
# ⚙️ [설정] 사용자 로컬 경로 설정
# ==============================================================================

# 1. AI-Hub 원본 데이터 경로 (Validation 경로만 남김)
PATHS = {
    "valid_images": r"D:\AIHUB\160. 차량파손 이미지 데이터\01.데이터\2.Validation\1.원천데이터\VS_damage\damage",
    "valid_labels": r"D:\AIHUB\160. 차량파손 이미지 데이터\01.데이터\2.Validation\2.라벨링데이터\VL_damage\damage"
}

# 2. 결과 데이터셋이 저장될 경로 (통합 저장)
OUTPUT_DIR = r"D:\AIHUB\160. 차량파손 이미지 데이터\SAMPLE_aihub_damaged_car_data"

# 3. 목표 수량 설정 (클래스별 밸런싱)
# 3,000장 x 4개 클래스 = 총 12,000장 수집 (단, Valid 총량이 부족하면 있는 만큼만 수집됨)
TARGET_COUNT_PER_CLASS = 3000 

# 4. 클래스 매핑 (카운팅 목적)
# 원본 JSON에서 "damage" 필드 값을 확인하여 밸런싱을 맞추기 위한 맵핑
CLASS_MAP = {
    "Scratched": 0,
    "Separated": 1,
    "Breakage": 2,
    "Crushed": 3,
    "Dent": 3,      # "Dent" 표기 예외 처리
    "Dented": 3     # "Dented" 표기 예외 처리
}

# 역매핑 (최종 출력 확인용)
ID_TO_TEXT = {v: k for k, v in CLASS_MAP.items() if k not in ["Dent", "Dented"]}

# ==============================================================================

def collect_json_files():
    """원본 Valid 라벨 폴더에서만 모든 JSON 파일 수집 및 풀링"""
    print("🔍 Validation 데이터 스캔 중...")
    
    file_list = []
    
    # 1. Validation Labels만 수집 (Train 부분 삭제)
    v_labels = glob.glob(os.path.join(PATHS["valid_labels"], "*.json"))
    for json_path in v_labels:
        file_list.append({"path": json_path, "img_root": PATHS["valid_images"]})
        
    # 데이터 섞기 (편향 방지를 위한 무작위 추출)
    random.shuffle(file_list)
    print(f"👉 총 {len(file_list)}개의 라벨 파일을 찾았습니다.")
    return file_list


def create_dataset():
    # 1. 출력 폴더 생성 (images, labels)
    if os.path.exists(OUTPUT_DIR):
        print(f"⚠️ 기존 폴더가 존재합니다: {OUTPUT_DIR}")
    
    os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'labels'), exist_ok=True)

    json_files = collect_json_files()
    
    # 2. 카운터 초기화
    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    
    # 진행률 표시 바
    pbar = tqdm(total=len(json_files))
    
    for item in json_files:
        pbar.update(1)
        json_path = item["path"]
        img_root_dir = item["img_root"]
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # --- JSON 파싱 및 타겟 클래스 확인 ---
            
            # 1. 이미지 정보 확인
            img_info = data.get('images', {})
            if isinstance(img_info, list):
                img_info = img_info[0]
                
            file_name = img_info.get('file_name')
            if not file_name:
                continue # 파일명 정보가 없으면 스킵

            # 2. 이미지 파일 존재 여부 확인
            src_img_path = os.path.join(img_root_dir, file_name)
            if not os.path.exists(src_img_path):
                # 파일명이 다를 경우 대비 (확장자 등)
                base = os.path.splitext(file_name)[0]
                candidates = glob.glob(os.path.join(img_root_dir, base + ".*"))
                if candidates:
                    src_img_path = candidates[0]
                    file_name = os.path.basename(src_img_path)
                else:
                    continue # 실제 이미지가 없으면 스킵

            # 3. 어노테이션 확인 및 대표 클래스 추출
            anns = data.get('annotations', [])
            target_cls = -1 
            
            for ann in anns:
                damage_type = ann.get('damage', '')
                bbox = ann.get('bbox')
                
                # 유효한 파손 타입이고, bbox 정보가 존재하는 경우
                if damage_type in CLASS_MAP and bbox:
                    cls_id = CLASS_MAP[damage_type]
                    
                    # 카운팅 로직: 이 이미지의 대표 파손 타입 설정
                    if target_cls == -1:
                        target_cls = cls_id
                        break # 첫 번째 유효한 파손 타입을 기준으로 분류 처리
            
            # 유효한 라벨이 없거나, 대상 클래스가 없으면 스킵
            if target_cls == -1:
                continue 

            # 4. 수집 목표 달성 여부 체크
            if counts[target_cls] >= TARGET_COUNT_PER_CLASS:
                continue

            # 5. 파일 복사 (이미지 및 원본 JSON 유지)
            dst_img_path = os.path.join(OUTPUT_DIR, 'images', file_name)
            shutil.copy2(src_img_path, dst_img_path)
            
            json_file_name = os.path.basename(json_path)
            dst_json_path = os.path.join(OUTPUT_DIR, 'labels', json_file_name)
            shutil.copy2(json_path, dst_json_path)
                
            # 카운트 증가
            counts[target_cls] += 1
            
            # 조기 종료 체크 (목표 달성 시)
            if sum(counts.values()) >= (TARGET_COUNT_PER_CLASS * 4):
                print("\n🎯 모든 클래스의 목표 수량을 달성하여 탐색을 조기 종료합니다.")
                break

        except Exception as e:
            continue
            
    pbar.close()
    
    # 6. 최종 결과 요약 출력
    print("\n" + "="*50)
    print("🎉 데이터셋 (Validation 전용) 샘플링 완료!")
    print(f"📁 저장 위치: {OUTPUT_DIR}/ (images, labels)")
    print("="*50)
    print("📊 최종 수집 결과 (목표: 각 3,000장, 총 12,000장):")
    total_collected = 0
    for cls_id, count in counts.items():
        cls_name = ID_TO_TEXT.get(cls_id, f"Class {cls_id}")
        print(f"  - {cls_name}: {count}장")
        total_collected += count
    print(f"  ▶ 총합: {total_collected}장")
    
    # Validation 총량이 목표치에 미달했을 경우의 안내 메시지
    if total_collected < (TARGET_COUNT_PER_CLASS * 4):
        print("\n💡 참고: Validation 폴더 내의 데이터 수량이 부족하여 목표치(12,000장)를 모두 채우지 못했습니다.")
        print("💡 학습을 위해 더 많은 데이터가 필요하다면 나중에 Train 압축을 풀고 추가하셔야 합니다.")

if __name__ == "__main__":
    create_dataset()