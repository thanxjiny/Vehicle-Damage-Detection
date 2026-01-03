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

# 1. AI-Hub 원본 데이터 경로 (사용자가 제공한 경로)
# (경로에 한글이 없어서 문제 없으나, 역슬래시 \ 대신 / 또는 r"" 사용 권장)
PATHS = {
    "train_images": r"C:\Users\strin\Downloads\AIHUB\DATA\1_Training\1_IMAGES\TS_damage\train",
    "train_labels": r"C:\Users\strin\Downloads\AIHUB\DATA\1_Training\2_LABLES\TL_damage\damage",
    "valid_images": r"C:\Users\strin\Downloads\AIHUB\DATA\2_Validation\1_IMAGES\VS_damage\valid",
    "valid_labels": r"C:\Users\strin\Downloads\AIHUB\DATA\2_Validation\2_LABLES\VL_damage\damage"
}

# 2. 결과 데이터셋이 저장될 경로 (이 폴더를 나중에 압축해서 구글 드라이브에 올림)
OUTPUT_DIR = r"C:\Users\strin\Downloads\AI_HUB_DAMAGE_DATASET"

# 3. 목표 수량 설정 (클래스별 밸런싱)
# Scratched, Separated, Breakage, Crushed 각각 아래 수량만큼 뽑음
TARGET_COUNTS = {
    "train": 2500,  # 클래스당 2500장 x 4 = 10,000장
    "val": 500      # 클래스당 500장 x 4 = 2,000장
}

# 4. 클래스 매핑 (Text -> ID)
# JSON의 "damage" 필드 값과 매칭
CLASS_MAP = {
    "Scratched": 0,
    "Separated": 1,
    "Breakage": 2,
    "Crushed": 3,
    "Dent": 3,      # 혹시 "Dent"로 표기된 데이터가 있다면 Crushed로 통합
    "Dented": 3     # 예외 처리
}

# 역매핑 (출력 확인용)
ID_TO_TEXT = {v: k for k, v in CLASS_MAP.items() if k not in ["Dent", "Dented"]}

# ==============================================================================

def convert_bbox(box, img_w, img_h):
    """[x, y, w, h] -> YOLO [cx, cy, w, h] 정규화"""
    x, y, w, h = box
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh

def collect_json_files():
    """Train/Val 라벨 폴더에서 모든 JSON 파일 수집"""
    print("🔍 데이터 스캔 중...")
    
    # 소스별로 리스트 생성
    file_list = []
    
    # 1. Training Labels
    t_labels = glob.glob(os.path.join(PATHS["train_labels"], "*.json"))
    for json_path in t_labels:
        file_list.append({"path": json_path, "type": "train", "img_root": PATHS["train_images"]})
        
    # 2. Validation Labels
    v_labels = glob.glob(os.path.join(PATHS["valid_labels"], "*.json"))
    for json_path in v_labels:
        file_list.append({"path": json_path, "type": "valid", "img_root": PATHS["valid_images"]})
        
    # 전체 섞기 (Train 폴더에 있는 것도 Validation용으로 쓸 수 있고 그 반대도 가능하도록 풀링)
    # 하지만 원본 데이터의 Train/Val 구분을 존중하려면 위 로직을 따르되, 
    # 여기서는 '균등 추출'이 핵심이므로 전체를 섞어서 재분배하는 것이 밸런싱에 더 유리합니다.
    random.shuffle(file_list)
    print(f"👉 총 {len(file_list)}개의 라벨 파일을 찾았습니다.")
    return file_list

def create_dataset():
    # 출력 폴더 초기화
    if os.path.exists(OUTPUT_DIR):
        print(f"⚠️ 기존 폴더가 존재합니다: {OUTPUT_DIR}")
    
    for split in ['train', 'val']:
        os.makedirs(os.path.join(OUTPUT_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, 'labels', split), exist_ok=True)

    json_files = collect_json_files()
    
    # 카운터 초기화
    counts = {
        "train": {0: 0, 1: 0, 2: 0, 3: 0},
        "val": {0: 0, 1: 0, 2: 0, 3: 0}
    }
    
    # 진행률 표시 바
    pbar = tqdm(total=len(json_files))
    
    for item in json_files:
        pbar.update(1)
        json_path = item["path"]
        img_root_dir = item["img_root"]
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # --- JSON 파싱 (제공해주신 샘플 기준) ---
            
            # 1. 이미지 정보 확인
            # 샘플: "images": {"id": 1, "width": 800, ...} (Dict 형태)
            img_info = data.get('images', {})
            if isinstance(img_info, list): # 혹시 리스트인 경우 대비
                img_info = img_info[0]
                
            file_name = img_info.get('file_name')
            img_w = img_info.get('width')
            img_h = img_info.get('height')
            
            if not file_name or not img_w or not img_h:
                continue # 정보 부족 시 스킵

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
                    continue # 이미지 없으면 스킵

            # 3. 어노테이션 파싱
            anns = data.get('annotations', [])
            yolo_labels = []
            
            # 이 이미지의 대표 클래스 (카운팅용) - 가장 많이 등장한 파손이나 첫 번째 파손
            target_cls = -1 
            
            for ann in anns:
                damage_type = ann.get('damage', '')
                bbox = ann.get('bbox') # [x, y, w, h]
                
                if damage_type in CLASS_MAP and bbox:
                    cls_id = CLASS_MAP[damage_type]
                    cx, cy, w, h = convert_bbox(bbox, img_w, img_h)
                    
                    # YOLO 라벨 포맷: class x y w h
                    yolo_labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    
                    # 카운팅 로직: 현재 부족한 클래스 우선 채움
                    if target_cls == -1:
                        target_cls = cls_id
            
            if not yolo_labels or target_cls == -1:
                continue # 유효한 라벨이 없으면 스킵

            # 4. Train/Val 배분 로직
            # 현재 이 이미지가 가진 클래스가 Train 목표를 못 채웠으면 Train으로,
            # Train은 찼는데 Val이 비었으면 Val로.
            split = None
            
            if counts['train'][target_cls] < TARGET_COUNTS['train']:
                split = 'train'
            elif counts['val'][target_cls] < TARGET_COUNTS['val']:
                split = 'val'
            
            if split is None:
                continue # 이미 목표 수량을 채운 클래스면 패스

            # 5. 파일 복사 및 라벨 저장
            # 이미지 복사
            dst_img_path = os.path.join(OUTPUT_DIR, 'images', split, file_name)
            shutil.copy2(src_img_path, dst_img_path)
            
            # 라벨 저장
            txt_name = os.path.splitext(file_name)[0] + ".txt"
            dst_txt_path = os.path.join(OUTPUT_DIR, 'labels', split, txt_name)
            with open(dst_txt_path, 'w', encoding='utf-8') as f_out:
                f_out.write("\n".join(yolo_labels))
                
            counts[split][target_cls] += 1
            
            # 조기 종료 체크 (모든 클래스 목표 달성 시)
            total_goal = (TARGET_COUNTS['train'] + TARGET_COUNTS['val']) * 4
            current_total = sum(sum(c.values()) for c in counts.values())
            
            if current_total >= total_goal:
                # 하지만 정확한 클래스별 밸런싱을 위해 루프를 바로 끊기보다, 
                # 위 split is None 조건에 의해 자연스럽게 필터링되도록 둠
                pass

        except Exception as e:
            # print(f"에러 발생 {json_path}: {e}")
            continue
            
    pbar.close()
    
    print("\n" + "="*50)
    print("🎉 데이터셋 생성 완료!")
    print(f"📁 저장 위치: {OUTPUT_DIR}")
    print("="*50)
    print("📊 최종 수집 결과:")
    for split in ['train', 'val']:
        print(f"\n[{split.upper()}]")
        for cls_id, count in counts[split].items():
            cls_name = ID_TO_TEXT.get(cls_id, f"Class {cls_id}")
            print(f"  - {cls_name}: {count}장")

if __name__ == "__main__":
    create_dataset()