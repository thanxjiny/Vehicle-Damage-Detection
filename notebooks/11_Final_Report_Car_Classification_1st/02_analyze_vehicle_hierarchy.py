import os
import csv
from collections import defaultdict

# ==============================================================================
# ⚙️ [설정] 사용자 로컬 경로 설정
# ==============================================================================

BASE_DIR = r"D:\AIHUB\091.차량 외관 영상 데이터\01.데이터\2.Validation"
OUTPUT_CSV = r"D:\AIHUB\091.차량 외관 영상 데이터\validation_structure_report_merged.csv"

# ==============================================================================

def analyze_and_merge_structure():
    print(f"🔍 [{BASE_DIR}] 경로의 정밀 분석 및 병합을 시작합니다...")
    
    # 세그먼트(제조사, 차종, 세부정보)별로 데이터를 합산할 딕셔너리
    agg_data = defaultdict(lambda: {
        "이미지_수": 0,
        "영상_수": 0,
        "라벨_수": 0,
        "기타_수": 0,
        "기타_확장자": set()
    })
    
    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.heic', '.webp'}
    label_exts = {'.json', '.xml', '.txt'}
    video_exts = {'.mp4', '.avi', '.mov', '.mkv'}
    
    for root, dirs, files in os.walk(BASE_DIR):
        if len(files) == 0:
            continue
            
        # 1. 하이라키(계층) 구조 추출
        rel_path = os.path.relpath(root, BASE_DIR)
        parts = rel_path.split(os.sep)
        
        # 정상적인 구조 (예: ['원천데이터', 'AU_아우디', '006_A4', '2017_검정_트림A'])
        if len(parts) >= 4:
            brand = parts[1]
            model = parts[2]
            detail = parts[3]
            
            # 병합을 위한 고유 키 생성 (데이터구분 제외)
            segment_key = (brand, model, detail)
            
            # 2. 파일 스캔 및 병합 카운팅
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                
                if ext in img_exts:
                    agg_data[segment_key]["이미지_수"] += 1
                elif ext in label_exts:
                    agg_data[segment_key]["라벨_수"] += 1
                elif ext in video_exts:
                    agg_data[segment_key]["영상_수"] += 1
                else:
                    agg_data[segment_key]["기타_수"] += 1
                    agg_data[segment_key]["기타_확장자"].add(ext if ext else "확장자없음")
        else:
            # 예상치 못한 상위 경로에 파일이 있는 경우 예외 처리
            for f in files:
                agg_data[("기타경로", "기타경로", rel_path)]["기타_수"] += 1

    # 3. 딕셔너리 데이터를 CSV 작성용 리스트로 변환 및 정렬
    data_rows = []
    for (brand, model, detail), counts in agg_data.items():
        data_rows.append({
            "제조사": brand,
            "차종": model,
            "세부정보": detail,
            "이미지_수": counts["이미지_수"],
            "영상(MP4등)_수": counts["영상_수"],
            "라벨(JSON)_수": counts["라벨_수"],
            "기타파일_수": counts["기타_수"],
            "발견된_기타_확장자": ", ".join(counts["기타_확장자"])
        })
        
    # 제조사 -> 차종 -> 세부정보 순으로 깔끔하게 정렬
    data_rows.sort(key=lambda x: (x["제조사"], x["차종"], x["세부정보"]))

    # 4. CSV 파일로 출력
    if data_rows:
        with open(OUTPUT_CSV, mode='w', newline='', encoding='utf-8-sig') as f:
            fieldnames = [
                "제조사", "차종", "세부정보", 
                "이미지_수", "영상(MP4등)_수", "라벨(JSON)_수", "기타파일_수", "발견된_기타_확장자"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_rows)
            
        print("\n" + "="*50)
        print("🎉 정밀 분석 및 데이터 병합 완료!")
        print(f"📁 총 {len(data_rows)}개의 차량 세그먼트를 파악했습니다.")
        print(f"📊 리포트 저장 위치: {OUTPUT_CSV}")
        print("="*50)
    else:
        print("\n❌ 분석할 파일이 없습니다. 경로를 다시 확인해주세요.")

if __name__ == "__main__":
    analyze_and_merge_structure()