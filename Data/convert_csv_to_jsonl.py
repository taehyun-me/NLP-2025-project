import pandas as pd
import json
import random
import re

# ----------------------------------------------------------------
# 1. 파일 경로 및 설정
# ----------------------------------------------------------------
INPUT_CSV = "lyrics_by_year_1964_2023.csv"
OUTPUT_JSONL = "lyrics_rm_data.jsonl"

# 프롬프트 템플릿 (랜덤 선택)
GENERIC_PROMPTS = [
    "한국어 노래 가사를 작사해줘.",
    "감성적인 노래 가사를 써줘.",
    "노랫말을 만들어봐.",
    "작사를 부탁해.",
]

# 접속사 리스트 (가사를 설명문처럼 망가뜨리기 위해 사용)
CONNECTORS = [" 그리고 ", " 그래서 ", " 또한 ", " 즉 ", " 왜냐하면 ", " 이윽고 ", " 하지만 "]

# ----------------------------------------------------------------
# 2. 헬퍼 함수 정의
# ----------------------------------------------------------------

def create_prompt(row):
    """
    메타데이터(제목, 가수)를 활용해 다양한 프롬프트를 생성합니다.
    """
    title = row['title']
    
    templates = [
        f"'{title}'라는 제목으로 노래 가사를 써줘.",
        f"제목이 '{title}'인 노래의 노랫말을 지어줘.",
        random.choice(GENERIC_PROMPTS) # 일반 프롬프트도 섞음
    ]
    return random.choice(templates)

def ruin_lyrics(lyrics):
    """
    Chosen(원본) 가사를 망가뜨려 Rejected(오답) 데이터를 만듭니다.
    줄바꿈을 없애고, 문장 사이에 접속사를 넣어 '줄글'처럼 만듭니다.
    """
    if not isinstance(lyrics, str):
        return ""
        
    # 기본 정제 (양쪽 공백 제거)
    lyrics = lyrics.strip()
    
    # 줄바꿈 기준으로 분리
    lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
    
    # 너무 짧으면 그냥 공백으로 연결
    if len(lines) < 3:
        return " ".join(lines)
    
    ruined_text = ""
    for i, line in enumerate(lines):
        ruined_text += line
        
        # 마지막 줄이 아니면 접속사나 공백 추가
        if i < len(lines) - 1:
            # 40% 확률로 접속사 투입, 나머지는 그냥 공백
            if random.random() < 0.4:
                ruined_text += random.choice(CONNECTORS)
            else:
                ruined_text += " "
    
    return ruined_text

# ----------------------------------------------------------------
# 3. 메인 로직
# ----------------------------------------------------------------

def main():
    print(f"📂 '{INPUT_CSV}' 로딩 중...")
    try:
        # CSV 읽기 (인코딩 에러 발생 시 'cp949'나 'euc-kr'로 변경 시도 필요할 수 있음)
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        print(f"CSV 로드 실패: {e}")
        return

    print(f"총 {len(df)}개의 데이터가 있습니다.")

    # 1. 데이터 전처리 (결측치 제거)
    df = df.dropna(subset=['lyric', 'title'])
    
    # 2. 너무 짧은 가사 제거 (예: 연주곡 등) - 50자 미만 제거
    df = df[df['lyric'].str.len() > 50]
    
    # (선택사항) 19금 가사 제외 여부
    # df = df[df['x_rated'] == False] 

    print(f"전처리 후 유효한 데이터: {len(df)}개")
    
    # 학습 시간을 고려하여 프로토타입용으로 일부만 샘플링 (예: 2,000개)
    # 전체를 다 쓰고 싶으면 아래 두 줄 주석 처리
    if len(df) > 2000:
        print("⚡ 빠른 학습을 위해 2,000개만 랜덤 샘플링합니다.")
        df = df.sample(n=2000, random_state=42)

    converted_data = []

    print("🔄 데이터 변환 중...")
    for _, row in df.iterrows():
        chosen_text = row['lyric'].strip()
        rejected_text = ruin_lyrics(chosen_text)
        
        # 오답 생성에 실패했거나 너무 짧아진 경우 스킵
        if len(rejected_text) < 10:
            continue
            
        entry = {
            "prompt": create_prompt(row),
            "chosen": chosen_text,
            "rejected": rejected_text
        }
        converted_data.append(entry)

    # JSONL 저장
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for entry in converted_data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print(f"\n✅ 변환 완료! '{OUTPUT_JSONL}' 파일이 생성되었습니다.")
    print(f"총 데이터 개수: {len(converted_data)}")
    print("\n[데이터 예시]")
    print(json.dumps(converted_data[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()