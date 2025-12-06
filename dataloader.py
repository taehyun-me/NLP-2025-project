import pandas as pd
from datasets import Dataset

# ==========================================
# 1. 데이터 로드 및 전처리
# ==========================================
def load_and_clean_data(file_path):
    # CSV 로드
    df = pd.read_csv(file_path)
    print(f"전체 데이터 수: {len(df)}")

    # 1. 가사가 없는 데이터 제거
    df = df.dropna(subset=['lyric'])
    
    # 2. (선택) 성인용 데이터 제거 (안전한 생성을 위해)
    if 'x_rated' in df.columns:
        df = df[df['x_rated'] == False]

    # 3. 텍스트 정제 (문자열이 아닌 데이터 방지)
    df['lyric'] = df['lyric'].astype(str)
    
    print(f"전처리 후 데이터 수: {len(df)}")
    return df

# ==========================================
# 2. Generator용 Query 데이터셋 생성 (PPO용)
# ==========================================
def create_ppo_dataset(df, min_length=5):
    """
    가사를 줄 단위로 쪼개서, PPO 모델에게 던져줄 '앞 문장(Query)' 리스트를 만듭니다.
    """
    queries = []
    
    for _, row in df.iterrows():
        # 가사를 줄바꿈으로 분리
        lines = row['lyric'].split('\n')
        
        # 각 줄을 순회하며 Query 생성
        # 마지막 줄은 '다음 줄'이 없으므로 제외
        for i in range(len(lines) - 1):
            current_line = lines[i].strip()
            
            # 너무 짧은 문장(예: "Yeah", "우")은 제외하여 학습 퀄리티 높임
            if len(current_line) >= min_length:
                # 모델에게 입력할 프롬프트 형식 정의
                # (Gemma-2 등 Instruct 모델은 명확한 지시문이 있으면 더 잘 작동함)
                prompt = f"다음 가사를 이어 써줘:\n{current_line}"
                queries.append(prompt)

    # 데이터셋 객체로 변환
    dataset = Dataset.from_dict({"query": queries})
    return dataset

# ==========================================
# 3. Discriminator용 Real 데이터셋 생성
# ==========================================
def create_discriminator_dataset(df):
    """
    Discriminator 학습을 위해 '진짜 가사' 문장들을 모읍니다.
    Label: 1 (Real)
    """
    real_lyrics = []
    
    for _, row in df.iterrows():
        lines = row['lyric'].split('\n')
        for line in lines:
            if len(line.strip()) > 5:
                real_lyrics.append(line.strip())
                
    # Discriminator 학습 시에는 Label이 필요 (1 = Real)
    dataset = Dataset.from_dict({
        "text": real_lyrics,
        "label": [1] * len(real_lyrics) # 모두 진짜 데이터이므로 1
    })
    return dataset

# ==========================================
# 실행
# ==========================================
if __name__ == "__main__":
    file_path = "Data/lyrics_by_year_1964_2023.csv"

    # 1. 데이터 로드
    df_clean = load_and_clean_data(file_path)

    # 2. PPO용 Query 데이터셋 생성
    # 데이터가 너무 많으면 학습이 오래 걸리므로, 테스트를 위해 샘플링(예: 1000개) 가능
    ppo_dataset_full = create_ppo_dataset(df_clean)
    print(f"PPO용 Query 샘플 수: {len(ppo_dataset_full)}")
    print(f"Query 예시: {ppo_dataset_full[0]['query']}")

    # 3. Discriminator용 데이터셋 생성
    disc_dataset = create_discriminator_dataset(df_clean)
    print(f"Discriminator용 Real 샘플 수: {len(disc_dataset)}")