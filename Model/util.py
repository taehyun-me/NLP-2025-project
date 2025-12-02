import pandas as pd

def load_and_clean_data(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    df = df.dropna(subset=['lyric']) # 가사가 없는 행 제거

    # 가사의 줄 바꿈 문자(\n)를 공백으로 변환 -> 개선 필요
    df['clean_lyric'] = df['lyric'].str.replace('\n', ' ').str.strip()

    print(f"Loaded {len(df)} lyrics after cleaning.")
    return df['clean_lyric'].tolist()