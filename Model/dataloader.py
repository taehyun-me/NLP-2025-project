from util import load_and_clean_data
from fake import generate_initial_fake_data

# 3. 데이터 로드 및 통합
FILE_PATH = "Data/lyrics_by_year_1964_2023.csv"
real_texts = load_and_clean_data(FILE_PATH)

if real_texts is None:
    exit()

NUM_REAL = len(real_texts)
fake_texts = generate_initial_fake_data(NUM_REAL) # 이제 len(fake_texts) == NUM_REAL

# 전체 데이터셋 구성
all_texts = real_texts + fake_texts
all_labels = [1] * NUM_REAL + [0] * NUM_REAL

# 길이 일치 확인 (디버깅)
print(f"Final all_texts length: {len(all_texts)}")
print(f"Final all_labels length: {len(all_labels)}")
if len(all_texts) != len(all_labels):
    print("FATAL ERROR: Lengths still do not match.")
else:
    print("SUCCESS: Data lengths are consistent.")