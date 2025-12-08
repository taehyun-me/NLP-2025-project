import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

# ----------------------------------------------------------------
# 1. 설정
# ----------------------------------------------------------------
MODEL_ID = "google/gemma-3-1b-it"  # 학습 때 쓴 베이스 모델
RM_ADAPTER_PATH = "./Models/reward_model_output" # 학습된 RM 어댑터 경로
DATASET_PATH = "./Data/lyrics_rm_data.jsonl"

# ----------------------------------------------------------------
# 2. 모델 로드 (4-bit)
# ----------------------------------------------------------------
print("⚖️ Reward Model 로드 중...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# Base Model
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=1,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
# Load Adapter
model = PeftModel.from_pretrained(model, RM_ADAPTER_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ----------------------------------------------------------------
# 3. 평가 루프
# ----------------------------------------------------------------
# 데이터셋 로드 (검증용 데이터가 따로 없다면 전체에서 일부 샘플링)
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
# 9:1로 나눴던 그 10% 부분만 가져오기 (랜덤 시드 고정)
eval_dataset = dataset.train_test_split(test_size=0.1, seed=42)["test"]

print(f"📊 총 {len(eval_dataset)}개의 쌍(Pair)에 대해 평가를 시작합니다.")

correct_count = 0
total_count = 0

with torch.no_grad():
    for sample in tqdm(eval_dataset):
        prompt = sample["prompt"]
        chosen = sample["chosen"]
        rejected = sample["rejected"]
        
        # Chosen(정답) 점수 계산
        inputs_c = tokenizer(prompt + chosen, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        score_c = model(**inputs_c).logits[0][0].item()
        
        # Rejected(오답) 점수 계산
        inputs_r = tokenizer(prompt + rejected, return_tensors="pt", truncation=True, max_length=512).to("cuda")
        score_r = model(**inputs_r).logits[0][0].item()
        
        # 비교
        if score_c > score_r:
            correct_count += 1
        total_count += 1

accuracy = correct_count / total_count
print(f"\n✅ Reward Model 평가 결과")
print(f"   - 정확도: {accuracy:.2%}")
print(f"   (Good 가사가 Bad 가사보다 점수가 높게 나온 비율)")

if accuracy > 0.6:
    print("🎉 심사위원이 제대로 작동하고 있습니다!")
else:
    print("⚠️ 정확도가 낮습니다. 데이터셋 품질을 점검하거나 더 오래 학습해야 합니다.")