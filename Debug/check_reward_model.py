import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

# 1. 설정
BASE_MODEL = "google/gemma-3-1b-it"
ADAPTER_PATH = "./Models/reward_model_output"  # 방금 학습된 어댑터 경로

print("⚖️ 심사위원(Reward Model) 모시는 중...")

# 2. 베이스 모델 로드 (4-bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # 학습때 썼던 bf16
)

model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL,
    num_labels=1,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# 3. 학습된 LoRA 어댑터 합치기
model = PeftModel.from_pretrained(model, ADAPTER_PATH)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# 4. 검증 함수
def get_score(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits[0][0].item()

# 5. 테스트 케이스
print("\n[검증 시작]")

good_lyric = """
차가운 바람이 불어오면
그대 생각에 잠겨요
유난히 밝은 저 달빛도
내 맘을 아는지 슬피 울죠
"""

bad_lyric = """
바람이 차갑게 불면 너 생각이 점점 난다.
달빛이 엄청 밝은데 내 마음을 아는 것 같아서 슬프다.
그래서 나는 오늘 밤에 잠을 잘 못 잘 것 같다.
"""

score_good = get_score(good_lyric)
score_bad = get_score(bad_lyric)

print(f"🎵 잘 쓴 가사 점수: {score_good:.4f}")
print(f"📝 못 쓴 가사 점수: {score_bad:.4f}")

if score_good > score_bad:
    print("✅ 검증 성공! 심사위원이 제대로 판단하고 있습니다.")
else:
    print("❌ 검증 실패... 더 많은 학습이 필요합니다.")