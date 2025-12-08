import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
import pandas as pd

# ----------------------------------------------------------------
# 1. 설정
# ----------------------------------------------------------------
BASE_MODEL_ID = "google/gemma-3-1b-it"
PPO_ADAPTER_PATH = "./Models/trainer_output/checkpoint-675"    # PPO 학습 결과
RM_ADAPTER_PATH = "./Models/reward_model_output" # 채점용 RM

# 평가할 프롬프트 예시 (다양하게 준비)
TEST_PROMPTS = [
    "이별의 아픔을 주제로 발라드 가사를 써줘.",
    "희망찬 내일을 향해 달려가는 댄스곡 가사 작사해줘.",
    "가을 밤에 어울리는 감성적인 노래 가사.",
    "힙합 스타일로 성공에 대한 야망을 표현한 가사.",
    "첫사랑의 설렘을 담은 아이돌 노래 가사."
]

# ----------------------------------------------------------------
# 2. 모델 로드 함수 (메모리 절약을 위해 함수화)
# ----------------------------------------------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

def get_model(adapter_path=None):
    print(f"🔄 모델 로드 중... (Adapter: {adapter_path if adapter_path else 'None'})")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return model, tokenizer

def get_reward_model():
    print("⚖️ 심사위원(RM) 모시는 중...")
    rm = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=1,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    rm = PeftModel.from_pretrained(rm, RM_ADAPTER_PATH)
    rm.eval()
    return rm

# ----------------------------------------------------------------
# 3. 평가 실행
# ----------------------------------------------------------------

# [Step 1] Base Model로 가사 생성
model, tokenizer = get_model(adapter_path=None) # 어댑터 없이 로드
base_results = []

print("1️⃣ Base Model 생성 시작...")
for prompt in tqdm(TEST_PROMPTS):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=64, 
            do_sample=True, 
            top_p=0.9,
            repetition_penalty=1.1
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 프롬프트 제거하고 답변만 추출
    generated = result[len(prompt):].strip()
    base_results.append(generated)

del model # 메모리 확보
torch.cuda.empty_cache()

# [Step 2] PPO Model로 가사 생성
model, tokenizer = get_model(adapter_path=PPO_ADAPTER_PATH) # PPO 어댑터 장착
ppo_results = []

print("2️⃣ PPO(RLHF) Model 생성 시작...")
for prompt in tqdm(TEST_PROMPTS):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=64, 
            do_sample=True, 
            top_p=0.9,
            repetition_penalty=1.1
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = result[len(prompt):].strip()
    ppo_results.append(generated)

del model
torch.cuda.empty_cache()

# [Step 3] Reward Model로 채점 및 비교
rm = get_reward_model()
scores_base = []
scores_ppo = []

print("3️⃣ 채점 중...")
with torch.no_grad():
    for i in range(len(TEST_PROMPTS)):
        # Base 점수
        input_base = tokenizer(TEST_PROMPTS[i] + base_results[i], return_tensors="pt", truncation=True).to("cuda")
        s_base = rm(**input_base).logits[0][0].item()
        scores_base.append(s_base)
        
        # PPO 점수
        input_ppo = tokenizer(TEST_PROMPTS[i] + ppo_results[i], return_tensors="pt", truncation=True).to("cuda")
        s_ppo = rm(**input_ppo).logits[0][0].item()
        scores_ppo.append(s_ppo)

# ----------------------------------------------------------------
# 4. 결과 출력
# ----------------------------------------------------------------
df = pd.DataFrame({
    "Prompt": TEST_PROMPTS,
    "Base_Lyrics": [t for t in base_results],
    "PPO_Lyrics": [t for t in ppo_results],
    "Base_Score": scores_base,
    "PPO_Score": scores_ppo,
    "Gap": [p - b for p, b in zip(scores_ppo, scores_base)]
})

print("\n" + "="*50)
print("📊 최종 성적표")
print("="*50)
print(df[["Base_Score", "PPO_Score", "Gap"]])

avg_base = sum(scores_base) / len(scores_base)
avg_ppo = sum(scores_ppo) / len(scores_ppo)

print(f"\n🏆 평균 점수 비교:")
print(f"   - Base Model: {avg_base:.2f}")
print(f"   - PPO  Model: {avg_ppo:.2f}")

if avg_ppo > avg_base:
    print(f"🎉 축하합니다! PPO 적용 후 평균 {avg_ppo - avg_base:.2f}점 상승했습니다!")
else:
    print("💧 PPO 모델 점수가 더 낮거나 비슷합니다. Reward Model이 너무 과적합되었거나 PPO 학습이 덜 되었을 수 있습니다.")

# 자세한 결과는 CSV로 저장
df.to_csv("evaluation_result.csv", index=False)
print("📄 상세 결과는 'evaluation_result.csv'에 저장되었습니다.")