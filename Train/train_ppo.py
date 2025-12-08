import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    DataCollatorWithPadding
)
from trl import PPOTrainer, PPOConfig
from peft import LoraConfig, get_peft_model, PeftModel
from datasets import load_dataset
import sys

# ----------------------------------------------------------------
# 1. 설정 (Configuration)
# ----------------------------------------------------------------
# [중요] Reward Model 학습 때 사용했던 모델과 동일해야 함 (사이즈 불일치 에러 해결)
MODEL_ID = "google/gemma-3-1b-it" 
RM_ADAPTER_PATH = "./Models/reward_model_output" 

config = PPOConfig(
    exp_name="lyrics_ppo_project",
    learning_rate=1.41e-5,
    
    # 배치 사이즈 및 최적화
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=8,
    
    # PPO 하이퍼파라미터
    num_ppo_epochs=1,
    kl_coef=0.05,
    
    # 메모리 절약
    gradient_checkpointing=True,
    fp16=False,
    bf16=True, # RTX 5060 Ti 지원
)

# ----------------------------------------------------------------
# 2. 모델 로드 (공통 설정: 4-bit)
# ----------------------------------------------------------------
print("🔄 모델 로드 준비 중... (Policy, Reward, Value)")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# ----------------------------------------------------------------
# 3. 각 모델 개별 로드
# ----------------------------------------------------------------

# [A] Policy Model (Actor)
print("1️⃣ Policy Model (Actor) 로드 중...")
policy_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
# 생성 설정 강제 주입 (Trainer 내부에서 사용)
policy_model.generation_config.max_new_tokens = 64
policy_model.generation_config.pad_token_id = tokenizer.pad_token_id
policy_model.generation_config.do_sample = True
policy_model.generation_config.top_p = 1.0

policy_peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
policy_model = get_peft_model(policy_model, policy_peft_config)


# [B] Reward Model (Evaluator)
print("2️⃣ Reward Model (Evaluator) 로드 중...")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=1,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
# 학습된 어댑터 로드 (에러가 나면 MODEL_ID가 맞는지 확인 필수)
try:
    reward_model = PeftModel.from_pretrained(reward_model, RM_ADAPTER_PATH)
    reward_model.eval() # 학습되지 않도록 설정
    reward_model.requires_grad_(False) # 그래디언트 계산 끔
except Exception as e:
    print(f"❌ Reward Model 로드 실패: {e}")
    print("💡 팁: train_reward_model.py에서 사용한 MODEL_ID와 현재 MODEL_ID가 같은지 확인하세요.")
    sys.exit(1)


# [C] Value Model (Critic)
print("3️⃣ Value Model (Critic) 로드 중...")
value_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=1,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
value_peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="SEQ_CLS",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)
value_model = get_peft_model(value_model, value_peft_config)


# ----------------------------------------------------------------
# 4. 데이터셋 준비 (수정됨: Train/Eval 분리)
# ----------------------------------------------------------------
print("📂 데이터셋 준비 중...")
raw_dataset = load_dataset("json", data_files="./Data/lyrics_rm_data.jsonl", split="train")

# 1. 데이터셋을 9:1로 분할 (Train 90%, Eval 10%)
# 이렇게 하면 eval_dataset이 자동으로 생성됩니다.
split_dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

print(f"   - 학습 데이터: {len(train_dataset)}개")
print(f"   - 검증 데이터: {len(eval_dataset)}개")

# 2. 토크나이저 설정
tokenizer.padding_side = "left" # 생성 모델은 왼쪽 패딩 필수

# 3. 전처리 함수
def tokenize(sample):
    # max_length는 모델 컨텍스트에 맞춰 조절 (여기선 512)
    outputs = tokenizer(sample["prompt"], padding=False, truncation=True, max_length=512)
    return {"input_ids": outputs["input_ids"]}

# 4. 매핑 및 컬럼 제거 (Train/Eval 각각 적용)
# remove_columns로 불필요한 텍스트 컬럼을 제거해야 충돌 방지
train_dataset = train_dataset.map(tokenize, batched=False, remove_columns=["prompt", "chosen", "rejected"])
eval_dataset = eval_dataset.map(tokenize, batched=False, remove_columns=["prompt", "chosen", "rejected"])

# 5. DataCollator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ----------------------------------------------------------------
# 5. PPOTrainer 초기화
# ----------------------------------------------------------------
print("🚀 PPOTrainer 초기화...")

trainer = PPOTrainer(
    args=config,
    processing_class=tokenizer,
    model=policy_model,
    ref_model=None,
    reward_model=reward_model,
    value_model=value_model,
    train_dataset=train_dataset, # 학습용
    eval_dataset=eval_dataset,   # 검증용
    data_collator=data_collator,
)

print("🔥 PPO 학습 시작! (자동화된 루프 실행)")
trainer.train()