import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import RewardTrainer, RewardConfig 

# ----------------------------------------------------------------
# 1. 설정 (Configuration)
# ----------------------------------------------------------------
MODEL_ID = "google/gemma-3-1b-it" 
DATASET_PATH = "./Data/lyrics_rm_data.jsonl"
OUTPUT_DIR = "./Models/reward_model_output"

# 16GB VRAM을 위한 최적화 설정
# 가사 데이터 특성상 길이가 길 수 있으나, 메모리를 위해 1024 정도로 제한
MAX_LENGTH = 1024  

# ----------------------------------------------------------------
# 2. 데이터셋 로드
# ----------------------------------------------------------------
print(f"📂 데이터셋 로드 중: {DATASET_PATH}")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# 학습/검증 데이터 분리 (9:1)
dataset = dataset.train_test_split(test_size=0.1)
print(f"학습 데이터: {len(dataset['train'])}개, 검증 데이터: {len(dataset['test'])}개")

# ----------------------------------------------------------------
# 3. 모델 및 토크나이저 로드 (QLoRA 설정)
# ----------------------------------------------------------------
print("🔄 모델 및 토크나이저 준비 중...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Reward Model용 Classification Head (label=1)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=1, 
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.config.pad_token_id = tokenizer.pad_token_id

# ----------------------------------------------------------------
# 4. LoRA 설정
# ----------------------------------------------------------------
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ----------------------------------------------------------------
# 5. RewardConfig 설정 (최신 TRL 문법 적용)
# ----------------------------------------------------------------
# 기존 TrainingArguments 대신 RewardConfig를 사용합니다.
reward_config = RewardConfig(
    output_dir=OUTPUT_DIR,
    # 배치 사이즈와 Gradient Accumulation으로 메모리 조절
    per_device_train_batch_size=1, 
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8, 
    
    learning_rate=1e-4,
    num_train_epochs=1,
    
    # 메모리 절약 옵션
    gradient_checkpointing=True,
    fp16=False,
    bf16=True,
    
    # 평가 및 저장 주기
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=100,
    logging_steps=10,
    
    # RewardTrainer 전용 인자 (여기서 max_length를 지정)
    max_length=MAX_LENGTH,
    center_rewards_coefficient=0.01, # 학습 안정성을 위한 옵션 (선택사항)
    
    remove_unused_columns=False,
    report_to="none", 
)

# ----------------------------------------------------------------
# 6. RewardTrainer 실행
# ----------------------------------------------------------------
# tokenizer 인자가 'processing_class'로 변경되었습니다.
trainer = RewardTrainer(
    model=model,
    processing_class=tokenizer,  # <--- 변경된 부분
    args=reward_config,          # <--- 변경된 부분
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
)

print("🚀 학습 시작! (Reward Model Training - New TRL Version)")
trainer.train()

# ----------------------------------------------------------------
# 7. 저장
# ----------------------------------------------------------------
print(f"💾 모델 저장 중: {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)
print("✅ Reward Model 학습 및 저장 완료!")