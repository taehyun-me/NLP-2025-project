import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)
from trl import PPOTrainer, PPOConfig
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm

from dataloader import (
    load_and_clean_data,
    create_ppo_dataset,
    create_discriminator_dataset,
)

# ==========================================
# 1. 4-bit 양자화 설정 (공통)
# ==========================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# ==========================================
# 2. 모델 로드
# ==========================================
model_name = "google/gemma-3-1b-it"
reward_model_name = "./best_discriminator_model"

# [A] Policy Model (Generator) - Gemma
# -------------------------------------
# AutoModelForCausalLMWithValueHead 대신 기본 CausalLM 사용 권장 (새 트레이너 구조상)
policy_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config, # 4-bit
    device_map="auto"
)
# Policy용 LoRA 설정
policy_peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none"
)
policy_model = get_peft_model(policy_model, policy_peft_config)


# [B] Value Model (Critic) - Gemma (Policy와 같은 구조 권장)
# -------------------------------------
# 16GiB 환경에서 메모리를 아끼기 위해 얘도 4-bit로 로드
# num_labels=1 (점수 하나만 출력)
value_model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=1,
    quantization_config=bnb_config, # 4-bit 필수
    device_map="auto"
)
# Value Model용 LoRA 설정 (Classifier용)
value_peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none"
)
value_model = get_peft_model(value_model, value_peft_config)


# [C] Reward Model (Discriminator) - KcBERT
# -------------------------------------
# Trainer의 인자 요구사항을 맞추기 위해 로드
reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_name,
    num_labels=1, # 점수 출력
    quantization_config=bnb_config, # 4-bit (BERT는 작아서 안해도 되지만 안전하게)
    device_map="auto"
)

# [D] Tokenizer
# -------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token


# ==========================================
# 3. PPO Config (TrainingArguments 상속 스타일)
# ==========================================
config = PPOConfig(
    output_dir="./ppo_result",
    learning_rate=1.41e-5,
    per_device_train_batch_size=4,    # 배치 사이즈
    gradient_accumulation_steps=4,    # 누적
    gradient_checkpointing=True,      # 메모리 절약
    logging_steps=10,
    
    # PPO 전용
    num_ppo_epochs=2,
    kl_coef=0.05,
    remove_unused_columns=False,
)


# ==========================================
# 3-1. 데이터셋 준비
# ==========================================
file_path = "Data/lyrics_by_year_1964_2023.csv"
df_clean = load_and_clean_data(file_path)

ppo_dataset_full = create_ppo_dataset(df_clean)
disc_dataset = create_discriminator_dataset(df_clean)

# 학습 속도를 위해 1000개만 샘플링 (실전에서는 늘리세요)
query_dataset = ppo_dataset_full.shuffle(seed=42).select(range(1000))

def tokenize_fn(examples):
    outputs = tokenizer(
        examples["query"],
        padding="max_length",
        max_length=32,
        truncation=True
    )
    return {"input_ids": outputs["input_ids"]}

dataset = query_dataset.map(tokenize_fn, batched=True)
dataset.set_format(type="torch", columns=["input_ids"], output_all_columns=True)


# ==========================================
# 3-2. Collator 정의 (누락된 부분 추가)
# ==========================================
collator = DataCollatorWithPadding(
    tokenizer=tokenizer,       # 필수: 토크나이저 전달
    padding=True,              # True: 배치 내 가장 긴 시퀀스에 맞춰 패딩 (Dynamic Padding)
    max_length=None,           # None: 제한 없음 (모델 max_length 따름)
    pad_to_multiple_of=8,      # 8의 배수로 맞춤 (Volta 아키텍처 이상 GPU에서 속도 향상)
    return_tensors="pt"        # PyTorch 텐서 반환
)


# ==========================================
# 4. PPOTrainer 초기화 (제공해주신 클래스 정의에 맞춤)
# ==========================================
ppo_trainer = PPOTrainer(
    args=config,                    # Config 전달
    processing_class=tokenizer,     # [변경점] tokenizer -> processing_class
    model=policy_model,             # Policy
    ref_model=None,                 # LoRA 사용 시 None (자동 처리)
    reward_model=reward_model,      # [필수] Reward Model
    value_model=value_model,        # [필수] Value Model
    train_dataset=dataset,          # 데이터셋
    data_collator=collator,         # Collator
    peft_config=policy_peft_config, # PEFT 설정 (선택사항)
)

print("PPO Trainer 초기화 완료 (Modular 구조)")

# ==========================================
# 5-1. 생성 설정 (Generation Config)
# ==========================================
# 모델이 가사를 쓸 때의 창의성 조절
generation_kwargs = {
    "min_length": -1, # 최소 길이 제한 없음
    "top_k": 0.0,     # Top-k 샘플링 끔 (창의성보다는 확률 분포 따름)
    "top_p": 1.0,     # Nucleus 샘플링
    "do_sample": True, # 샘플링 활성화 (다양한 가사 생성)
    "pad_token_id": tokenizer.eos_token_id, # 패딩은 EOS로
    "max_new_tokens": 32, # 생성할 가사 길이 (너무 길면 메모리 터짐 주의)
}

# ==========================================
# 5-2. Reward Pipeline 함수 정의
# ==========================================
# 서로 다른 토크나이저를 연결해주는 핵심 함수
def get_rewards(texts, reward_model, reward_tokenizer, device):
    inputs = reward_tokenizer(
        texts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=64
    ).to(device)
    
    with torch.no_grad():
        outputs = reward_model(**inputs)
        # 로직(Logits)을 확률(Softmax)로 변환하거나, 
        # Binary Classification의 경우 Logit[1] (Real 클래스) 자체를 점수로 쓸 수도 있음
        # 여기서는 Logit 그 자체를 점수로 사용 (안정적 학습)
        logits = outputs.logits
        
        # 가정: Label 1이 "Real(좋은 가사)", Label 0이 "Fake"
        # Label 1에 해당하는 점수를 추출
        rewards = logits[:, 1] 
        
    return rewards

# Reward Model용 토크나이저 별도 로드 (KcBERT용)
from transformers import AutoTokenizer
reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)

# ==========================================
# 5-3. 실제 학습 루프 실행
# ==========================================
print("학습 시작! (Ctrl+C로 중단 가능)")

# 모델 저장 경로
save_dir = "./final_korean_lyric_model"

for epoch in range(config.num_ppo_epochs):
    print(f"Epoch {epoch + 1}/{config.num_ppo_epochs}")
    
    for batch in tqdm(ppo_trainer.dataloader):
        try:
            # 1. [Input] 시드 텍스트(Query) 가져오기
            query_tensors = batch["input_ids"]
            
            # 2. [Generate] 모델이 가사 생성 (Response)
            # response_tensors는 [Query + Response] 전체를 포함함
            generated_sequence = ppo_trainer.generate(
                query_tensors, 
                **generation_kwargs
            )
            
            # Query 부분을 잘라내고 순수 생성된 부분만 추출 (Response)
            # Gemma Tokenizer 기준
            response_tensors = []
            for query, gen_seq in zip(query_tensors, generated_sequence):
                # 쿼리 길이만큼 잘라냄
                response_tensors.append(gen_seq[len(query):])
            
            # 3. [Decode] Discriminator에게 보여주기 위해 텍스트로 변환
            batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]
            
            # 4. [Score] KcBERT로 점수 매기기 (Reward Calculation)
            # 생성된 텍스트가 비어있지 않은지 확인
            valid_texts = [text if len(text.strip()) > 0 else "..." for text in batch["response"]]
            
            # 점수 계산 (List of Tensors 형태여야 함)
            reward_tensors = get_rewards(
                valid_texts, 
                reward_model, 
                reward_tokenizer, 
                ppo_trainer.accelerator.device
            )
            
            # List[torch.Tensor] 형태로 변환 (trl 요구사항)
            reward_list = [r for r in reward_tensors]
            
            # 5. [Step] PPO 업데이트
            # query: 질문, response: 대답(토큰), reward: 점수
            stats = ppo_trainer.step(query_tensors, response_tensors, reward_list)
            
            # 6. [Log] 로그 기록 (WandB 등이 켜져 있다면 자동 전송됨)
            ppo_trainer.log_stats(stats, batch, reward_list)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("| WARNING: OOM 발생. 배치를 건너뜁니다.")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

# ==========================================
# 6. 학습된 모델 저장
# ==========================================
print("학습 완료! 모델 저장 중...")
ppo_trainer.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
print(f"모델이 {save_dir}에 저장되었습니다.")