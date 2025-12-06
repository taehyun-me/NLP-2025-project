import pandas as pd
import torch
import random
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score

# ==========================================
# 1. 가짜 데이터 생성 함수 (Negative Sampling)
# ==========================================
def create_fake_sentence(text, method="shuffle"):
    words = text.split()
    if len(words) < 2:
        return text # 너무 짧으면 그대로 (나중에 필터링)
    
    if method == "shuffle":
        # 단어 순서를 섞어 문법을 파괴함
        random.shuffle(words)
        return " ".join(words)
    
    return text

# ==========================================
# 2. 데이터셋 구축 (Real vs Fake)
# ==========================================
def prepare_discriminator_dataset(file_path):
    print("데이터 로드 및 가짜 데이터 생성 중...")
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['lyric'])
    
    # 2000년 이후 데이터만 사용 (최신 가사 스타일 반영 권장)
    # df = df[df['year'] >= 2000] 

    real_texts = []
    fake_texts = []

    for _, row in df.iterrows():
        lyrics = str(row['lyric']).split('\n')
        for line in lyrics:
            line = line.strip()
            if len(line) < 10: # 너무 짧은 문장은 제외
                continue
                
            # Real Data
            real_texts.append(line)
            
            # Fake Data (단어 섞기)
            fake_texts.append(create_fake_sentence(line, method="shuffle"))
    
    # 데이터 밸런스 맞추기 (Real 개수만큼만 Fake 사용)
    min_len = min(len(real_texts), len(fake_texts))
    real_texts = real_texts[:min_len]
    fake_texts = fake_texts[:min_len]

    # 데이터셋 통합
    texts = real_texts + fake_texts
    labels = [1] * len(real_texts) + [0] * len(fake_texts) # 1: Real, 0: Fake
    
    # 셔플
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)

    dataset = Dataset.from_dict({"text": texts, "label": labels})
    
    # Train/Test 분리 (9:1)
    return dataset.train_test_split(test_size=0.1)

# ==========================================
# 3. 평가 함수
# ==========================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return {"accuracy": acc, "f1": f1}

# ==========================================
# 4. 학습 실행 메인 함수
# ==========================================
def train_discriminator():
    # 설정
    model_name = "beomi/kcbert-base"
    file_path = "Data/lyrics_by_year_1964_2023.csv" # 경로 확인 필요
    output_dir = "./discriminator_model_result"
    
    # 1. 데이터 준비
    dataset = prepare_discriminator_dataset(file_path)
    print(f"학습 데이터 수: {len(dataset['train'])}, 평가 데이터 수: {len(dataset['test'])}")
    
    # 2. 토크나이저 및 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2 # 0: Fake, 1: Real
    )
    
    # 3. 토크나이징
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=64
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 4. Trainer 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,              # 3 epoch 정도면 충분
        per_device_train_batch_size=32,  # BERT-base는 가벼워서 32~64 가능
        per_device_eval_batch_size=64,
        eval_strategy="epoch",     # 매 epoch마다 평가
        save_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=True,                       # GPU 가속
        save_total_limit=1,              # 용량 절약을 위해 베스트 모델 1개만 저장
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
    )
    
    # 5. 학습 시작
    print(">>> Discriminator 학습 시작...")
    trainer.train()
    
    # 6. 최종 모델 저장 (이 경로를 PPO에서 씁니다)
    final_path = "./best_discriminator_model"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f">>> 학습 완료! 모델이 '{final_path}'에 저장되었습니다.")

if __name__ == "__main__":
    train_discriminator()