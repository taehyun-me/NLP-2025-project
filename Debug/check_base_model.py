import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# ------------------------------------------------------------------------
# 1. 설정 (Configuration)
# ------------------------------------------------------------------------
# 사용하려는 모델 ID (Gemma-2-2b-it 또는 Llama-3.2-1B-Instruct 추천)
# 만약 로컬에 다운로드 받은 경로가 있다면 그 경로를 적어주세요.
MODEL_ID = "google/gemma-3-1b-it" 

print(f"🔄 모델 로드 중... ({MODEL_ID})")

# ------------------------------------------------------------------------
# 2. 모델 및 토크나이저 로드 (4-bit Quantization 적용)
# ------------------------------------------------------------------------
# 메모리 절약을 위한 4비트 양자화 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # 4-bit NormalFloat (성능 유지에 유리)
    bnb_4bit_compute_dtype=torch.float16 # 연산은 fp16으로 수행
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto" # GPU 자동 할당
)

# 파이프라인 생성 (텍스트 생성을 쉽게 하기 위함)
text_generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

print("✅ 모델 로드 완료! 테스트를 시작합니다.\n")

# ------------------------------------------------------------------------
# 3. 테스트 프롬프트 (SFT 필요 여부 판단용)
# ------------------------------------------------------------------------
# 다양한 스타일의 가사를 요청해봅니다.
test_prompts = [
    {
        "role": "user", 
        "content": "너는 한국어 작사가야. '헤어진 연인을 그리워하는 밤'을 주제로 감성적인 발라드 가사를 써줘. (Verse 1 - Chorus 구조로)"
    },
    {
        "role": "user", 
        "content": "너는 힙합 작사가야. '성공을 향한 열정'을 주제로 라임(Rhyme)을 살려서 랩 가사를 써줘."
    }
]

# ------------------------------------------------------------------------
# 4. 추론 및 결과 출력
# ------------------------------------------------------------------------
for i, msg in enumerate(test_prompts):
    print(f"--- [Test Case {i+1}] ---")
    print(f"주제: {msg['content']}")
    
    # Gemma/Llama의 채팅 템플릿 적용
    messages = [msg]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    outputs = text_generator(
        prompt,
        max_new_tokens=512,      # 생성할 최대 길이
        do_sample=True,          # 창의적인 생성을 위해 샘플링 사용
        temperature=0.8,         # 1.0에 가까울수록 창의적, 낮을수록 보수적
        top_p=0.9,
        repetition_penalty=1.1   # 같은 말 반복 방지
    )
    
    generated_text = outputs[0]["generated_text"]
    
    # 프롬프트 부분을 제외하고 생성된 답변만 추출 (모델마다 출력이 다를 수 있어 단순화)
    # 보통 <start_of_turn>model 이후가 답변입니다.
    answer = generated_text[len(prompt):]
    
    print("\n[모델 생성 결과]:")
    print(answer.strip())
    print("\n" + "="*50 + "\n")