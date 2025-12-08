# 🎵 Korean Lyrics Generator with RLHF (PPO)

> **PPO(Proximal Policy Optimization) 기반의 강화학습(RLHF)을 적용하여, 감성적이고 운율이 살아있는 한국어 가사를 생성하는 AI 프로젝트입니다.**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![TRL](https://img.shields.io/badge/Library-TRL-blueviolet)

## 📖 프로젝트 개요

일반적인 대규모 언어 모델(LLM)은 단순히 다음 단어를 예측하도록 학습되어 있어, 노래 가사처럼 **운율(Rhyme)**이 중요하고 **구조적(Verse/Chorus)**인 텍스트를 생성하는 데 한계가 있습니다.

이 프로젝트는 **Gemma-3-1B-it** 모델을 기반으로, 인간의 피드백을 모사한 **Reward Model**과 **PPO 알고리즘**을 사용하여 모델이 "사람이 선호하는 가사 형태"를 학습하도록 조정(Alignment)했습니다.

### ✨ 핵심 기능 및 특징
* **QLoRA & 4-bit Quantization:** 소비자용 GPU(RTX 5060 Ti, 16GB VRAM) 환경에서 학습 가능하도록 경량화.
* **Synthetic Data Generation:** 기존 가사 데이터를 활용해 '좋은 가사'와 '망가진 가사(설명조)' 쌍을 자동 생성하여 Reward Model 학습.
* **RLHF Pipeline:** `SFT(생략/Base)` -> `Reward Modeling` -> `PPO` 의 전체 파이프라인 구현.

---

## 📂 디렉토리 구조 (Directory Structure)

```bash
NLP-2025-project/
├── Data/                   # 데이터셋 관리
│   ├── lyrics_by_year_...csv  # 원본 가사 데이터
│   ├── convert_csv_to_jsonl.py # CSV -> 학습용 JSONL 변환 스크립트
│   └── lyrics_rm_data.jsonl    # 변환된 Reward Model 학습 데이터
├── Models/                 # 학습된 모델 및 어댑터 저장소
│   ├── reward_model_output/   # 학습된 Reward Model 어댑터
│   └── trainer_output/        # (PPO 등) 체크포인트
├── Train/                  # 학습 스크립트
│   ├── train_reward_model.py   # Reward Model 학습 (LoRA)
│   └── train_ppo.py            # PPO 강화학습 (RLHF Main)
├── Evaluate/               # 성능 평가 및 검증
│   ├── eval_reward_model.py    # Reward Model 정확도 측정
│   ├── eval_policy_model.py    # Base vs PPO 모델 생성 결과 비교
│   └── evaluation_result.csv   # 평가 결과 저장
├── Debug/                  # 디버깅 및 테스트
│   ├── check_base_model.py     # Base 모델 기본 성능 테스트
│   └── check_reward_model.py   # RM 점수 산출 테스트
└── requirements.txt        # 필요 라이브러리 목록
```

#### 🚀 시작하기 (Getting Started)

프로젝트를 로컬 환경에서 실행하기 위한 단계별 가이드입니다.

#### 1\. 환경 설정 (Installation)

이 프로젝트는 Python 3.10+ 및 CUDA가 지원되는 GPU 환경(VRAM 12GB 이상 권장)이 필요합니다.

```bash
# 1. 저장소 복제
git clone https://github.com/your-username/NLP-2025-project.git
cd NLP-2025-project

# 2. 가상환경 생성 (선택사항)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 필요 라이브러리 설치
pip install -r requirements.txt
```

#### 2\. 데이터 준비 (Data Preparation)

CSV 형식의 원본 가사 데이터를 학습 가능한 `jsonl` 포맷(Chosen/Rejected 쌍)으로 변환합니다.

  * **입력:** `Data/lyrics_by_year_1964_2023.csv`
  * **출력:** `Data/lyrics_rm_data.jsonl`

<!-- end list -->

```bash
python Data/convert_csv_to_jsonl.py
```

#### 3\. 리워드 모델 학습 (Reward Model Training)

생성된 데이터셋을 사용하여, 모델이 '좋은 가사'와 '나쁜 가사'를 구분할 수 있도록 Reward Model을 학습합니다.

  * **모델:** `google/gemma-3-1b-it` (QLoRA 적용)
  * **결과물:** `Models/reward_model_output/`

<!-- end list -->

```bash
python Train/train_reward_model.py
```

#### 4\. PPO 강화학습 (PPO Training - RLHF)

학습된 Reward Model을 심사위원으로 삼아, Policy Model(Actor)이 더 높은 점수를 받는 가사를 생성하도록 강화학습(PPO)을 진행합니다.

  * **입력:** Base Model + Reward Model Adapter
  * **결과물:** `Models/ppo_final_model/`

<!-- end list -->

```bash
python Train/train_ppo.py
```

-----

### 📊 평가 (Evaluation)

RLHF 적용 전(Base Model)과 후(PPO Model)의 성능 차이를 정량적/정성적으로 평가합니다.

#### 평가 실행

준비된 프롬프트 셋을 사용하여 두 모델이 각각 가사를 생성하고, Reward Model이 이를 채점하여 비교합니다.

```bash
python Evaluate/eval_policy_model.py
```

#### 평가 결과 예시

실행 후 `Evaluate/evaluation_result.csv` 파일에 상세 결과가 저장됩니다.

| Prompt | Base Model (Before) | PPO Model (After) | Score Gap |
| :--- | :--- | :--- | :--- |
| **"이별의 아픔을 주제로..."** | 이별은 슬프다. 나는 운다.<br>너는 떠났다. (단조로운 문장) | 차가운 바람이 불어오면<br>그대 향기 흩어지네 (운율 형성) | **+2.5** 🔼 |
| **"희망찬 내일을 노래해줘"** | 내일은 해가 뜬다.<br>열심히 살자. | 저기 떠오르는 태양처럼<br>우리 꿈도 빛날 거야 | **+3.1** 🔼 |

-----

### 🛠️ 기술 스택 (Tech Stack)

본 프로젝트는 최신 LLM 파인튜닝 및 강화학습 라이브러리를 적극 활용하여 구축되었습니다.

#### Core Libraries

  *  **(Model Loading & Inference)**
  *  **(PPO & Reward Trainer)**
  *  **(LoRA/QLoRA Adapter)**

#### Hardware Optimization (for 16GB VRAM)

단일 소비자용 GPU(RTX 5060 Ti)에서 3개의 모델(Policy, Reward, Value)을 동시에 로드하기 위해 다음과 같은 최적화 기술을 적용했습니다.

  * **QLoRA (4-bit Quantization):** `bitsandbytes`를 사용하여 모델 가중치를 4비트로 압축, 메모리 사용량을 약 1/4로 절감.
  * **Gradient Checkpointing:** 중간 활성화(Activation) 값을 저장하지 않고 재계산하여 메모리 확보.
  * **Gradient Accumulation:** 작은 배치 사이즈(Batch=1)로 학습하되, 그래디언트를 누적하여 큰 배치 사이즈 효과 구현.

#### Models

  * **Base Model:** `google/gemma-2-2b-it` (Instruct Tuned)
  * **Reward Model:** Gemma-2-2B based Binary Classifier (Scalar Output)
  * **Policy/Value Model:** Gemma-2-2B with LoRA Adapters
