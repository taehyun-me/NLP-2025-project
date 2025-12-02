import numpy as np

# 2. Fake Data 생성 함수 (오류 수정)
def generate_initial_fake_data(num_samples):
    """
    초기 Discriminator 학습을 위한 임시 Fake Data를 생성합니다.
    num_samples 개수와 정확히 일치하도록 생성 로직을 수정했습니다.
    """

    # 1. 무작위 한국어 글자 조합 (약 num_samples // 2 개)
    chars = "가나다라마바사아자차카타파하" * 10
    num_random = num_samples // 2
    random_texts = [''.join(np.random.choice(list(chars), size=np.random.randint(20, 100)))
                    for _ in range(num_random)]

    # 2. 일반 한국어 문장 (나머지 샘플)
    # 필요한 나머지 샘플 개수를 정확히 계산
    remaining_samples = num_samples - len(random_texts)

    base_sentences = [
        "오늘의 날씨는 매우 맑고 청명하며 구름 한 점 없는 화창한 가을 하늘입니다.",
        "인공지능과 머신러닝 기술은 산업 전반에 걸쳐 혁신적인 변화를 가져오고 있습니다.",
        "서울 지하철 노선도는 복잡하지만 편리하게 주요 도심을 연결하여 효율적인 대중교통 시스템을 제공합니다.",
        "새로운 프로젝트의 목표는 시장 점유율을 20%까지 끌어올리고 브랜드 인지도를 강화하는 것입니다."
    ]

    # base_sentences를 반복하면서 필요한 개수(remaining_samples)만큼만 리스트에 추가
    general_sentences = []
    for i in range(remaining_samples):
        # 모듈로 연산(%)을 사용하여 base_sentences를 순환적으로 재사용
        general_sentences.append(base_sentences[i % len(base_sentences)])

    # len(random_texts) + len(general_sentences) == num_samples 보장
    return random_texts + general_sentences