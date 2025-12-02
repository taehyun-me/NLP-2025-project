from Model.Discriminator.setting import train_loader
from Model.setting import tokenizer

# 1. DataLoader에서 첫 번째 배치 추출
# 이전에 구성된 train_loader 인스턴스를 사용합니다.
if 'train_loader' not in locals():
    print("Error: train_loader가 정의되지 않았습니다. 이전 단계의 데이터 준비 코드를 먼저 실행하세요.")
else:
    # 데이터 로더에서 이터레이터를 생성하고 첫 번째 배치를 가져옵니다.
    batch = next(iter(train_loader))

    # 2. 추출된 배치의 형태(Shape) 확인
    print("==============================================")
    print("✅ DataLoader 배치 출력 확인")
    print("==============================================")
    print(f"Input IDs (토큰 ID) Shape: {batch['input_ids'].shape}")
    print(f"Attention Mask (어텐션 마스크) Shape: {batch['attention_mask'].shape}")
    print(f"Labels (레이블) Shape: {batch['labels'].shape}")

    # 3. 데이터 내용 확인 (첫 번째 샘플)
    sample_index = 1

    # Input IDs 및 Attention Mask 확인
    sample_ids = batch['input_ids'][sample_index].cpu().numpy()
    sample_mask = batch['attention_mask'][sample_index].cpu().numpy()
    sample_label = batch['labels'][sample_index].item()

    # 토큰 ID를 실제 텍스트로 디코딩
    decoded_text = tokenizer.decode(sample_ids, skip_special_tokens=True)

    # 4. 결과 출력
    print("\n--- 첫 번째 샘플 내용 ---")
    print(f"레이블 (0: Fake, 1: Real): {sample_label:.0f}")
    print(f"디코딩된 텍스트:\n'{decoded_text}'")

    print("\n--- 텐서 상세 정보 (일부) ---")
    print(f"Input IDs (일부): {sample_ids[:10]}")
    print(f"Attention Mask (일부): {sample_mask[:10]}")

    # 레이블 해석
    if sample_label == 1.0:
        print("-> 이 샘플은 실제 가사(Real)입니다.")
    else:
        print("-> 이 샘플은 가짜 데이터(Fake)입니다.")

    print("==============================================")