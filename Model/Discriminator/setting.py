from sklearn.model_selection import train_test_split
from Model.dataloader import (
    all_labels,
    all_texts,
)
from Model.setting import tokenizer, MAX_LEN, BATCH_SIZE
from dataset import CustomDataset
from torch.utils.data import DataLoader

train_texts, val_texts, train_labels, val_labels = train_test_split(
    all_texts, all_labels, test_size=0.1, random_state=42
)

train_encodings = tokenizer(train_texts, truncation=True, padding='max_length', max_length=MAX_LEN)
val_encodings = tokenizer(val_texts, truncation=True, padding='max_length', max_length=MAX_LEN)

train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, val_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

print(f"\nDiscriminator Training Data Preparation Complete:")
print(f"  Total Samples: {len(all_texts)}")
print(f"  Train Samples: {len(train_dataset)}")
print(f"  Validation Samples: {len(val_dataset)}")
print(f"  Max Sequence Length: {MAX_LEN}")