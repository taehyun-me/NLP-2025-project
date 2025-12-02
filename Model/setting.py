import torch
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from torch.optim import Adam, AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import pandas as pd

# Discriminator 설정 (이전 단계에서 정의됨)
D_MODEL_NAME = "beomi/kcbert-base"
tokenizer = AutoTokenizer.from_pretrained(D_MODEL_NAME)
MAX_LEN = 128 # Discriminator 입력의 최대 길이
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

# device 확인 (local : gpu (5060ti 16gb))
print(device)