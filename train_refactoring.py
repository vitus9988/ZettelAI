import os
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForCausalLM, 
                          TrainingArguments, BitsAndBytesConfig)
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer

# 환경 변수 설정
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Torch 데이터 타입 설정
torch_dtype = torch.float16
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# LoRA 설정 구성
lora_config = LoraConfig(
    r=6,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

# 모델 및 토크나이저 로드
BASE_MODEL = "google/gemma-2-9b-it"
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", quantization_config=quant_config, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = get_peft_model(model, lora_config)
model.config.use_cache = False

def load_and_split_data(file_path, column, threshold, test_size=0.2):
    """
    주어진 CSV 파일에서 데이터를 로드하고 지정된 컬럼을 기준으로 데이터를 필터링하여 훈련 및 테스트 세트로 나눕니다.
    """
    csv_data = pd.read_csv(file_path)
    filtered_data = csv_data[csv_data[column] >= threshold]
    train_data, test_data = train_test_split(filtered_data, test_size=test_size)
    return Dataset.from_pandas(train_data.reset_index(drop=True)), Dataset.from_pandas(test_data.reset_index(drop=True))

# 데이터 로드 및 분할
file_path = "./dataset/sample_measured_dataset.csv"
df = pd.read_csv(file_path)
lower_bound = df['value'].quantile(0.25) - 1.5 * (df['value'].quantile(0.75) - df['value'].quantile(0.25))
train_dataset, test_dataset = load_and_split_data(file_path, 'value', lower_bound)

dataset = DatasetDict({'train': train_dataset, 'test': test_dataset})

def generate_prompt(example):
    """
    주어진 예제를 기반으로 프롬프트를 생성합니다.
    """
    return [f"<bos><start_of_turn>user\n{instruction}\n<end_of_turn>\n<start_of_turn>model\n{output}<end_of_turn><eos>"
            for instruction, output in zip(example['instruction'], example['output'])]

# 훈련 인자 설정
training_args = SFTConfig(
    output_dir="./results",
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    eval_strategy="steps",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=64,
    learning_rate=2e-4,
    weight_decay=0.01,
    num_train_epochs=1,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    log_level="info",
    logging_steps=20,
    save_strategy="steps",
    bf16=False,
    fp16=False,
    seed=42,
    max_seq_length=1024,
    optim="paged_adamw_8bit",
    packing=False
)

# 트레이너 설정 및 모델 훈련
trainer = SFTTrainer(
    model=model,
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    formatting_func=generate_prompt
)

trainer.train()

# 모델 저장
ADAPTER_MODEL = "zettel_adapter"
trainer.save_model(ADAPTER_MODEL)

'''
개선점:
1. 코드 블록을 기능별로 나누어 가독성을 높였습니다.
2. `load_and_split_data` 함수에 docstring을 추가하여 함수의 역할을 명확히 했습니다.
3. `generate_prompt` 함수에서 `for` 루프 대신 리스트 컴프리헨션을 사용하여 코드 간결성을 개선했습니다.
4. `lower_bound` 계산을 간소화하여 중복된 코드를 제거했습니다.
5. 불필요한 변수 선언을 제거하고, 각 설정에 대한 주석을 추가하여 코드 이해를 돕도록 했습니다. 
6. 환경 변수 설정, 모델 로드, 데이터 로드 및 처리 등 각 기능이 명확히 구분되어 있어 유지보수와 수정이 용이해졌습니다.
'''