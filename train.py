import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import pandas as pd
from sklearn.model_selection import train_test_split
import os

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

torch_dtype=torch.float16
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

BASE_MODEL = "google/gemma-2-9b-it"
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", quantization_config=quant_config, attn_implementation="eager")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL,trust_remote_code = True)
model = get_peft_model(model, lora_config)
model.config.use_cache = False

def load_and_split_data(file_path, column, threshold, test_size=0.2):
    csv_data = pd.read_csv(file_path)
    filtered_data = csv_data[csv_data[column] >= threshold]
    train_data, test_data = train_test_split(filtered_data, test_size=test_size)
    train_dataset = Dataset.from_pandas(train_data.reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_data.reset_index(drop=True))
    return train_dataset, test_dataset

file_path = "./dataset/sample_measured_dataset.csv"

df = pd.read_csv(file_path)

Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)

IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
lower_bound = lower_bound
train_dataset, test_dataset = load_and_split_data(file_path, 'value', lower_bound)

dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# 프롬프트 생성 함수 정의
def generate_prompt(example):
    prompt_list = []
    for i in range(len(example['instruction'])):
        prompt_list.append(
                f"""<bos><start_of_turn>user
                {example['instruction'][i]}
                <end_of_turn>
                <start_of_turn>model
                {example['output'][i]}<end_of_turn><eos>"""
                )

    return prompt_list

train_data = dataset['train']
test_data = dataset['test']


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
        #max_steps=-1,
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
        packing= False
)




trainer = SFTTrainer(
        model=model,
        peft_config=lora_config,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        formatting_func=generate_prompt
)


# 모델 트레이닝
trainer.train()

# LoRA 어댑터 저장
ADAPTER_MODEL = "zettel_adapter"
trainer.save_model(ADAPTER_MODEL)

