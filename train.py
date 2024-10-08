import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
import pandas as pd
from sklearn.model_selection import train_test_split

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=6,
    lora_alpha = 8,
    lora_dropout = 0.05,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

BASE_MODEL = "google/gemma-2-9b-it"

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'


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

train_dataset, test_dataset = load_and_split_data(file_path, 'value', lower_bound)

dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})


def generate_prompt(example):
    prompt_list = []
    for i in range(len(example['instruction'])):
        prompt_list.append(r"""<bos><start_of_turn>user
아래의 두 메모의 메모내용을 읽고 파악하여 각 메모로부터 키워드를 찾고 이를 통해 새로운 개념이나 아이디어를 200글자 이하로 도출하세요.
추출한 키워드에 대한 설명없이 새로운 개념이나 아이디어만을 출력하세요.:
{}<end_of_turn>
<start_of_turn>model
{}<end_of_turn><eos>""".format(example['instruction'][i], example['output'][i]))
    return prompt_list

train_data = dataset['train']
test_data = dataset['test']

trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=test_data,
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir="outputs",
        num_train_epochs = 3,
        #max_steps=3000,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        warmup_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=100,
        push_to_hub=False,
        report_to='none',
    ),
    peft_config=lora_config,
    formatting_func=generate_prompt,
)

trainer.train()

ADAPTER_MODEL = "zettel_adapter"

trainer.save_model(ADAPTER_MODEL)