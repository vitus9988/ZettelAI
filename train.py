import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import DataCollatorForSeq2Seq,AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from sklearn.model_selection import train_test_split

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
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


file_path = "DATASET.csv"

train_dataset, test_dataset = load_and_split_data(file_path, 'value', 8.5)

dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})


def preprocess_function(examples):
    inputs = examples['instruction']
    labels = examples['output']
    
    model_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=2048)
    label_tokens = tokenizer(labels, padding="max_length", truncation=True, max_length=2048)
    
    model_inputs["labels"] = label_tokens["input_ids"]
    
    for i, label in enumerate(model_inputs["labels"]):
        model_inputs["labels"][i] = [
            token_id if token_id != tokenizer.pad_token_id else -100 for token_id in label
        ]
    
    return model_inputs

tokenized_datasets = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100
)

peft_params = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = SFTConfig(
        output_dir="./results",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=64,
        learning_rate=2e-4,
        weight_decay=0.01,
        num_train_epochs=3,
        max_steps=-1,
        lr_scheduler_type="cosine",
        warmup_steps=20,
        log_level="info",
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        packing=True,
        seed=42,
        max_seq_length=2048,
        optim="paged_adamw_8bit",
    )


trainer = SFTTrainer(
        model=model,
        peft_config=peft_params,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
    )

trainer.train()

model.save_pretrained("./zettelAI")
tokenizer.save_pretrained("./zettelAI")
