from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq

model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSeq2SeqLM.from_pretrained(model_name,local_files_only=True)

dataset = load_dataset("json", data_files="data/train.json")

def format_and_tokenize(example):    
    prompt = f"Instruction: {example['instruction']}\nInput: {example['input']}\nResponse:"
    inputs = tokenizer(prompt, max_length=512, padding="max_length", truncation=True)
    targets = tokenizer(example['output'], max_length=128, padding="max_length", truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs

tokenized = dataset["train"].map(format_and_tokenize, remove_columns=dataset["train"].column_names)

args = TrainingArguments(
    output_dir="./flan-tuned",
    per_device_train_batch_size=2,
    num_train_epochs=30,
    logging_steps=10,
    save_strategy="no",
    report_to="none",
    fp16=False,
    label_smoothing_factor=0.0,
)

collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
    data_collator=collator,
)

trainer.train()
model.save_pretrained("./flan-tuned")
tokenizer.save_pretrained("./flan-tuned")
