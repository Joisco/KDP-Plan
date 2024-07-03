import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig

def main():
    # Set up model and tokenizer
    model_path = "C:\\Users\\Hadlock\\Downloads\\AI Stuff\\Joseph\\Models\\models--cognitivecomputations--dolphin-2.9-llama3-8b\\snapshots\\5aeb036f9215c558b483a654a8c6e1cc22e841bf"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load the model with 8-bit quantization and CPU offloading
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=True,
        device_map="auto",
    )

    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Load and preprocess the dataset
    dataset_path = "C:\\AI Stuff\\Joseph\\Josephing\\Official Script\\Compiled Text\\compiled_books.txt"
    with open(dataset_path, "r", encoding="utf-8") as file:
        data = file.read().splitlines()

    dataset = Dataset.from_dict({"text": data})

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="C:\\AI Stuff\\Joseph\\Josephing\\Official Script\\FineTunedModels",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=True,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
    )

    # Set up trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Start training
    trainer.train()

    # Save the fine-tuned model
    output_dir = "C:\\AI Stuff\\Joseph\\Josephing\\Official Script\\FineTunedModels"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
