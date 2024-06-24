import os
import logging
from datetime import datetime
from datasets import load_dataset
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments
import math
import psutil
import torch
from tqdm import tqdm

# Configure logging
log_filename = 'finetuning.log'
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_filename, filemode='w')

# Add console handler to logging
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(console_handler)

# Add more logging for system resource usage
def log_system_resources():
    memory_info = psutil.virtual_memory()
    logging.info(f"Memory usage: {memory_info.percent}%")
    cpu_info = psutil.cpu_percent(interval=1)
    logging.info(f"CPU usage: {cpu_info}%")

# Define a function to generate a unique output directory
def generate_unique_output_dir(base_dir):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return os.path.join(base_dir, f"finetuned_model_{timestamp}")

# Load the dataset
def load_and_prepare_dataset(dataset_path):
    try:
        dataset = load_dataset('json', data_files=dataset_path, split='train[:10%]')
        logging.info(f"Dataset loaded successfully with {len(dataset)} examples")
        return dataset
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

# Compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(dim=-1)
    # Mask out the padding tokens
    labels = labels.where(labels != -100, predictions)
    accuracy = (predictions == labels).float().mean()
    return {'accuracy': accuracy.item()}

def main():
    model_path = r"C:\Users\camth\Models\models--cognitivecomputations--dolphin-2.9-llama3-8b\snapshots\5aeb036f9215c558b483a654a8c6e1cc22e841bf"
    dataset_path = r"C:\Users\camth\Joseph\Josephing\Official Script\Compiled Text\total_books.jsonl"
    output_dir = r"C:\Users\camth\Joseph\Josephing\Official Script\FineTunedModels"

    logging.info(f"Model path: {model_path}")
    logging.info(f"Dataset path: {dataset_path}")
    logging.info(f"Output dir: {output_dir}")

    # Verify paths are strings
    assert isinstance(model_path, str), f"model_path is not a string: {model_path}"
    assert isinstance(dataset_path, str), f"dataset_path is not a string: {dataset_path}"
    assert isinstance(output_dir, str), f"output_dir is not a string: {output_dir}"

    # Log system resources
    log_system_resources()

    # Ensure the model path exists and list its contents
    if os.path.exists(model_path):
        logging.info(f"Directory exists: {model_path}")
        logging.info(f"Contents of the directory: {os.listdir(model_path)}")
    else:
        logging.error(f"Model path does not exist: {model_path}")
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    try:
        # Log the type of model_path before using it
        logging.info(f"Type of model_path: {type(model_path)}")
        logging.info(f"Type of dataset_path: {type(dataset_path)}")
        logging.info(f"Type of output_dir: {type(output_dir)}")

        # Load the tokenizer and model
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path, legacy=False)
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

        # Ensure that special tokens are properly initialized
        model.resize_token_embeddings(len(tokenizer))

        # Ensure that special tokens are fine-tuned
        special_tokens = list(tokenizer.special_tokens_map.values())
        logging.info(f"Special tokens to be fine-tuned: {special_tokens}")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.sep_token_id = tokenizer.sep_token_id

    except Exception as e:
        logging.error(f"Error loading tokenizer or model: {e}")
        raise

    # Log system resources after loading model
    log_system_resources()

    # Load and prepare the dataset
    dataset = load_and_prepare_dataset(dataset_path)

    # Inspect dataset structure
    logging.info(f"Dataset features: {dataset.features}")

    # Tokenize the dataset with maximum resource allocation
    num_cpus = psutil.cpu_count(logical=True)
    logging.info(f"Using {num_cpus} CPUs for tokenization")

    # Identify the correct field name (assuming it is "tokens" here)
    def tokenize_function(examples):
        return tokenizer(examples["tokens"], is_split_into_words=True, padding="max_length", truncation=True, max_length=512)

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=num_cpus, remove_columns=["tokens"])

    # Check the columns in the tokenized dataset
    logging.info(f"Tokenized dataset columns: {tokenized_datasets.column_names}")

    # Ensure columns exist
    if 'input_ids' not in tokenized_datasets.column_names or 'attention_mask' not in tokenized_datasets.column_names:
        logging.error(f"Tokenization failed to produce expected columns. Columns present: {tokenized_datasets.column_names}")
        raise ValueError("Tokenization did not produce the expected columns.")

    logging.info("Dataset tokenization completed successfully.")

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=generate_unique_output_dir(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,  # Increased batch size for better utilization
        gradient_accumulation_steps=2,  # Adjusted gradient accumulation steps
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        save_steps=1000,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=50,  # More frequent logging
        eval_strategy="steps",
        eval_steps=500,
        report_to="none",
        dataloader_num_workers=num_cpus,  # Utilize all available CPU cores for data loading
        fp16=True,  # Enable mixed precision training if your GPU supports it
        optim="adamw_torch",
    )

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,  # Using the same dataset for both training and evaluation in this example
        compute_metrics=compute_metrics
    )

    # Start training
    try:
        logging.info("Starting training process.")
        total_steps = len(tokenized_datasets) // training_args.per_device_train_batch_size * training_args.num_train_epochs
        progress_bar = tqdm(total=total_steps, desc="Training Progress", unit="step")

        trainer.train()

        for step in range(total_steps):
            progress = (step + 1) / total_steps * 100
            progress_bar.update(1)
            if step % 100 == 0:
                logging.info(f"Step {step + 1}/{total_steps} completed. Progress: {progress:.2f}%")

        progress_bar.close()
        logging.info("Training completed successfully.")
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

    # Evaluate the model
    try:
        logging.info("Starting evaluation process.")
        eval_result = trainer.evaluate()
        perplexity = math.exp(eval_result["eval_loss"])
        eval_result["perplexity"] = perplexity
        logging.info(f"Evaluation results: {eval_result}")
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")

    # Save the model
    try:
        logging.info("Saving the fine-tuned model.")
        model_path = os.path.join(training_args.output_dir, "final_model")
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        logging.info(f"Model saved to {model_path}")
    except Exception as e:
        logging.error(f"Error saving the model: {e}")

if __name__ == "__main__":
    log_system_resources()
    main()
