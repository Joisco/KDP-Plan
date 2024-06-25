import os
import logging
from datetime import datetime
from datasets import load_dataset, Dataset
from transformers import LlamaForCausalLM, PreTrainedTokenizerFast, Trainer, TrainingArguments
import math
import psutil
import torch
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

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

# Tokenize the dataset
def tokenize_function(tokenizer, examples):
    return tokenizer(examples["tokens"], padding="max_length", truncation=True, max_length=512)

# Compute metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(dim=-1)
    labels = labels.where(labels != -100, predictions)
    accuracy = (predictions == labels).float().mean()
    return {'accuracy': accuracy.item()}

def main():
    model_path = r"C:\AI Stuff\Joseph\Models\models--cognitivecomputations--dolphin-2.9-llama3-8b\snapshots\5aeb036f9215c558b483a654a8c6e1cc22e841bf"
    dataset_path = r"C:\AI Stuff\Joseph\Josephing\Official Script\Compiled Text\total_books.jsonl"
    output_dir = r"C:\AI Stuff\Joseph\Josephing\Official Script\FineTunedModels"

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

    # Tokenize the dataset using multithreading
    logging.info("Using CPU for tokenization")

    tokenized_data = []
    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        futures = [executor.submit(tokenize_function, tokenizer, example) for example in dataset]
        for future in concurrent.futures.as_completed(futures):
            try:
                tokenized_data.append(future.result())
            except Exception as e:
                logging.error(f"Error in tokenization: {e}")

    # Convert the list back to a Dataset object
    tokenized_datasets = Dataset.from_dict({'input_ids': [item['input_ids'] for item in tokenized_data],
                                            'attention_mask': [item['attention_mask'] for item in tokenized_data]})

    # Check the columns in the tokenized dataset
    logging.info(f"Tokenized dataset columns: {tokenized_datasets.column_names}")

    if 'input_ids' not in tokenized_datasets.column_names or 'attention_mask' not in tokenized_datasets.column_names:
        logging.error(f"Tokenization failed to produce expected columns. Columns present: {tokenized_datasets.column_names}")
        raise ValueError("Tokenization did not produce the expected columns.")

    logging.info("Dataset tokenization completed successfully.")

    # Set training arguments
    training_args = TrainingArguments(
        output_dir=generate_unique_output_dir(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,  # Lowered batch size
        gradient_accumulation_steps=4,  # Adjusted gradient accumulation steps
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        warmup_steps=500,
        save_steps=1000,
        save_total_limit=2,
        logging_dir='./logs',
        logging_steps=10,  # More frequent logging
        eval_strategy="steps",
        eval_steps=500,
        report_to="none",
        dataloader_num_workers=cpu_count(),  # Use all available CPU cores
        fp16=True,
        optim="adamw_torch",
    )

    # Create a Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        eval_dataset=tokenized_datasets,
        compute_metrics=compute_metrics
    )

    # Start training
    try:
        logging.info("Starting training process.")
        trainer.train()
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
