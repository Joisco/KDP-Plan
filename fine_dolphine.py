import os
import torch
import logging
import psutil
import numpy as np
from datetime import datetime, timedelta
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_scheduler,
    EarlyStoppingCallback,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.nn.parallel import DataParallel
from transformers import AutoConfig
import optuna

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_system_resources():
    memory_info = psutil.virtual_memory()
    logging.info(f"Memory usage: {memory_info.percent}%")
    cpu_info = psutil.cpu_percent(interval=1)
    logging.info(f"CPU usage: {cpu_info}%")

def generate_unique_output_dir(base_dir):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(base_dir, f"finetuned_model_{timestamp}")

def load_and_prepare_dataset(dataset_path, tokenizer, cache_dir=None, subset_size=0.1):
    try:
        dataset = load_dataset("text", data_files=dataset_path, split="train", cache_dir=cache_dir)
        logging.info(f"Dataset loaded successfully with {len(dataset)} examples")

        # Take a subset of the dataset for hyperparameter optimization
        dataset = dataset.select(range(int(len(dataset) * subset_size)))

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

        cache_file_name = os.path.join(cache_dir, "tokenized_dataset.arrow") if cache_dir else None
        if cache_file_name and os.path.exists(cache_file_name):
            tokenized_dataset = Dataset.load_from_disk(cache_file_name)
        else:
            tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
            if cache_file_name:
                tokenized_dataset.save_to_disk(cache_file_name)

        logging.info("Dataset tokenized and prepared")
        logging.info(f"Dataset size after preparation: {len(tokenized_dataset)}")
        logging.info(f"Sample prepared example: {tokenized_dataset[0]}")
        return tokenized_dataset
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = np.mean(predictions == labels)
    perplexity = np.exp(-np.mean(logits[np.arange(len(logits)), labels]))
    return {"accuracy": accuracy, "perplexity": perplexity}

class CurriculumTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_difficulty = 0
        self.max_difficulty = 5

    def get_train_dataloader(self):
        logging.info(f"Current difficulty: {self.current_difficulty}")
        logging.info(f"Train dataset size: {len(self.train_dataset)}")
        dataset = self.train_dataset.filter(lambda example: len(example['input_ids']) <= (self.current_difficulty + 1) * 1000)
        if len(dataset) == 0:
            logging.warning("Filtered dataset is empty. Using full dataset for this difficulty level.")
            dataset = self.train_dataset
        logging.info(f"Filtered dataset size: {len(dataset)}")
        return DataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True
        )

    def train(self, *args, **kwargs):
        for epoch in range(int(self.args.num_train_epochs)):
            if epoch > 0 and epoch % 2 == 0:
                self.current_difficulty = min(self.current_difficulty + 0.5, self.max_difficulty)
            super().train(*args, **kwargs)

def objective(trial):
    try:
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
        num_train_epochs = trial.suggest_int("num_train_epochs", 1, 3)
        gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 8)

        model_path = r"C:\AI Stuff\Joseph\Models\models--cognitivecomputations--dolphin-2.9-llama3-8b\snapshots\5aeb036f9215c558b483a654a8c6e1cc22e841bf"
        dataset_path = r"C:\AI Stuff\Joseph\Josephing\Official Script\Compiled Text\compiled_books_1.txt"
        output_dir = r"C:\AI Stuff\Joseph\Josephing\Official Script\FineTunedModels"
        cache_dir = r"C:\AI Stuff\Joseph\Josephing\Cache"

        for path in [model_path, dataset_path, output_dir, cache_dir]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path does not exist: {path}")

        logging.info(f"Model path: {model_path}")
        logging.info(f"Dataset path: {dataset_path}")
        logging.info(f"Output dir: {output_dir}")

        log_system_resources()

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        config.dropout = 0.1
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config)

        if torch.cuda.is_available():
            model = DataParallel(model)
            model.to(torch.device("cuda"))
            logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            model.to(torch.device("cpu"))
            logging.info("CUDA is not available. Using CPU.")

        model.gradient_checkpointing_enable()

        dataset = load_and_prepare_dataset(dataset_path, tokenizer, cache_dir, subset_size=0.1)
        train_dataset, validation_dataset = dataset.train_test_split(test_size=0.1).values()

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        training_args = TrainingArguments(
            output_dir=generate_unique_output_dir(output_dir),
            overwrite_output_dir=True,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            lr_scheduler_type="cosine_with_restarts",
            warmup_steps=500,
            logging_dir="./logs",
            logging_steps=10,
            eval_steps=500,
            evaluation_strategy="steps",
            save_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            fp16=True,
        )

        optimizer = AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
        lr_scheduler = get_scheduler(
            "cosine_with_restarts",
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.num_train_epochs * len(train_dataset) // training_args.per_device_train_batch_size,
        )

        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

        trainer = CurriculumTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[early_stopping_callback],
            optimizers=(optimizer, lr_scheduler),
        )

        logging.info("Starting training process.")
        start_time = datetime.now()
        trainer.train()
        elapsed_time = datetime.now() - start_time
        logging.info(f"Step 'training' completed in {elapsed_time}.")

        eval_result = trainer.evaluate()
        logging.info(f"Evaluation result: {eval_result}")

        return eval_result["eval_loss"]
    except ValueError as ve:
        logging.error(f"ValueError in Optuna trial: {ve}")
        return float('inf')
    except Exception as e:
        logging.error(f"Unexpected error in Optuna trial: {e}")
        raise optuna.exceptions.TrialPruned()

def print_process_overview():
    steps = [
        "1. Check system resources.",
        "2. Check dataset compatibility.",
        "3. Load and prepare dataset.",
        "4. Tokenize dataset.",
        "5. Optimize hyperparameters using Optuna.",
        "6. Train the model using best hyperparameters.",
        "7. Evaluate the model.",
        "8. Save the model for future use."
    ]
    for step in steps:
        logging.info(step)

def check_dataset_compatibility(dataset_path):
    try:
        with open(dataset_path, 'r', encoding='utf-8') as file:
            first_line = file.readline()
        if not first_line:
            raise ValueError("Dataset is empty or not properly formatted.")
        logging.info("Dataset compatibility check passed.")
    except Exception as e:
        logging.error(f"Dataset compatibility check failed: {e}")
        raise

def analyze_dataset(dataset):
    lengths = [len(example['input_ids']) for example in dataset]
    logging.info(f"Min length: {min(lengths)}, Max length: {max(lengths)}, Avg length: {sum(lengths)/len(lengths)}")
    logging.info(f"Length distribution: {np.percentile(lengths, [25, 50, 75, 90, 95, 99])}")

def estimate_trial_duration(model, training_args, train_dataset, data_collator):
    sample_size = 1000
    sample_dataset = train_dataset.select(range(min(sample_size, len(train_dataset))))
    data_loader = DataLoader(sample_dataset, batch_size=training_args.per_device_train_batch_size, collate_fn=data_collator)

    start_time = datetime.now()
    for batch in data_loader:
        outputs = model(**batch)
    elapsed_time = (datetime.now() - start_time).total_seconds()
    return (elapsed_time / sample_size) * len(train_dataset)

if __name__ == "__main__":
    log_system_resources()
    print_process_overview()
    check_dataset_compatibility(r"C:\AI Stuff\Joseph\Josephing\Official Script\Compiled Text\compiled_books_1.txt")

    try:
        study = optuna.create_study(direction="minimize")

        model_path = r"C:\AI Stuff\Joseph\Models\models--cognitivecomputations--dolphin-2.9-llama3-8b\snapshots\5aeb036f9215c558b483a654a8c6e1cc22e841bf"
        dataset_path = r"C:\AI Stuff\Joseph\Josephing\Official Script\Compiled Text\compiled_books_1.txt"
        cache_dir = r"C:\AI Stuff\Joseph\Josephing\Cache"

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        config.dropout = 0.1
        model = AutoModelForCausalLM.from_pretrained(model_path, config=config)

        if torch.cuda.is_available():
            model = DataParallel(model)
            model.to(torch.device("cuda"))
            logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            model.to(torch.device("cpu"))
            logging.info("CUDA is not available. Using CPU.")

        model.gradient_checkpointing_enable()
        dataset = load_and_prepare_dataset(dataset_path, tokenizer, cache_dir, subset_size=0.1)
        train_dataset, validation_dataset = dataset.train_test_split(test_size=0.1).values()
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        estimated_time = estimate_trial_duration(model, TrainingArguments, train_dataset, data_collator)
        estimated_duration = str(timedelta(seconds=estimated_time))
        logging.info(f"Estimated duration for each trial: {estimated_duration}")

        study.optimize(objective, n_trials=20, timeout=3600 * 8)

        logging.info(f"Best trial: {study.best_trial.value}")
        logging.info(f"Best parameters: {study.best_trial.params}")

        best_params = study.best_trial.params

        final_model_path = r"C:\AI Stuff\Joseph\Models\models--cognitivecomputations--dolphin-2.9-llama3-8b\snapshots\5aeb036f9215c558b483a654a8c6e1cc22e841bf"
        final_dataset_path = r"C:\AI Stuff\Joseph\Josephing\Official Script\Compiled Text\compiled_books_1.txt"
        final_output_dir = r"C:\AI Stuff\Joseph\Josephing\Official Script\FineTunedModels"

        tokenizer = AutoTokenizer.from_pretrained(final_model_path)
        config = AutoConfig.from_pretrained(final_model_path)
        config.dropout = 0.1
        final_model = AutoModelForCausalLM.from_pretrained(final_model_path, config=config)

        if torch.cuda.is_available():
            final_model = DataParallel(final_model)
            final_model.to(torch.device("cuda"))
            logging.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            final_model.to(torch.device("cpu"))
            logging.info("CUDA is not available. Using CPU.")

        final_model.gradient_checkpointing_enable()

        final_dataset = load_and_prepare_dataset(final_dataset_path, tokenizer, cache_dir)
        final_train_dataset, final_validation_dataset = final_dataset.train_test_split(test_size=0.1).values()

        final_data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        final_training_args = TrainingArguments(
            output_dir=generate_unique_output_dir(final_output_dir),
            overwrite_output_dir=True,
            num_train_epochs=best_params["num_train_epochs"],
            per_device_train_batch_size=best_params["batch_size"],
            per_device_eval_batch_size=best_params["batch_size"],
            gradient_accumulation_steps=best_params["gradient_accumulation_steps"],
            learning_rate=best_params["learning_rate"],
            weight_decay=0.01,
            lr_scheduler_type="cosine_with_restarts",
            warmup_steps=500,
            logging_dir="./logs",
            logging_steps=10,
            eval_steps=500,
            evaluation_strategy="steps",
            save_steps=1000,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            fp16=True,
        )

        final_optimizer = AdamW(final_model.parameters(), lr=final_training_args.learning_rate, weight_decay=final_training_args.weight_decay)
        final_lr_scheduler = get_scheduler(
            "cosine_with_restarts",
            optimizer=final_optimizer,
            num_warmup_steps=final_training_args.warmup_steps,
            num_training_steps=final_training_args.num_train_epochs * len(final_train_dataset) // final_training_args.per_device_train_batch_size,
        )

        final_early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

        final_trainer = CurriculumTrainer(
            model=final_model,
            args=final_training_args,
            train_dataset=final_train_dataset,
            eval_dataset=final_validation_dataset,
            data_collator=final_data_collator,
            compute_metrics=compute_metrics,
            callbacks=[final_early_stopping_callback],
            optimizers=(final_optimizer, final_lr_scheduler),
        )

        logging.info("Starting final training process.")
        start_time = datetime.now()
        final_trainer.train()
        elapsed_time = datetime.now() - start_time
        logging.info(f"Step 'final training' completed in {elapsed_time}.")

        logging.info("Saving the fine-tuned model.")
        final_model_save_path = os.path.join(final_training_args.output_dir, "final_model")
        final_model.save_pretrained(final_model_save_path)
        tokenizer.save_pretrained(final_model_save_path)
        logging.info(f"Model saved to {final_model_save_path}")

    except Exception as e:
        logging.error(f"Error during script execution: {e}")
        raise
