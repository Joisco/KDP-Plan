import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import signal
import sys
import json
from datetime import datetime
import asyncio
import onnx
from onnxruntime.transformers import optimizer

# Custom JSONL Logger
class JSONLLogger(logging.Handler):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def emit(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.msg,
            "details": record.args
        }
        with open(self.filename, 'a') as f:
            json.dump(log_entry, f)
            f.write('\n')

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
jsonl_handler = JSONLLogger("log.jsonl")
logger.addHandler(jsonl_handler)

# Graceful shutdown
def signal_handler(sig, frame):
    logger.info("Interrupt received. Shutting down...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def load_model_to_device(model, device, use_amp):
    try:
        logger.info("Loading model to device...")
        if use_amp:
            model = model.half()
        model = model.to(device)
        logger.info("Model loaded successfully.")
    except RuntimeError as e:
        logger.error(f"CUDA out of memory: {e}. Trying to reduce memory usage...")
        if "CUDA out of memory" in str(e):
            torch.cuda.empty_cache()
            model = model.half().to(device)
            logger.info("Loaded model with half precision.")
        else:
            raise e
    return model

async def generate_response(model, tokenizer, input_ids, device, **gen_kwargs):
    input_ids = input_ids.to(device)
    logger.info("Generating response...")
    response_ids = await asyncio.to_thread(model.generate, input_ids, **gen_kwargs)
    return response_ids

def export_model_to_onnx(model, tokenizer, output_path):
    dummy_input = tokenizer.encode("This is a dummy input for ONNX export", return_tensors='pt').to(model.device)
    torch.onnx.export(model, dummy_input, output_path, export_params=True, opset_version=14, do_constant_folding=True, input_names=['input'], output_names=['output'])
    logger.info(f"Model exported to ONNX format at {output_path}")

def main():
    # Configuration settings
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "model_name": r"C:\Users\Hadlock\Downloads\AI Stuff\Joseph\Models\models--cognitivecomputations--dolphin-2.9-llama3-8b\snapshots\5aeb036f9215c558b483a654a8c6e1cc22e841bf",
        "use_amp": True,
        "max_length": 5000,  # Increase max length to prevent premature truncation
        "temperature": 0.8,
        "top_p": 0.9,
        "onnx_model_path": r"C:\Users\Hadlock\Downloads\AI Stuff\Joseph\Josephing\Official Script\V.B\model.onnx"  # Update this path to where you want to save the ONNX model
    }

    device = torch.device(config["device"])
    logger.info(f"Using device: {device}")

    model_name = config["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = load_model_to_device(model, device, config["use_amp"])

    # Export model to ONNX
    export_model_to_onnx(model, tokenizer, config["onnx_model_path"])

    history = []
    gen_kwargs = {
        "max_length": config["max_length"],
        "temperature": config["temperature"],
        "top_p": config["top_p"],
        "num_return_sequences": 1,
        "pad_token_id": tokenizer.eos_token_id,  # Ensure proper padding
        "do_sample": True  # Enable sampling
    }

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break

            history.append(f"You: {user_input}")
            input_ids = tokenizer.encode(user_input, return_tensors='pt')
            response_ids = asyncio.run(generate_response(model, tokenizer, input_ids, device, **gen_kwargs))
            response = tokenizer.decode(response_ids[0], skip_special_tokens=True)
            history.append(f"ChatGPT: {response}")

            print(f"ChatGPT: {response}")

        except Exception as e:
            logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
