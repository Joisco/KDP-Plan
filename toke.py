import os
import json
import logging
from nltk.tokenize import word_tokenize
from datetime import datetime

# Configure logging
log_filename = 'tokenization.log'
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=log_filename, filemode='w')

# Define a function to clean and tokenize text
def clean_and_tokenize(text):
    try:
        # Clean text (e.g., remove extra whitespace, lowercase, etc.)
        cleaned_text = text.strip().lower()
        
        # Tokenize the cleaned text
        tokens = word_tokenize(cleaned_text)
        
        return tokens
    except Exception as e:
        logging.error(f"Error during tokenization: {e}")
        return []

# Define a function to read a text file in chunks
def read_text_file_in_chunks(file_path, chunk_size=4096):  # Increased chunk size to 4096 bytes
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                yield chunk
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        raise

# Define a function to generate a unique filename
def generate_unique_filename(base_name, extension):
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    return f"{base_name}_{timestamp}.{extension}"

# Define the main function
def main(input_file, output_dir, batch_size=500):
    tokenized_batches = []

    # Generate a unique output filename
    output_file = os.path.join(output_dir, generate_unique_filename("total_books", "jsonl"))

    # Calculate total lines in the input file
    total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))
    logging.info(f"Total lines to process: {total_lines}")

    try:
        # Read the text from the input file in chunks
        text_chunks = read_text_file_in_chunks(input_file)
        processed_lines = 0

        # Process each chunk and accumulate tokenized results
        for i, chunk in enumerate(text_chunks):
            tokenized_chunk = clean_and_tokenize(chunk)
            tokenized_batches.append(tokenized_chunk)
            processed_lines += chunk.count('\n')

            # Write the batch to the output file if batch size is reached
            if len(tokenized_batches) >= batch_size:
                with open(output_file, 'a', encoding='utf-8') as out_file:
                    for tokens in tokenized_batches:
                        json.dump({"tokens": tokens}, out_file)
                        out_file.write('\n')
                logging.info(f"Processed batch {i + 1} - Lines processed: {processed_lines}/{total_lines}")
                tokenized_batches = []

        # Write any remaining tokens to the output file
        if tokenized_batches:
            with open(output_file, 'a', encoding='utf-8') as out_file:
                for tokens in tokenized_batches:
                    json.dump({"tokens": tokens}, out_file)
                    out_file.write('\n')
            logging.info(f"Processed final batch - Lines processed: {processed_lines}/{total_lines}")
        
        logging.info(f"Tokenized text saved to {output_file}")
    except Exception as e:
        logging.error(f"Error in main function: {e}")

if __name__ == "__main__":
    input_file = r"C:\Users\camth\Joseph\Josephing\Official Script\Compiled Text\compiled_books_2.txt"  # Define your input file
    output_dir = r"C:\Users\camth\Joseph\Josephing\Official Script\Compiled Text"  # Define your output directory

    main(input_file, output_dir)

