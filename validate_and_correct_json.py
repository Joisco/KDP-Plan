import json
import tqdm
import re

input_file_path = r"C:\Users\camth\Joseph\Josephing\Official Script\Compiled Text\total_books.jsonl"
output_file_path = r"C:\Users\camth\Joseph\Josephing\Official Script\Compiled Text\corrected_total_books.jsonl"

def correct_json_line(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError as e:
        print(f"Error in line: {e}")
        error_message = str(e)

        # Attempt to fix issues with concatenated JSON objects or missing commas
        corrected_line = line
        if "Expecting ',' delimiter" in error_message or "Expecting '}'" in error_message:
            corrected_line = corrected_line.replace("}{", "},{")
            corrected_line = re.sub(r'(\w)"(\w)', r'\1", "\2', corrected_line)  # Ensure proper comma separation
            corrected_line = re.sub(r'(\w)(\{)', r'\1, \2', corrected_line)     # Ensure comma before object start
            corrected_line = re.sub(r'(\})(\w)', r'\1, \2', corrected_line)     # Ensure comma after object end

        # Attempt to fix issues with extra data
        if "Extra data" in error_message:
            parts = re.split(r'(?<=\})(?=\{)', corrected_line)
            corrected_line = ','.join(parts)
            corrected_line = f'[{corrected_line}]'

        # Attempt to fix issues with missing quotes
        if "Unterminated string" in error_message or "Expecting property name enclosed in double quotes" in error_message:
            corrected_line = re.sub(r'(?<!\\)"', r'\"', corrected_line)  # Add escape characters for unescaped quotes
            corrected_line = re.sub(r'([^\\])\\"', r'\1"', corrected_line)  # Fix incorrectly escaped quotes

        # Remove non-ASCII characters
        corrected_line = corrected_line.encode("ascii", "ignore").decode()

        # Additional custom error corrections can be added here

        try:
            return json.loads(corrected_line)
        except json.JSONDecodeError as e:
            print(f"Further error: {e}")
            return None

def validate_and_correct_json_lines(input_file_path, output_file_path):
    total_lines = 0
    failed_lines = 0

    with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
        total_lines = sum(1 for _ in infile)
        infile.seek(0)  # Reset file pointer to beginning

        for line_num, line in enumerate(tqdm.tqdm(infile, total=total_lines, desc="Processing lines"), 1):
            corrected_json = correct_json_line(line)
            if corrected_json:
                if isinstance(corrected_json, list):
                    for obj in corrected_json:
                        outfile.write(json.dumps(obj) + '\n')
                else:
                    outfile.write(json.dumps(corrected_json) + '\n')
            else:
                print(f"Line {line_num} could not be corrected. Line content: {line.strip()}")
                failed_lines += 1

    print(f"Total lines processed: {total_lines}")
    print(f"Total lines failed: {failed_lines}")
    print(f"Percentage of lines failed: {failed_lines / total_lines * 100:.2f}%")

validate_and_correct_json_lines(input_file_path, output_file_path)
print("Validation and correction complete.")
