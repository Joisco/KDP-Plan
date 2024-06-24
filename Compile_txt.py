import os
import hashlib

# Define the directory containing text files and the output file
text_directory = r"C:\Users\camth\Joseph\Josephing\RoyalRoad"  # Update this path if needed
output_file_path = r'C:\Users\camth\Joseph\Josephing\Scripts\Tokenized\compiled_books.txt'  # Update this path if needed

# Create the output directory if it doesn't exist
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

def calculate_hash(file_path):
    """Calculate the MD5 hash of the given file."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_unique_filename(filepath):
    """Generate a unique filename by adding a number suffix if the file already exists."""
    base, ext = os.path.splitext(filepath)
    counter = 1
    new_filepath = filepath
    while os.path.exists(new_filepath):
        new_filepath = f"{base}_{counter}{ext}"
        counter += 1
    return new_filepath

def compile_texts_with_delimiters(directory, output_file, delimiter="\n"):
    seen_hashes = set()
    book_titles = []
    output_file = get_unique_filename(output_file)  # Ensure unique output filename

    try:
        with open(output_file, 'a', encoding='utf-8') as outfile:  # Append mode to add new content
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        file_hash = calculate_hash(file_path)
                        if file_hash not in seen_hashes:
                            seen_hashes.add(file_hash)
                            book_title = os.path.splitext(file)[0]  # Extract the book title from the filename
                            book_titles.append(book_title)
                            with open(file_path, 'r', encoding='utf-8') as infile:
                                outfile.write(f"{delimiter}## {book_title} ##{delimiter}")  # Add book title as a header
                                outfile.write(infile.read() + delimiter)
                                print(f'Added {file_path} to {output_file}')
                        else:
                            print(f'Skipped duplicate {file_path}')
    except Exception as e:
        print(f"Error compiling text files: {e}")

    return book_titles

def create_index(book_titles, output_file, delimiter="\n"):
    index = f"{delimiter}Index of Compiled Books{delimiter}"
    index += f"Total Books: {len(book_titles)}{delimiter}"
    for i, title in enumerate(book_titles, start=1):
        index += f"{i}. {title}{delimiter}"
    
    try:
        with open(output_file, 'r+', encoding='utf-8') as outfile:
            content = outfile.read()
            outfile.seek(0, 0)  # Move the cursor to the beginning of the file
            outfile.write(index + delimiter + content)
    except Exception as e:
        print(f"Error creating index: {e}")

# Compile the text files with delimiters and duplicate check
book_titles = compile_texts_with_delimiters(text_directory, output_file_path)

# Create the index
create_index(book_titles, output_file_path)
