import os
from bs4 import BeautifulSoup

# Define the base input and output directories
base_input_dir = '/Users/jochem/Desktop/HTML files Basler/Basler'
base_output_dir = '/Users/jochem/Desktop/HTML files Basler/Basler_output_txt_cleaned'

# List of subdirectories to process
subdirectories = [
    'Pulse', 
    {
        'Ace': ['ace Classic', 'ace L', 'ace U']
    }, 
    {
        'Ace 2': ['ace 2 R', 'ace 2 V', 'ace 2 X']
    },
    'Blaze',
    { 
    'Boost': ['boost R', 'boost V']
    },
    {
    'Dart': ['dart Classic', 'dart M', 'dart R']
    },
    'Med ace'
]

# Ensure the base output directory exists
os.makedirs(base_output_dir, exist_ok=True)

def process_files(input_dir, output_dir):
    # Ensure the output subdirectory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate over all HTML files in the current subdirectory
    for filename in os.listdir(input_dir):
        if filename.endswith('.html'):
            # Construct full file path for input and output
            input_filepath = os.path.join(input_dir, filename)
            output_filepath = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
            
            # Load your HTML file
            with open(input_filepath, 'r', encoding='utf-8') as file:
                html_content = file.read()
            
            # Parse the HTML content
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Get plain text by stripping all tags
            plain_text = soup.get_text()
            
            # Save the plain text to a new file
            with open(output_filepath, 'w', encoding='utf-8') as file:
                file.write(plain_text)
            
            print(f"HTML tags removed and plain text saved to {output_filepath}")
            
            # Process the text file to remove unwanted content before the model name
            model_name = os.path.splitext(filename)[0]
            with open(output_filepath, 'r', encoding='utf-8') as file:
                text_content = file.read()
            
            # Convert both text content and marker to lowercase for case-insensitive comparison
            lower_text_content = text_content.lower()
            lower_marker = f"{model_name.lower()}#"
            marker_pos = lower_text_content.find(lower_marker)
            if marker_pos != -1:
                # Remove everything before and including the marker
                text_content = text_content[marker_pos + len(lower_marker):]
            
            # Further process the text content to remove "Precautions#" and everything after it
            precautions_marker = "Precautions#"
            precautions_pos = text_content.find(precautions_marker)
            if precautions_pos != -1:
                # Remove everything from "Precautions#" to the end of the text
                text_content = text_content[:precautions_pos]
            
            # Save the cleaned content back to the file
            with open(output_filepath, 'w', encoding='utf-8') as file:
                file.write(text_content)
            
            print(f"Cleaned text saved to {output_filepath}")

# Iterate over each subdirectory
for subdir in subdirectories:
    if isinstance(subdir, dict):
        # Handle nested subdirectories
        for main_dir, nested_dirs in subdir.items():
            main_input_dir = os.path.join(base_input_dir, main_dir)
            main_output_dir = os.path.join(base_output_dir, main_dir)
            process_files(main_input_dir, main_output_dir)
            for nested_dir in nested_dirs:
                input_dir = os.path.join(main_input_dir, nested_dir)
                output_dir = os.path.join(main_output_dir, nested_dir)
                process_files(input_dir, output_dir)
    else:
        input_dir = os.path.join(base_input_dir, subdir)
        output_dir = os.path.join(base_output_dir, subdir)
        process_files(input_dir, output_dir)

print("All files processed.")