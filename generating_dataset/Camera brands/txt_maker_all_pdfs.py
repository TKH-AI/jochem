import os
import re
from langchain_community.document_loaders import PyMuPDFLoader

# Define the base directories
base_input_dir = "/Users/jochem/Desktop/mijn_project2/Camera brands/datasheet_pdfs"
base_output_dir = "/Users/jochem/Desktop/mijn_project2/Camera brands/datasheet_txts_cleaned_up"

# List of camera brand folders to process
camera_brands = [
    "ADIMEC", 
    "Allied_Vision", 
    "Basler", 
    "Baumer", 
    "Cognex", 
    "FLIR", 
    "Framos", 
    "Hamamatsu", 
    "IDS_Imaging", 
    "Imperx",
    "Keyence", 
    "OPTO", 
    "Teledyne_Dalsa"
]

# def clean_text(text):
#     """
#     Function to clean the extracted text.
#     Add any text cleaning steps here if needed.
#     """
#     # Remove multiple spaces, tabs, new lines
#     text = re.sub(r'\s+', ' ', text)
#     return text

def apply_brand_specific_processing(text, brand):
    """
    Apply brand-specific text processing rules.
    """
    if brand == "ADIMEC":
        pattern = r'Mechanical outline.*?(Sensor Mounting Accuracy)'
        text = re.sub(pattern, r'\1', text, flags=re.DOTALL)

    elif brand == "Allied_Vision":
        pattern = r'Technical drawing.*'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

    elif brand == "Baumer":
        pattern = r'GenICam™ Features.*'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

        pattern = r'Partial Scan @ FullFrame.*'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

        pattern = r'@ FullFrame, min Exposure.*'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

        pattern = r'ROI Frame Rates, min Exposure.*'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

        '@ FullFrame, min Exposure,'

    elif brand == "Cognex":        
        pattern = r'Field of view diagrams.*'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

        pattern = r'Corporate Headquarters.*'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

    elif brand == "FLIR":
        pattern = r'Shipping information.*'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

        pattern = r'Supplies & accessories.*'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

        pattern = r'Supplies and accessories.*'
        text = re.sub(pattern, '', text, flags=re.DOTALL)
        
    elif brand == "Framos":
        pattern = r'Table of Contents[\s\S]*?(www.framos.com)'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

        pattern = r'Pinouts[\s\S]*?(6 Platform and Software Specification  )'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

        pattern = r'Pinouts[\s\S]*?(6 Platform & Software Specification  )'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

        pattern = r'D400E SERIES MECHANICAL DRAWINGS.*'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

    elif brand == "Hamamatsu":
        pattern = r'Comparison of image quality at different pixel sizes[\s\S]*?(SPECIALIZED FOR THE SPECIALIST)'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

        pattern = r'Ultra-low readout noise 0.30 electrons rms[\s\S]*?(Application and Measurement Examples)'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

    elif brand == "Teledyne_Dalsa":

        pattern = r'Response & QE curves[\s\S]*?(Input/output Connectors and LED)'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

        pattern = r'RESPONSIVITY GRAPHS.*'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

        pattern = r'Standard Confomity [\s\S]*?(Models)'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

        pattern = r'Camera Interface : NBASE-TTM [\s\S]*?(Models )'
        text = re.sub(pattern, '', text, flags=re.DOTALL)

    return text

def convert_pdf_to_txt(input_pdf_path, output_txt_path, brand):
    """
    Function to convert a single PDF to TXT, applying brand-specific processing.
    """
    try:
        loader = PyMuPDFLoader(input_pdf_path)
        documents = loader.load()
        # Assuming each document contains one or more pages of text
        text = "\n".join([doc.page_content for doc in documents])
        # Clean up text
        # cleaned_text = clean_text(text)
        # Apply brand-specific processing
        processed_text = apply_brand_specific_processing(text, brand)
        # Write to output file
        with open(output_txt_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(processed_text)
        print(f"Converted and saved: {output_txt_path}")
    except Exception as e:
        print(f"Failed to convert {input_pdf_path}: {e}")

def process_directory(input_dir, output_dir, brand):
    """
    Process all PDFs in the given directory, including subdirectories.
    """
    for root, _, files in os.walk(input_dir):
        # Determine the relative path from the base input directory
        relative_path = os.path.relpath(root, input_dir)
        # Determine the corresponding output directory path
        target_output_dir = os.path.join(output_dir, relative_path)
        # Ensure the output directory exists
        os.makedirs(target_output_dir, exist_ok=True)
        # Process each file in the current directory
        for filename in files:
            if filename.lower().endswith('.pdf'):
                input_pdf_path = os.path.join(root, filename)
                output_txt_filename = f"{os.path.splitext(filename)[0]}.txt"
                output_txt_path = os.path.join(target_output_dir, output_txt_filename)
                convert_pdf_to_txt(input_pdf_path, output_txt_path, brand)

def process_camera_brands(camera_brands, base_input_dir, base_output_dir):
    """
    Process each brand's PDF files, handling subdirectories.
    """
    for brand in camera_brands:
        input_dir = os.path.join(base_input_dir, brand)
        output_dir = os.path.join(base_output_dir, brand)
        process_directory(input_dir, output_dir, brand)

# Run the process
process_camera_brands(camera_brands, base_input_dir, base_output_dir)
