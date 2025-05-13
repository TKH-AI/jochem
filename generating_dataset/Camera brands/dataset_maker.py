import os
import pandas as pd
import re
from unidecode import unidecode

# Define the base output directory and the folders to process
base_output_dir = "/Users/jochem/Desktop/mijn_project2/Camera brands/datasheet_txts_cleaned_up"
folders = [
    "Teledyne_Dalsa",
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
    "OPTO"
]

# Initialize an empty list to collect the data
data = []

# Parsing the model names in the right way
def get_camera_model(folder, filename, content, root):
    if folder == "Teledyne_Dalsa":
        # Replace _ or + by a space
        camera_model = filename.replace('_', ' ').replace('+', ' ').replace('-', ' ')
        # Remove everything after and including "Datasheet", "datasheet", or "dsheet"
        camera_model = re.split(r'\s*(Datasheet|datasheet|dsheet)', camera_model)[0].strip()
        camera_model = camera_model.replace(".txt", '')
    elif folder == "ADIMEC":
        # Replace - with space
        camera_model = filename.replace('-', ' ')
        camera_model = camera_model.replace('Datasheet', '').strip()
        camera_model = camera_model.replace('PSD', '').strip()
        # Transform the first character if it is Q, D, or S
        if camera_model.startswith('Q'):
            camera_model = 'Quartz' + camera_model[1:]
        elif camera_model.startswith('D'):
            camera_model = 'Diamond' + camera_model[1:]
        elif camera_model.startswith('S'):
            camera_model = 'Sapphire' + camera_model[1:]
        # Keep only the first two words
        camera_model = ' '.join(camera_model.split()[:2])
    elif folder == "Allied_Vision":
        camera_model = filename.replace('_', ' ')
        camera_model = re.split(r'\s*(DataSheet)', camera_model)[0].strip()
    elif folder == "Basler":
        # Custom logic for Basler
        camera_model = filename.replace('-', ' ').replace('.txt', '')
        subfolder_name = os.path.basename(root)
        if not camera_model.startswith(subfolder_name):
            camera_model = f"{subfolder_name} {camera_model}"
        print(camera_model)
    elif folder == "Baumer":
        # Custom logic for Baumer
        camera_model = os.path.splitext(filename)[0]
        # Remove first one or two underscores
        camera_model = camera_model.replace('Baumer_', '').replace('TDS_', '')
        camera_model = re.sub(r'^_{1,2}', '', camera_model)
        # Remove everything after "EN"
        camera_model = re.split(r'_EN', camera_model)[0].strip()
    elif folder == "Cognex":
        # Custom logic for Cognex
        camera_model = filename.replace('_', ' ').replace('-', ' ')
        camera_model = camera_model.replace("Datasheet", "").strip()
        # Replace "DM" with "Dataman" and "IS" with "In-Sight"
        if camera_model.startswith('DM'):
            camera_model = camera_model.replace('DM', 'Dataman ')
        elif camera_model.startswith('IS'):
            camera_model = camera_model.replace('IS', 'In-Sight ')
        elif camera_model.startswith('AD'):
            camera_model = camera_model.replace('AD', 'Advantage ')
        camera_model = camera_model.replace('.txt', '')
        camera_model = ' '.join(camera_model.split()[:2])
    elif folder == "FLIR":
        camera_model = content.split('\n', 1)[0].strip()
        camera_model = camera_model.replace('FLIR', '').strip()
    elif folder == "Framos":
        # Custom logic for Framos
        camera_model = filename.replace('_', ' ').replace('+', ' ').replace('-', ' ')
        camera_model = ' '.join(camera_model.split()[:2])
    elif folder == "Hamamatsu":
        # Custom logic for Hamamatsu
        camera_model = filename.replace('_', ' ').replace('-', ' ').replace('.txt', '')
    elif folder == "IDS_Imaging":
        # Custom logic for IDS
        camera_model = os.path.splitext(filename)[0]
        # Ensure the camera model starts with the name of the sub-folder and its parent folder
        subfolder_name = os.path.basename(root)
        parent_folder_name = os.path.basename(os.path.dirname(root))
        combined_folder_name = f"{folder} {parent_folder_name} {subfolder_name}"
        if not camera_model.startswith(combined_folder_name):
            camera_model = f"{combined_folder_name} {camera_model}"
        camera_model = camera_model.replace('IDS_Imaging', '').replace('-', ' ').strip()
    elif folder == "Imperx":
        # Custom logic for Imperx
        camera_model = filename.replace('_', ' ').replace('.txt', ' ').replace('-', ' ')
        camera_model = ' '.join(camera_model.split()[:2])
        if camera_model.startswith('B'):
            camera_model = camera_model.replace('B', 'bobcat ')
        if not camera_model.startswith('bobcat'):
            camera_model = f"{'Cheetah'} {camera_model}"
    elif folder == "Keyence":
        # Custom logic for Keyence
        camera_model = filename.replace('-', ' ').replace('_', ' ')
        camera_model = camera_model.replace('Datasheet', ' ').strip()
        camera_model = ' '.join(camera_model.split()[:2])
    elif folder == "OPTO":
        # Custom logic for OPTO
        camera_model = filename.replace('-', ' ')
        camera_model = re.split(r'_EN', camera_model)[0].strip()
    else:
        # Default logic
        camera_model = os.path.splitext(filename)[0]

    return camera_model

# Enhanced sanitization function
from unidecode import unidecode

def sanitize_text(text):
    # Convert non-ASCII characters to their closest ASCII equivalents
    text = unidecode(text)
    
    # Replace specific known problematic characters with a space or an appropriate substitute
    replacements = {
        '•': '-',
        '\u2022': '-',  # Unicode bullet
        '»': '-',
        '™': '(TM)',
        '®': '(R)',
        '©': '(C)',
        'µ': 'u',
        '–': '-',  # en dash
        '—': '-',  # em dash
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'",
        '': '',  # unwanted character
        # Add more replacements as needed
    }

    # Apply the replacements
    for old, new in replacements.items():
        text = text.replace(old, new)

    return text

# Function to read text files and collect data
def collect_data_from_txt_files(folder):
    folder_path = os.path.join(base_output_dir, folder)
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(".txt"):
                file_path = os.path.join(root, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()

                # Sanitize the content
                sanitized_content = sanitize_text(content)
                
                camera_model = get_camera_model(folder, filename, sanitized_content, root)

                data.append({
                    "id": len(data) + 1,
                    "company_name": folder,
                    "camera_model": camera_model,
                    "datasheet_description": sanitized_content,
                    "Press_Release": "",
                    "Rationale": ""
                })

# Collect data from all specified folders
for folder in folders:
    collect_data_from_txt_files(folder)

# Check if data is collected
if not data:
    print("No data collected. Please check if the text files exist in the specified directories.")
else:
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Print the DataFrame to check if data is correctly collected
    print(df.head())

    # Define the output file paths
    output_csv = "/Users/jochem/Desktop/mijn_project2/Camera brands/dataset_output_txt_way/dataset_Press_Release_perpendicular.csv"
    output_excel = "/Users/jochem/Desktop/mijn_project2/Camera brands/dataset_output_txt_way/dataset_Press_Release_perpendicular.xlsx"

    # Save the DataFrame to CSV and Excel formats
    df.to_csv(output_csv, index=False)
    df.to_excel(output_excel, index=False, engine='xlsxwriter')

    print("Data successfully saved to CSV and Excel files.")
