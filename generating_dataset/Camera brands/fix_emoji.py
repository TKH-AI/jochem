import os
import pandas as pd
import ast
import numpy as np

tweets_base_folder_path = "/Users/jochem/Desktop/Mijn_project/Camera brands/tweets_csv"
output_base_folder_path = "/Users/jochem/Desktop/Mijn_project/Camera brands/updated_tweets_csv"
folders = [
    "Teledyne_Dalsa",
    "Allied_Vision",
    "Basler",
    "Cognex",
    "FLIR",
    "Framos",
    "Hamamatsu",
    "IDS_Imaging",
    "Imperx",
    "Keyence",
    "OPTO"
]

# Ensure output directory exists
os.makedirs(output_base_folder_path, exist_ok=True)

def decode_emoji_code(emoji_code):
    return emoji_code.encode().decode('unicode_escape')

def insert_emojis(content, emojis):
    if not emojis or pd.isna(emojis):
        return content

    emoji_list = ast.literal_eval(emojis)
    emoji_list = [decode_emoji_code(e) for e in emoji_list]  # Convert codes to visual emojis

    # Insert up to 2 emojis at the beginning of the content if there is a leading space
    if content.startswith(" "):
        content = " ".join(emoji_list[:2]) + content
        emoji_list = emoji_list[2:]
    
    # Insert up to 2 emojis where there is a double space
    while "  " in content and emoji_list:
        insert_count = min(2, len(emoji_list))
        content = content.replace("  ", " " + " ".join(emoji_list[:insert_count]) + " ", 1)
        emoji_list = emoji_list[insert_count:]
    
    # Ensure any remaining emojis are appended at the end
    if emoji_list:
        content += " " + " ".join(emoji_list)

    return content.strip()

def process_csv(file_path, output_path):
    df = pd.read_csv(file_path)
    df['Content'] = df.apply(lambda row: insert_emojis(row['Content'], row['Emojis']), axis=1)
    df.to_csv(output_path, index=False)

for folder in folders:
    folder_path = os.path.join(tweets_base_folder_path, folder)
    output_folder_path = os.path.join(output_base_folder_path, folder)
    os.makedirs(output_folder_path, exist_ok=True)  # Ensure output subfolder exists
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            output_path = os.path.join(output_folder_path, file_name)
            process_csv(file_path, output_path)
