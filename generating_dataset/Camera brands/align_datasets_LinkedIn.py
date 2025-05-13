import os
import pandas as pd
import re
import random
import emoji

# Paths to the dataset and tweets directory
dataset_path = "/Users/jochem/Desktop/Mijn_project2/Camera brands/dataset_output_txt_way/dataset_LinkedIn_perpendicular.csv"
linkedin_posts_folder_path = "/Users/jochem/Desktop/Mijn_project2/Camera brands/LinkedIn_posts_csv"

# List of folders to process
folders = [
    "Adimec",
    "Teledyne_Dalsa",
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

# company_to_process = "Allied_Vision"

# Load the main dataset
filtered_dataset = pd.read_csv(dataset_path)

# filtered_dataset = dataset[dataset['company_name'].str.lower() == company_to_process.lower()].copy()

# Ensure the 'X' and 'LinkedIn' columns are initialized as string type to avoid dtype issues
for column in ['LinkedIn']:
    if column not in filtered_dataset.columns:
        filtered_dataset[column] = pd.Series(dtype=str)
    else:
        filtered_dataset[column] = filtered_dataset[column].astype(str).replace('nan', '')

# Define a set to keep track of used tweets and LinkedIn posts
used_tweets = set()
used_linkedin_posts = set()

# Function to remove emojis
# def convert_emojis_to_unicode(text):
#     return emoji.demojize(text)

# Define a function to check if all parts of the camera model are in the text
def contains_all_parts(camera_model, text):
    parts = re.split(r'\s+', camera_model)  # Split model name into parts by whitespace
    return all(re.search(re.escape(part), text, re.IGNORECASE) for part in parts)

# Define a function to adjust the camera model for FLIR
def adjust_flir_model(camera_model):
    parts = re.split(r'\s+', camera_model)
    if len(parts) > 1 and parts[1][0].isdigit():
        return parts[0]  # Use only the first word
    return ' '.join(parts[:2])  # Use the first two words

def only_choose_first_two_words(camera_model):
    parts = re.split(r'\s+', camera_model)
    return ' '.join(parts[:2])  # Use the first two words

def only_choose_first_three_words(camera_model):
    parts = re.split(r'\s+', camera_model)
    return ' '.join(parts[:3])  # Use the first three words

def only_choose_first_word(camera_model):
    parts = re.split(r'\s+', camera_model)
    return ' '.join(parts[:1])  # Use the first word

def only_choose_second_word(camera_model):
    parts = re.split(r'\s+', camera_model)
    second_word = parts[1] if len(parts) > 1 else parts[0]
    return second_word[:-1] if len(second_word) > 1 else second_word  # Remove the last letter of the second word if it exists

def extract_code(sequence):
    # Regex to match the first two letters and the first letter/number sequence
    match = re.match(r'([A-Z]{2}).*?([A-Z]|\d+)', sequence, re.IGNORECASE)
    if match:
        return match.group(1) + '-' + match.group(2)
    return None    

def assign_posts_to_models(dataset, linkedin_posts_folder_path):
    for index, row in dataset.iterrows():
        camera_model = row['camera_model']
        company = row['company_name']
        linkedin_posts_list = []

        # Process LinkedIn posts
        if os.path.exists(linkedin_posts_folder_path):
            # print(f"Processing LinkedIn posts for company: {company}")  # Debugging info
            for file_name in os.listdir(linkedin_posts_folder_path):
                if file_name.endswith(".csv"):
                    file_path = os.path.join(linkedin_posts_folder_path, file_name)
                    linkedin_df = pd.read_csv(file_path)
                    
                    linkedin_df['reaction_count'] = pd.to_numeric(linkedin_df['reaction_count'], errors='coerce').fillna(0).astype(int)
                    linkedin_df['comment_count'] = pd.to_numeric(linkedin_df['comment_count'], errors='coerce').fillna(0).astype(int)
                    linkedin_df['repost_count'] = pd.to_numeric(linkedin_df['repost_count'], errors='coerce').fillna(0).astype(int)
                    
                    for _, post_row in linkedin_df.iterrows():
                        post_content = str(post_row['post_text'])  # Ensure post_content is a string
                        # post_content = convert_emojis_to_unicode(post_content)  # Remove emojis from the post content
                        if post_row['company_name'].lower() == company.lower():
                            if company.lower() == 'teledyne_dalsa':
                                if re.search(re.escape(camera_model), post_content, re.IGNORECASE):
                                    linkedin_posts_list.append(post_row)
                                elif contains_all_parts(camera_model, post_content):
                                    linkedin_posts_list.append(post_row)
                                elif contains_all_parts(only_choose_first_three_words(camera_model), post_content):
                                    linkedin_posts_list.append(post_row)
                            elif company.lower() == 'allied_vision':
                                if re.search(re.escape(camera_model), post_content, re.IGNORECASE):
                                    linkedin_posts_list.append(post_row)
                                elif contains_all_parts(camera_model, post_content):
                                    linkedin_posts_list.append(post_row)
                            elif company.lower() == 'basler':
                                parts = re.split(r'\s+', camera_model)
                                pattern = r'\b' + re.escape(camera_model) + r'\b'
                                if re.search(pattern, post_content, re.IGNORECASE):
                                    linkedin_posts_list.append(post_row)
                                elif contains_all_parts(camera_model, post_content):
                                    linkedin_posts_list.append(post_row)
                                elif parts[1][0].isalpha():
                                    adjusted_model = only_choose_first_two_words(camera_model)
                                    pattern = r'\b' + re.escape(adjusted_model) + r'\b'
                                    if re.search(pattern, post_content, re.IGNORECASE):
                                        linkedin_posts_list.append(post_row)
                                    if parts[0] == "Pulse":
                                        adjusted_model = only_choose_first_word(camera_model)
                                        pattern = r'\b' + re.escape(adjusted_model) + r'\b'
                                        if re.search(pattern, post_content, re.IGNORECASE):
                                            linkedin_posts_list.append(post_row)
                                elif parts[1][0].isdigit():
                                    adjusted_model = only_choose_first_three_words(camera_model)
                                    pattern = r'\b' + re.escape(adjusted_model) + r'\b'
                                    if re.search(pattern, post_content, re.IGNORECASE):
                                        linkedin_posts_list.append(post_row)
                            elif company.lower() == 'cognex':
                                pattern = r'\b' + re.escape(camera_model) + r'\b'
                                if re.search(pattern, post_content, re.IGNORECASE):
                                    linkedin_posts_list.append(post_row)
                            elif company.lower() == 'flir':
                                if re.search(re.escape(camera_model), post_content, re.IGNORECASE):
                                    linkedin_posts_list.append(post_row)
                                elif contains_all_parts(adjust_flir_model(camera_model), post_content):
                                    linkedin_posts_list.append(post_row)
                            elif company.lower() == 'framos':
                                adjusted_model = only_choose_second_word(camera_model)
                                if re.search(re.escape(adjusted_model), post_content, re.IGNORECASE):
                                    linkedin_posts_list.append(post_row)
                            elif company.lower() == 'hamamatsu':
                                parts = re.split(r'\s+', camera_model)
                                pattern = r'\b' + re.escape(camera_model) + r'\b'
                                if re.search(re.escape(camera_model), post_content, re.IGNORECASE):
                                    linkedin_posts_list.append(post_row)
                                elif contains_all_parts(camera_model, post_content):
                                    linkedin_posts_list.append(post_row)
                                if parts[0] == "InGaAs":
                                    adjusted_model = only_choose_first_word(camera_model)
                                    pattern = r'\b' + re.escape(adjusted_model) + r'\b'
                                    if re.search(pattern, post_content, re.IGNORECASE):
                                        linkedin_posts_list.append(post_row)
                                elif parts[0] == "ORCA" or parts[0] == "CMOS":
                                    adjusted_model = only_choose_first_two_words(camera_model)
                                    pattern = re.escape(adjusted_model)
                                    if re.search(pattern, post_content, re.IGNORECASE):
                                        linkedin_posts_list.append(post_row)
                            elif company.lower() == 'ids_imaging':
                                parts = re.split(r'\s+', camera_model)
                                pattern = r'\b' + re.escape(camera_model) + r'\b'
                                if re.search(pattern, post_content, re.IGNORECASE):
                                    linkedin_posts_list.append(post_row)
                                if parts[0].lower() == "ids":
                                    adjusted_model = only_choose_first_three_words(camera_model)
                                    pattern = re.escape(adjusted_model)
                                    if re.search(pattern, post_content, re.IGNORECASE):
                                        linkedin_posts_list.append(post_row)
                                elif parts[2].isdigit():
                                    adjusted_model = only_choose_first_two_words(camera_model)
                                    pattern = re.escape(adjusted_model)
                                    if re.search(pattern, post_content, re.IGNORECASE):
                                        linkedin_posts_list.append(post_row)
                            elif company.lower() == 'imperx':
                                parts = re.split(r'\s+', camera_model)
                                pattern = r'\b' + re.escape(camera_model) + r'\b'
                                if re.search(re.escape(camera_model), post_content, re.IGNORECASE):
                                    linkedin_posts_list.append(post_row)
                                adjusted_model = only_choose_first_word(camera_model)
                                pattern = r'\b' + re.escape(adjusted_model) + r'\b'
                                if re.search(pattern, post_content, re.IGNORECASE):
                                    linkedin_posts_list.append(post_row)
                            elif company.lower() == 'keyence':
                                parts = re.split(r'\s+', camera_model)
                                pattern = r'\b' + re.escape(camera_model) + r'\b'
                                if re.search(re.escape(camera_model), post_content, re.IGNORECASE):
                                    linkedin_posts_list.append(post_row)
                                if parts[0] == 'IV' or parts[0] == 'IV2' or parts[0] == 'IV3':
                                    adjusted_model = only_choose_first_word(camera_model)
                                    pattern = r'\b' + re.escape(adjusted_model) + r'\b'
                                    if re.search(pattern, post_content, re.IGNORECASE):
                                        linkedin_posts_list.append(post_row)
                                else:
                                    adjusted_model = extract_code(camera_model)
                                    pattern = r'\b' + re.escape(adjusted_model) + r'\b'
                                    if re.search(pattern, post_content, re.IGNORECASE):
                                        linkedin_posts_list.append(post_row)
                            elif company.lower() == 'opto':
                                parts = re.split(r'\s+', camera_model)
                                pattern = r'\b' + re.escape(camera_model) + r'\b'
                                if re.search(re.escape(camera_model), post_content, re.IGNORECASE):
                                    linkedin_posts_list.append(post_row)
                                adjusted_model = only_choose_first_word(camera_model)
                                pattern = r'\b' + re.escape(adjusted_model) + r'\b'
                                if re.search(pattern, post_content, re.IGNORECASE):
                                    linkedin_posts_list.append(post_row)
                            else:
                                if re.search(re.escape(camera_model), post_content, re.IGNORECASE) or contains_all_parts(camera_model, post_content):
                                    linkedin_posts_list.append(post_row)

        if len(linkedin_posts_list) > 0:
            linkedin_posts_list.sort(key=lambda x: (x['reaction_count'] + x['comment_count'] + x['repost_count']), reverse=True)
            top_post = next((post for post in linkedin_posts_list if post['post_text'] not in used_linkedin_posts), None)
            if top_post is not None:
                used_linkedin_posts.add(top_post['post_text'])
                dataset.at[index, 'LinkedIn'] = f"{top_post['post_text']}"
            else:
                print(f"No suitable LinkedIn post found for company: {company}, model: {camera_model}")

    return dataset

# Call the function to assign posts
updated_dataset = assign_posts_to_models(filtered_dataset, linkedin_posts_folder_path)

# Print product descriptions with their corresponding value in X and LinkedIn
for index, row in updated_dataset.iterrows():
    print(f"Product: {row['camera_model']} - LinkedIn: {row['LinkedIn']}")

# Print the number of rows in the 'X' and 'LinkedIn' columns that have been filled
filled_linkedin_count = updated_dataset['LinkedIn'].apply(lambda x: isinstance(x, str) and x.strip() != "").sum()
print(f"Number of entries in 'LinkedIn' column that have been filled: {filled_linkedin_count}")

# Save the updated dataset to CSV and Excel
updated_dataset.to_csv("/Users/jochem/Desktop/Mijn_project2/Camera brands/dataset_output_txt_way/dataset_LinkedIn_aligned.csv", index=False)
updated_dataset.to_excel("/Users/jochem/Desktop/Mijn_project2/Camera brands/dataset_output_txt_way/dataset_LinkedIn_aligned.xlsx", index=False, engine='xlsxwriter')

print("Data successfully saved to CSV and Excel files.")
