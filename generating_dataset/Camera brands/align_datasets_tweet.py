import os
import pandas as pd
import re
import random

# Paths to the dataset and tweets directory
dataset_path = "/Users/jochem/Desktop/Mijn_project2/Camera brands/dataset_output_txt_way/dataset_twitter_perpendicular.csv"
tweets_base_folder_path = "/Users/jochem/Desktop/Mijn_project2/Camera brands/updated_tweets_csv"

# List of folders to process
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

# company_to_process = "Allied_Vision"

# Load the main dataset
filtered_dataset = pd.read_csv(dataset_path)

# filtered_dataset = dataset[dataset['company_name'].str.lower() == company_to_process.lower()].copy()

# Ensure the 'X' and 'LinkedIn' columns are initialized as string type to avoid dtype issues
for column in ['X']:
    if column not in filtered_dataset.columns:
        filtered_dataset[column] = pd.Series(dtype=str)
    else:
        filtered_dataset[column] = filtered_dataset[column].astype(str).replace('nan', '')

# Define a set to keep track of used tweets and LinkedIn posts
used_tweets = set()

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

def extract_prefix_allied_vision(camera_model):
    # Split the string at the first occurrence of a space or hyphen
    parts = camera_model.split(' ')
    # Combine the first two parts to get "Alvium G1"
    prefix = ' '.join(parts[:2])
    return prefix
    
def remove_hypen_allied(camera_model):
    parts = camera_model.split('-')
    prefix = ' '.join(parts)
    return prefix
    

def extract_code(sequence):
    # Regex to match the first two letters and the first letter/number sequence
    match = re.match(r'([A-Z]{2}).*?([A-Z]|\d+)', sequence, re.IGNORECASE)
    if match:
        return match.group(1) + '-' + match.group(2)
    return None

# Define a function to find and assign tweets and LinkedIn posts to the appropriate camera model
def assign_posts_to_models(dataset, tweets_base_folder_path):
    for index, row in dataset.iterrows():
        camera_model = row['camera_model']
        company = row['company_name']
        tweets_list = []

        # Skip "Baumer" and "ADIMEC" as there are no posts for these companies
        if company.lower() in ["baumer", "adimec"]:
            continue

        # Define the path to the specific company's tweets folders
        tweets_folder_path = os.path.join(tweets_base_folder_path, company)

        # Process tweets
        if os.path.exists(tweets_folder_path):
            for file_name in os.listdir(tweets_folder_path):
                if file_name.endswith(".csv"):
                    file_path = os.path.join(tweets_folder_path, file_name)
                    tweets_df = pd.read_csv(file_path)
                    
                    # Convert Analytics and Likes to integers, filling NaNs with 0
                    tweets_df['Analytics'] = pd.to_numeric(tweets_df['Analytics'], errors='coerce').fillna(0).astype(int)
                    tweets_df['Likes'] = pd.to_numeric(tweets_df['Likes'], errors='coerce').fillna(0).astype(int)
                    
                    for _, tweet_row in tweets_df.iterrows():
                        tweet_content = tweet_row['Content']
                        
                        if company.lower() == 'teledyne_dalsa':
                            if re.search(re.escape(camera_model), tweet_content, re.IGNORECASE):
                                tweets_list.append(tweet_row)
                            elif contains_all_parts(camera_model, tweet_content):
                                tweets_list.append(tweet_row)
                            elif contains_all_parts(only_choose_first_three_words(camera_model), tweet_content):
                                tweets_list.append(tweet_row)
                        elif company.lower() == 'allied_vision':
                            parts = re.split(r'[ -]', camera_model)
                            if re.search(re.escape(camera_model), tweet_content, re.IGNORECASE):
                                tweets_list.append(tweet_row)
                            elif contains_all_parts(camera_model, tweet_content):
                                tweets_list.append(tweet_row)
                            

                            elif parts[0] == "Nerian":
                                adjusted_model = remove_hypen_allied(camera_model)
                                pattern = re.escape(adjusted_model)
                                if re.search(pattern, tweet_content, re.IGNORECASE):
                                    tweets_list.append(tweet_row)
                            

                            # elif parts[1] == "1800" and parts[2] == "U":
                            #     pattern = r"Alvium (1800) (U) "
                                
                            #     # Check if the pattern is present in the tweet content
                            #     if re.search(pattern, tweet_content, re.IGNORECASE):
                            #         tweets_list.append(tweet_row)

                            # elif parts[1] == "1800" and parts[2] == "C":
                            #     pattern = r"Alvium (1800) (C) "
                                
                            #     # Check if the pattern is present in the tweet content
                            #     if re.search(pattern, tweet_content, re.IGNORECASE):
                            #         tweets_list.append(tweet_row)

                            # elif parts[1] == "1500" and parts[2] == "U":
                            #     pattern = r"Alvium (1500) (U) "
                                
                            #     # Check if the pattern is present in the tweet content
                            #     if re.search(pattern, tweet_content, re.IGNORECASE):
                            #         tweets_list.append(tweet_row)

                            # elif parts[1] == "1500" and parts[2] == "C":
                            #     pattern = r"Alvium (1500) (C) "
                                
                            #     # Check if the pattern is present in the tweet content
                            #     if re.search(pattern, tweet_content, re.IGNORECASE):
                            #         tweets_list.append(tweet_row)

                            # elif parts[1] not in {"1800", "1500"}:
                            #     adjusted_model = remove_hypen_allied(camera_model)
                            #     adjusted_model2 = only_choose_first_two_words(adjusted_model)
                            #     pattern = re.escape(adjusted_model2)
                            #     if re.search(pattern, tweet_content, re.IGNORECASE):
                            #         tweets_list.append(tweet_row)
            
                            

                        elif company.lower() == 'basler':
                            parts = re.split(r'\s+', camera_model)
                            pattern = r'\b' + re.escape(camera_model) + r'\b'
                            if re.search(pattern, tweet_content, re.IGNORECASE):
                                tweets_list.append(tweet_row)
                            elif contains_all_parts(camera_model, tweet_content):
                                tweets_list.append(tweet_row)
                            elif parts[1][0].isalpha():
                                adjusted_model = only_choose_first_two_words(camera_model)
                                pattern = r'\b' + re.escape(adjusted_model) + r'\b'
                                if re.search(pattern, tweet_content, re.IGNORECASE):
                                    tweets_list.append(tweet_row)
                                if parts[0] == "Pulse":
                                    adjusted_model = only_choose_first_word(camera_model)
                                    pattern = r'\b' + re.escape(adjusted_model) + r'\b'
                                    if re.search(pattern, tweet_content, re.IGNORECASE):
                                        tweets_list.append(tweet_row)
                            elif parts[1][0].isdigit():
                                adjusted_model = only_choose_first_three_words(camera_model)
                                pattern = r'\b' + re.escape(adjusted_model) + r'\b'
                                if re.search(pattern, tweet_content, re.IGNORECASE):
                                    tweets_list.append(tweet_row)
                        elif company.lower() == 'cognex':
                            pattern = r'\b' + re.escape(camera_model) + r'\b'
                            if re.search(pattern, tweet_content, re.IGNORECASE):
                                tweets_list.append(tweet_row)
                        elif company.lower() == 'flir':
                            if re.search(re.escape(camera_model), tweet_content, re.IGNORECASE):
                                tweets_list.append(tweet_row)
                            elif contains_all_parts(adjust_flir_model(camera_model), tweet_content):
                                tweets_list.append(tweet_row)
                        elif company.lower() == 'framos':
                            adjusted_model = only_choose_second_word(camera_model)
                            if re.search(re.escape(adjusted_model), tweet_content, re.IGNORECASE):
                                tweets_list.append(tweet_row)
                        elif company.lower() == 'hamamatsu':
                            parts = re.split(r'\s+', camera_model)
                            pattern = r'\b' + re.escape(camera_model) + r'\b'
                            if re.search(re.escape(camera_model), tweet_content, re.IGNORECASE):
                                tweets_list.append(tweet_row)
                            elif contains_all_parts(camera_model, tweet_content):
                                tweets_list.append(tweet_row)
                            if parts[0] == "InGaAs":
                                adjusted_model = only_choose_first_word(camera_model)
                                pattern = r'\b' + re.escape(adjusted_model) + r'\b'
                                if re.search(pattern, tweet_content, re.IGNORECASE):
                                    tweets_list.append(tweet_row)
                            elif parts[0] == "ORCA" or parts[0] == "CMOS":
                                adjusted_model = only_choose_first_two_words(camera_model)
                                pattern = re.escape(adjusted_model)
                                if re.search(pattern, tweet_content, re.IGNORECASE):
                                    tweets_list.append(tweet_row)
                        elif company.lower() == 'ids_imaging':
                            parts = re.split(r'\s+', camera_model)
                            pattern = r'\b' + re.escape(camera_model) + r'\b'
                            if re.search(pattern, tweet_content, re.IGNORECASE):
                                tweets_list.append(tweet_row)
                            if parts[0].lower() == "ids":
                                adjusted_model = only_choose_first_three_words(camera_model)
                                pattern = re.escape(adjusted_model)
                                if re.search(pattern, tweet_content, re.IGNORECASE):
                                    tweets_list.append(tweet_row)
                            elif parts[2].isdigit():
                                adjusted_model = only_choose_first_two_words(camera_model)
                                pattern = re.escape(adjusted_model)
                                if re.search(pattern, tweet_content, re.IGNORECASE):
                                    tweets_list.append(tweet_row)
                        elif company.lower() == 'imperx':
                            parts = re.split(r'\s+', camera_model)
                            pattern = r'\b' + re.escape(camera_model) + r'\b'
                            if re.search(re.escape(camera_model), tweet_content, re.IGNORECASE):
                                tweets_list.append(tweet_row)
                            adjusted_model = only_choose_first_word(camera_model)
                            pattern = r'\b' + re.escape(adjusted_model) + r'\b'
                            if re.search(pattern, tweet_content, re.IGNORECASE):
                                tweets_list.append(tweet_row)
                        elif company.lower() == 'keyence':
                            parts = re.split(r'\s+', camera_model)
                            pattern = r'\b' + re.escape(camera_model) + r'\b'
                            if re.search(re.escape(camera_model), tweet_content, re.IGNORECASE):
                                tweets_list.append(tweet_row)
                            if parts[0] == 'IV' or parts[0] == 'IV2' or parts[0] == 'IV3':
                                adjusted_model = only_choose_first_word(camera_model)
                                pattern = r'\b' + re.escape(adjusted_model) + r'\b'
                                if re.search(pattern, tweet_content, re.IGNORECASE):
                                    tweets_list.append(tweet_row)
                            else:
                                adjusted_model = extract_code(camera_model)
                                pattern = r'\b' + re.escape(adjusted_model) + r'\b'
                                if re.search(pattern, tweet_content, re.IGNORECASE):
                                    tweets_list.append(tweet_row)
                        elif company.lower() == 'opto':
                            parts = re.split(r'\s+', camera_model)
                            pattern = r'\b' + re.escape(camera_model) + r'\b'
                            if re.search(re.escape(camera_model), tweet_content, re.IGNORECASE):
                                tweets_list.append(tweet_row)
                            adjusted_model = only_choose_first_word(camera_model)
                            pattern = r'\b' + re.escape(adjusted_model) + r'\b'
                            if re.search(pattern, tweet_content, re.IGNORECASE):
                                tweets_list.append(tweet_row)
                        else:
                            if re.search(re.escape(camera_model), tweet_content, re.IGNORECASE) or contains_all_parts(camera_model, tweet_content):
                                tweets_list.append(tweet_row)

        if len(tweets_list) > 0:
            tweets_list.sort(key=lambda x: (x['Analytics'] + x['Likes']), reverse=True)
            top_tweet = next((tweet for tweet in tweets_list if tweet['Content'] not in used_tweets), None)
            if top_tweet is not None:
                used_tweets.add(top_tweet['Content'])
                dataset.at[index, 'X'] = f"{top_tweet['Content']}"

    return dataset

# Call the function to assign posts
updated_dataset = assign_posts_to_models(filtered_dataset, tweets_base_folder_path)

# Print product descriptions with their corresponding value in X and LinkedIn
for index, row in updated_dataset.iterrows():
    print(f"Product: {row['camera_model']} - Tweets: {row['X']}")

# Print the number of rows in the 'X' and 'LinkedIn' columns that have been filled
filled_x_count = updated_dataset['X'].apply(lambda x: isinstance(x, str) and x.strip() != "").sum()
print(f"Number of entries in 'X' column that have been filled: {filled_x_count}")

# Save the updated dataset to CSV and Excel
updated_dataset.to_csv("/Users/jochem/Desktop/Mijn_project2/Camera brands/dataset_output_txt_way/dataset_twitter_aligned.csv", index=False)
updated_dataset.to_excel("/Users/jochem/Desktop/Mijn_project2/Camera brands/dataset_output_txt_way/dataset_twitter_aligned.xlsx", index=False, engine='xlsxwriter')

print("Data successfully saved to CSV and Excel files.")
