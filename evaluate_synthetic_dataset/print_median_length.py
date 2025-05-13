import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
import transformers

# Download NLTK tokenizer data if not already available
nltk.download('punkt')

# Paths to the datasets
real_datasets = {
    'LinkedIn': '/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Real/dataset_LinkedIn_aligned.csv',
    'Press_Release': '/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Real/dataset_Press_Release_aligned.csv',
    'X': '/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Real/dataset_twitter_aligned.csv'
}

synthetic_datasets = {
    'LinkedIn': '/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Synthetic/final_linkedin_clean_llama.csv',
    'Press_Release': '/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Synthetic/final_press_release_clean_llama.csv',
    'X': '/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Synthetic/final_x_clean_llama.csv'
}

# Initialize tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Function to calculate average token length of filled rows
def calculate_average_token_length(df, column):
    filled_values = df[column].dropna()
    avg_token_length = filled_values.apply(lambda x: len(tokenizer.tokenize(x))).mean()
    return avg_token_length

# Calculate average token length for Real datasets
print("Average Token Lengths for Real Datasets:")
for dataset_name, path in real_datasets.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        avg_token_length = calculate_average_token_length(df, dataset_name)
        print(f"{dataset_name} (Real): {avg_token_length:.2f}")
    else:
        print(f"{dataset_name} (Real): File not found")

# Calculate average token length for Synthetic datasets
print("\nAverage Token Lengths for Full (Synthetic + Real) Datasets:")
for dataset_name, path in synthetic_datasets.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        avg_token_length = calculate_average_token_length(df, dataset_name)
        print(f"{dataset_name} (Full): {avg_token_length:.2f}")
    else:
        print(f"{dataset_name} (Full): File not found")

# Calculate average token length for Synthetic parts
print("\nAverage Token Lengths for Synthetic Datasets:")
for dataset_name, real_path in real_datasets.items():
    synthetic_path = synthetic_datasets[dataset_name]
    if os.path.exists(real_path) and os.path.exists(synthetic_path):
        real_df = pd.read_csv(real_path)
        synthetic_df = pd.read_csv(synthetic_path)
        # Align both DataFrames by using the real_df index to filter synthetic_df
        real_df = real_df.reset_index(drop=True)
        synthetic_df = synthetic_df.reset_index(drop=True)
        synthetic_only = synthetic_df[(synthetic_df[dataset_name].notna()) & (real_df[dataset_name].isna())]
        avg_token_length = synthetic_only[dataset_name].apply(lambda x: len(tokenizer.tokenize(x))).mean()
        print(f"{dataset_name} (Synthetic Only): {avg_token_length:.2f}")
    else:
        print(f"{dataset_name} (Synthetic Only): File not found")
