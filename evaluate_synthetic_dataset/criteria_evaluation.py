import os
import pandas as pd
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import re
import requests.exceptions
import time
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Set up the ChatOpenAI LLM
OPENAI_KEY_GPT4 = os.getenv("OPENAI_KEY_GPT4")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key="EMPTY",
    api_key=OPENAI_KEY_GPT4,
    openai_api_base="https://ai-gateway.mytkhgroup.com",
    temperature=0.1,
    streaming=False,
    max_tokens=10,
)

def generate_prompt(source_description, synthetic_copy, criterion_name):
    criterion_descriptions = {
        'relevance': "Evaluate the semantic relevance of the synthetic marketing copy to the original product sheet description.",
        'attractiveness': "Evaluate how attractive or engaging the synthetic marketing copy is to the audience.",
        'fluency': "Evaluate the fluency, clarity, and grammatical correctness of the synthetic marketing copy.",
        'style_strength': "Evaluate how well the synthetic marketing copy conforms to the target style for the specific platform (e.g., casual for X, professional for LinkedIn, or formal for Press Release)."
    }
    prompt = (
        f"**Product Sheet Description:**\n{source_description}\n\n"
        f"**Synthetic Marketing Copy:**\n{synthetic_copy}\n\n"
        f"**Task:** {criterion_descriptions[criterion_name]} Rate it on a Likert scale from 1 to 5, where:\n"
        f"1 - Very Poor\n"
        f"3 - Average\n"
        f"5 - Excellent\n\n"
        f"**Provide only the rating number (1, 2, 3, 4, or 5):**"
    )
    return prompt

def evaluate_criterion(source_description, synthetic_copy, criterion_name, llm, timeout=60):
    prompt = generate_prompt(source_description, synthetic_copy, criterion_name)
    max_attempts = 3
    attempt = 0
    rating = None

    while attempt < max_attempts and rating is None:
        try:
            start_time = time.time()
            response = llm.invoke(prompt)
            end_time = time.time()

            if end_time - start_time > timeout:
                raise requests.exceptions.ReadTimeout

            response_text = response.content.strip()
            print(f"Response for {criterion_name}, attempt {attempt + 1}: {response_text}")

            # Extract the rating (1-5) from the response
            rating_match = re.search(r'\b([1-5])\b', response_text)
            if rating_match:
                rating = int(rating_match.group(1))
            else:
                print("Error: No valid rating found. Retrying...")

        except requests.exceptions.ReadTimeout:
            print(f"Timeout occurred during {criterion_name} evaluation, attempt {attempt + 1}. Retrying...")

        attempt += 1

    if rating is None:
        print(f"Failed to obtain a valid rating for {criterion_name} after multiple attempts.")
    
    return rating

def identify_synthetic_data(real_dataset_path, synthetic_dataset_path, synthetic_col):
    if not os.path.exists(real_dataset_path) or not os.path.exists(synthetic_dataset_path):
        print(f"Error: File {real_dataset_path} or {synthetic_dataset_path} does not exist. Skipping.")
        return None

    # Load real and synthetic datasets
    real_df = pd.read_csv(real_dataset_path)
    synthetic_df = pd.read_csv(synthetic_dataset_path)

    # Identify synthetic rows as those that have non-empty values in the synthetic dataset column but are empty in the real dataset
    real_empty_mask = real_df[synthetic_col].isna() | (real_df[synthetic_col].str.strip() == '')
    synthetic_rows = synthetic_df[real_empty_mask]

    # Return the synthetic rows only
    return synthetic_rows

def sample_synthetic_data(synthetic_datasets_info, real_datasets_info, sample_fraction=0.1, output_suffix='_sampled'):
    for real_path, synthetic_path, synthetic_col in zip(real_datasets_info, synthetic_datasets_info.keys(), synthetic_datasets_info.values()):
        synthetic_rows = identify_synthetic_data(real_path, synthetic_path, synthetic_col)
        
        if synthetic_rows is None or synthetic_rows.empty:
            print(f"No synthetic data found in column '{synthetic_col}' of {synthetic_path}. Skipping.")
            continue

        # Sample 10% of the synthetic rows
        sampled_df = synthetic_rows.sample(frac=sample_fraction, random_state=42)

        # Define the sampled file path
        sampled_file_path = synthetic_path.replace(".csv", f"{output_suffix}.csv")

        # Save the sampled data
        sampled_df.to_csv(sampled_file_path, index=False)
        print(f"Sampled {len(sampled_df)} synthetic rows saved to {sampled_file_path}")

async def evaluate_criteria_async(row, synthetic_col, llm):
    source = row['datasheet_description']
    synthetic_copy = row[synthetic_col]

    if pd.isna(synthetic_copy) or synthetic_copy.strip() == '':
        print("Synthetic copy is empty. Skipping this row.")
        return None

    criteria = ['relevance', 'attractiveness', 'fluency', 'style_strength']
    results = {'input': source, 'synthetic_copy': synthetic_copy}

    for criterion in criteria:
        rating = evaluate_criterion(source, synthetic_copy, criterion, llm)
        if rating is None:
            print(f"Invalid rating found for {criterion}. Skipping this row.")
            return None
        results[f'{criterion}_rating'] = rating

    return results

async def evaluate_criteria_dataset(file_path, synthetic_col, dataset_name, llm, save_interval=10, start_index=0):
    df = pd.read_csv(file_path)
    result_file_path = file_path.replace(".csv", "_criteria_evaluation.csv")

    if os.path.exists(result_file_path):
        print(f"Resuming from existing file: {result_file_path}")
        existing_results = pd.read_csv(result_file_path)
        existing_results = existing_results.to_dict('records')
        start_index = len(existing_results)
    else:
        print(f"Starting fresh evaluation for {dataset_name} with {len(df)} rows")
        existing_results = []

    results = existing_results

    for idx, row in df.iterrows():
        if idx < start_index:
            continue

        print(f"Processing row {idx + 1} of {len(df)} in {dataset_name}")
        eval_result = await evaluate_criteria_async(row, synthetic_col, llm)

        if eval_result is None:
            print(f"Row {idx + 1} skipped due to invalid rating or empty synthetic copy.")
            continue

        results.append(eval_result)

        if (idx + 1) % save_interval == 0 or idx == len(df) - 1:
            result_df = pd.DataFrame(results)
            result_df.to_csv(result_file_path, mode='w', header=True, index=False)
            print(f"Progress saved after {idx + 1} rows to {result_file_path}")

async def main_evaluation():
    # Define your synthetic datasets and their corresponding synthetic copy column names
    synthetic_datasets_info = {
        '/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Synthetic/final_linkedin_clean_llama.csv': 'LinkedIn',
        '/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Synthetic/final_press_release_clean_llama.csv': 'Press_Release',
        '/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Synthetic/final_x_clean_llama.csv': 'X'
    }

    real_datasets_info = [
        '/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Real/dataset_LinkedIn_aligned.csv',
        '/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Real/dataset_Press_Release_aligned.csv',
        '/Users/jochem/Desktop/thesis-code-methods/evaluate_synthetic_dataset/Real/dataset_twitter_aligned.csv'
    ]

    # Perform sampling before evaluation
    print("Starting sampling of synthetic datasets...")
    sample_synthetic_data(synthetic_datasets_info, real_datasets_info, sample_fraction=0.1)
    print("Sampling completed.\n")

    # Update the paths to include the sampled files
    sampled_datasets_info = {file_path.replace(".csv", "_sampled.csv"): col for file_path, col in synthetic_datasets_info.items()}

    with ThreadPoolExecutor() as executor:
        tasks = []
        for file_path, synthetic_col in sampled_datasets_info.items():
            if not os.path.exists(file_path):
                print(f"Sampled file {file_path} does not exist. Skipping evaluation for this dataset.")
                continue

            dataset_name = os.path.basename(file_path).replace(".csv", "")
            print(f"Starting evaluation for dataset: {dataset_name}")
            task = asyncio.create_task(
                evaluate_criteria_dataset(
                    file_path=file_path,
                    synthetic_col=synthetic_col,
                    dataset_name=dataset_name,
                    llm=llm,
                    save_interval=10
                )
            )
            tasks.append(task)
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main_evaluation())
