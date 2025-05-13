import os
import pandas as pd
import asyncio
import random
from concurrent.futures import ThreadPoolExecutor
from langchain_community.llms import HuggingFaceEndpoint
from dotenv import load_dotenv
import re
import requests.exceptions
import time
from langchain_openai import ChatOpenAI

# Constants
INPUT_FILES = [
    '/Users/jochem/Desktop/thesis-code-methods/evaluation_llm/generated_answers_linkedin/answers_all_runs_big_and_small_fourth_run_goodone.csv',
    '/Users/jochem/Desktop/thesis-code-methods/evaluation_llm/generated_answers_press_release/answers_all_runs_big_and_small_fourth_run_goodone.csv', 
    '/Users/jochem/Desktop/thesis-code-methods/evaluation_llm/generated_answers_x/answers_all_runs_big_and_small_fourth_run_goodone.csv'
]
SAVE_INTERVAL = 10

# Load environment variables 
load_dotenv()

# Set up the LLM
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

def generate_prompt(criterion_name, source, distill_output, vanilla_output):
    methods = [(distill_output, 'Llama'), (vanilla_output, 'Vanilla')]
    random.shuffle(methods)
    
    method_1_output, method_1_label = methods[0]
    method_2_output, method_2_label = methods[1]

    if criterion_name == 'relevance':
        prompt = (f"You will be presented with two marketing copies generated from the same product sheet description: "
                  f"one from Method 1 and one from Method 2. Please select the marketing copy that is more semantically relevant to the original product sheet description.\n\n"
                  f"Source: {source}\n"
                  f"Method 1: {method_1_output}\n"
                  f"Method 2: {method_2_output}\n"
                  f"Which marketing copy is more semantically relevant to the original product sheet description?\n"
                  f"* Method 1\n* Method 2\n"
                  f"Only provide the number '1' or '2', nothing else.")
    elif criterion_name == 'attractiveness':
        prompt = (f"You will be presented with two marketing copies generated from the same product sheet description: "
                  f"one from Method 1 and one from Method 2. Please select the marketing copy that is more attractive or engaging to the audience.\n\n"
                  f"Source: {source}\n"
                  f"Method 1: {method_1_output}\n"
                  f"Method 2: {method_2_output}\n"
                  f"Which marketing copy is more attractive and engaging to the audience?\n"
                  f"* Method 1\n* Method 2\n"
                  f"Only provide the number '1' or '2', nothing else.")
    elif criterion_name == 'fluency':
        prompt = (f"You will be presented with two marketing copies generated from the same product sheet description: "
                  f"one from Method 1 and one from Method 2. Please select the marketing copy that is more fluent, clear, and grammatically correct.\n\n"
                  f"Source: {source}\n"
                  f"Method 1: {method_1_output}\n"
                  f"Method 2: {method_2_output}\n"
                  f"Which marketing copy is more fluent and grammatically correct?\n"
                  f"* Method 1\n* Method 2\n"
                  f"Only provide the number '1' or '2', nothing else.")
    elif criterion_name == 'style_strength':
        prompt = (f"You will be presented with two marketing copies generated from the same product sheet description: "
                  f"one from Method 1 and one from Method 2. Please select the marketing copy that better conforms to the target style for the specific platform (e.g., casual for X, professional for LinkedIn, or formal for Press Release).\n\n"
                  f"Source: {source}\n"
                  f"Method 1: {method_1_output}\n"
                  f"Method 2: {method_2_output}\n"
                  f"Which marketing copy better conforms to the target style?\n"
                  f"* Method 1\n* Method 2\n"
                  f"Only provide the number '1' or '2', nothing else.")

    return prompt, method_1_label, method_2_label

def evaluate_criterion(criterion_name, source, llama_output, vanilla_output, timeout=300):
    prompt, method_1_label, method_2_label = generate_prompt(criterion_name, source, llama_output, vanilla_output)
    max_attempts = 3
    attempt = 0
    choice = None

    while attempt < max_attempts and choice is None:
        try:
            start_time = time.time()
            response = llm.invoke(prompt)
            end_time = time.time()

            if end_time - start_time > timeout:
                raise requests.exceptions.ReadTimeout

            response_text = response.content

            print(f"Response for {criterion_name}, attempt {attempt + 1}: {response_text}")

            choice_match = re.search(r'\b(1|2)\b', response_text)
            if choice_match:
                choice = int(choice_match.group())
            else:
                print(f"Error: No valid choice found in response for {criterion_name}. Retrying...")

        except requests.exceptions.ReadTimeout:
            print(f"Timeout occurred for {criterion_name}, attempt {attempt + 1}. Retrying...")

        attempt += 1

    if choice is None:
        print(f"Failed to get valid choice for {criterion_name} after {max_attempts} attempts. Skipping this row.")
    return choice, method_1_label, method_2_label

async def evaluate_outputs(row, size):
    source = row['input']
    llama_output = row[f'distill_run_{size}'] # Changed from llama_run to distill_run
    vanilla_output = row[f'vanilla_run_{size}']

    criteria = ['relevance', 'attractiveness', 'fluency', 'style_strength']
    scores = {}
    labels = {}

    for criterion in criteria:
        score, method_1_label, method_2_label = evaluate_criterion(criterion, source, llama_output, vanilla_output)
        scores[criterion] = score
        labels[criterion] = (method_1_label, method_2_label)

    if any(score is None for score in scores.values()):
        print("Invalid scores found. Skipping this row.")
        return None

    return scores, labels

def get_result_filepath(original_path: str) -> str:
    dir_path = os.path.dirname(original_path)
    filename = os.path.basename(original_path)
    base_name = filename.replace(".csv", "")
    return os.path.join(dir_path, f"{base_name}_evaluation_results.csv")


async def evaluate_dataset(file_path: str, save_interval: int = 10):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return

    sizes = [10, 12, 75, 100]  # Define the sizes we want to evaluate
    
    for size in sizes:
        result_file_path = get_result_filepath(file_path).replace('.csv', f'_size_{size}.csv')
        criteria = ['relevance', 'attractiveness', 'fluency', 'style_strength']
        
        if os.path.exists(result_file_path):
            print(f"Resuming from existing file for size {size}: {result_file_path}")
            existing_results = pd.read_csv(result_file_path).to_dict('records')
            start_index = len(existing_results)
        else:
            print(f"Starting fresh evaluation for {file_path} with {len(df)} rows for size {size}")
            existing_results = []
            start_index = 0

        results = existing_results

        for idx, row in df.iterrows():
            if idx < start_index:
                continue

            print(f"Processing row {idx + 1} of {len(df)} for size {size}")
            eval_result = await evaluate_outputs(row, size)

            if eval_result is None:
                print(f"Row {idx + 1} removed due to invalid scores.")
                continue

            scores, labels = eval_result

            criterion_results = {
                'input': row['input'],
                f'distill_run_{size}': row[f'distill_run_{size}'],
                f'vanilla_run_{size}': row[f'vanilla_run_{size}']
            }

            for criterion in criteria:
                criterion_score = scores[criterion]
                method_1, method_2 = labels[criterion]
                
                if criterion_score == 1:
                    preferred_method = method_1
                else:
                    preferred_method = method_2
                
                criterion_results[criterion] = 1 if preferred_method == 'Llama' else 0
                criterion_results[f'{criterion}_method'] = preferred_method

            results.append(criterion_results)

            if (idx + 1) % save_interval == 0 or idx == len(df) - 1:
                result_df = pd.DataFrame(results)
                result_df.to_csv(result_file_path, mode='w', header=True, index=False)
                print(f"Progress saved after {idx + 1} rows to {result_file_path}")

async def main():
    with ThreadPoolExecutor() as executor:
        for file_path in INPUT_FILES:
            if not os.path.exists(file_path):
                print(f"Warning: File not found - {file_path}")
                continue
                
            print(f"Starting evaluation for {file_path}")
            
            try:
                await evaluate_dataset(
                    file_path=file_path,
                    save_interval=SAVE_INTERVAL
                )
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
if __name__ == "__main__":
    asyncio.run(main())