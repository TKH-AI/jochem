import os
import pandas as pd
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time
import re
import requests.exceptions

# Constants
BASE_DIR = '/Users/jochem/Desktop/thesis-code-methods/evaluation_llm/llama/generated_answers'
MEDIA_TYPES = ['x', 'linkedin', 'press_release']
SIZES = [12, 100]
OUTPUT_DIR = '/Users/jochem/Desktop/thesis-code-methods/evaluation_llm/llama/rated_examples'

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def generate_rating_prompt(source, output):
    prompt = (f"Please rate the following marketing copy on a scale of 1-10 based on overall quality, "
             f"considering relevance, attractiveness, fluency and style:\n\n"
             f"Source text: {source}\n"
             f"Marketing copy: {output}\n\n"
             f"Provide only a number from 1-10, nothing else.")
    return prompt

def get_rating(source, output, timeout=300):
    prompt = generate_rating_prompt(source, output)
    max_attempts = 3
    attempt = 0
    rating = None
    
    while attempt < max_attempts and rating is None:
        try:
            print(f"Attempt {attempt + 1} to get rating...")
            start_time = time.time()
            response = llm.invoke(prompt)
            end_time = time.time()
            
            if end_time - start_time > timeout:
                raise requests.exceptions.ReadTimeout
                
            response_text = response.content
            print(f"Received response: {response_text}")
            
            rating_match = re.search(r'\b([1-9]|10)\b', response_text)
            
            if rating_match:
                rating = int(rating_match.group())
                print(f"Extracted rating: {rating}")
            else:
                print(f"Error: No valid rating found in response. Retrying...")
                
        except requests.exceptions.ReadTimeout:
            print(f"Timeout occurred, attempt {attempt + 1}. Retrying...")
            
        attempt += 1
        
    return rating

def process_dataset():
    all_results = []
    
    for media_type in MEDIA_TYPES:
        print(f"\nProcessing media type: {media_type}")
        csv_file = f"{BASE_DIR}/{media_type}/generation_results_full_first_run_deterministic.csv"
        
        if not os.path.exists(csv_file):
            print(f"Warning: File not found - {csv_file}")
            continue
            
        df = pd.read_csv(csv_file)
        df = df.head(100)  # Only process first 100 rows
        
        for idx, row in df.iterrows():
            print(f"\nProcessing row {idx + 1}/100 for {media_type}")
            source = row['input']
            
            for size in SIZES:
                print(f"\nEvaluating size {size}")
                distill_output = row[f'distill_run_{size}']
                vanilla_output = row[f'vanilla_run_{size}']
                
                print("Rating distill output...")
                distill_rating = get_rating(source, distill_output)
                print("Rating vanilla output...")
                vanilla_rating = get_rating(source, vanilla_output)
                
                print(f"Ratings - Distill: {distill_rating}, Vanilla: {vanilla_rating}")
                
                if distill_rating and vanilla_rating and distill_rating > vanilla_rating:
                    print("Found better performing distill example!")
                    result = {
                        'media_type': media_type,
                        'size': size,
                        'source': source,
                        'distill_output': distill_output,
                        'vanilla_output': vanilla_output,
                        'distill_rating': distill_rating,
                        'vanilla_rating': vanilla_rating
                    }
                    all_results.append(result)
                    
            # Save intermediate results every 10 rows
            if (idx + 1) % 10 == 0:
                print(f"\nSaving intermediate results after {idx + 1} rows...")
                interim_df = pd.DataFrame(all_results)
                interim_file = os.path.join(OUTPUT_DIR, f'rated_examples_interim_{media_type}.csv')
                interim_df.to_csv(interim_file, index=False)
                    
    # Save final results to CSV
    print("\nSaving final results...")
    results_df = pd.DataFrame(all_results)
    output_file = os.path.join(OUTPUT_DIR, 'rated_examples.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    print("Starting evaluation process...")
    process_dataset()
    print("Evaluation complete!")
