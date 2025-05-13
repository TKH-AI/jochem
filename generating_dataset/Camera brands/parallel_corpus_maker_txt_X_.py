import os
import pandas as pd
import asyncio
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from concurrent.futures import ThreadPoolExecutor
import random
import emoji
import aiofiles
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize LLM
CHAT_LLM_URL = "http://192.168.209.204/v1"
model = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8-dynamic"

llm = ChatOpenAI(
    model=model,
    openai_api_key="EMPTY",
    openai_api_base=CHAT_LLM_URL,
    temperature=0.2,
    streaming=True,
)

# Function to convert emojis to Unicode
def convert_emojis_to_unicode(text):
    return emoji.demojize(text)

# Async function to read files
async def load_data_async(folder_prefix, item_type):
    data = []
    for i in range(1, 7):
        folder_path = f'/Users/jochem/Desktop/mijn_project2/Rationales/X/{folder_prefix} {i}'
        datasheet_file = os.path.join(folder_path, f'datasheet_description_{i}.txt')
        content_file = os.path.join(folder_path, f'{item_type}_{i}.txt')
        if os.path.exists(datasheet_file) and os.path.exists(content_file):
            async with aiofiles.open(datasheet_file, 'r') as dsf, aiofiles.open(content_file, 'r') as cf:
                datasheet = convert_emojis_to_unicode(await dsf.read())
                content = convert_emojis_to_unicode(await cf.read())
                data.append((f'datasheet_description_{i}.txt', datasheet.strip(), content.strip()))
    return data

# Function to check if a cell is empty
def is_cell_empty(value):
    if pd.isna(value):
        return True
    elif isinstance(value, str):
        stripped_value = value.strip()
        return stripped_value == '' or stripped_value.lower() == 'nan'
    return False

# Common function to fetch responses concurrently
async def fetch_responses_concurrently(description, prompt, index, executor):
    logger.info(f"Sending prompt for index {index}: {prompt[:100]}")

    # Use executor to run the blocking LLM call concurrently
    loop = asyncio.get_event_loop()
    try:
        # Execute the LLM model call for each individual prompt asynchronously
        response = await loop.run_in_executor(executor, llm.invoke, prompt)
        
        response_text = response.text if hasattr(response, 'text') else str(response)

        if response_text and response_text.strip():
            logger.info(f"Received response for index {index}: {response_text[:100]}")  # Log part of the response
            return (index, response_text.strip())
        else:
            raise ValueError(f"Empty response for description at index {index}")
    except Exception as e:
        logger.error(f"Error processing description at index {index}: {e}")
        return (index, "No response generated")

# Generate content for a batch (send prompts separately but in parallel)
async def generate_content_batch(descriptions, data, content_type, start_index, indices, executor):
    tasks = []
    for i, (description, idx) in enumerate(zip(descriptions, indices)):  # Now we also track index in the original DataFrame

        # Shuffle the data for each individual prompt
        random.shuffle(data)  # This will shuffle the data before every prompt generation

        examples = "\n\n".join([f"Datasheet Description: {ds}\n{content_type.capitalize()}: {ra}"
                                for _, ds, ra in data])

        if content_type == "x":
            full_prompt = (
                "You are a content creator generating X posts for various camera models. "
                "Based on the six provided examples, generate a new X post for the following camera model description. "
                "**Do not include any introductory sentences like 'Here is a X post for camera' or 'X:'. Only output the X post content.**\n\n"
                f"**Examples:**\n{examples}\n\n"
                "Now, focus on the **new camera model** in the Datasheet Description below and create a X post specifically for this model, not for any of the examples.\n\n"
                f"**New Camera Model:**\nDatasheet Description: {description}\nX Post:"
            )
        else:
            full_prompt = (
                "You are a content creator tasked with explaining the reasoning behind X posts based on the camera datasheet description. "
                "Based on the six provided examples, generate a new rationale for the following camera model description. "
                "**Do not include any introductory sentences like 'Here is a rationale for the X post'. Only output the rationale content.**\n\n"
                f"**Examples:**\n{examples}\n\n"
                "Now, focus on the **new camera model** in the Datasheet Description below and create a rationale for this specific model, not for any of the examples.\n\n"
                f"**New Camera Model:**\nDatasheet Description: {description}\nRationale:"
            )
        
        print(full_prompt)
            
        # Log the prompt
        # logger.info(f"Full prompt for index {idx}: {full_prompt[:600]}")

        # Send each prompt concurrently using executor (not combined into one prompt)
        task = fetch_responses_concurrently(description, full_prompt, idx, executor)  # Pass the original DataFrame index
        tasks.append(task)

    # Gather all the responses for the current batch
    return await asyncio.gather(*tasks)

# Common batch processor function
async def process_batch(descriptions, data, column, content_type):
    batch_size = 1  # Ensuring batch size is 5
    checkpoint_interval = 1 # Save progress every 20 rows
    processed_count = 0

    # Executor for parallel requests
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        # Process in batches
        for i in range(3900, len(descriptions), batch_size):
            batch_descriptions = descriptions[i:i + batch_size]
            logger.info(f"Processing batch for {content_type.capitalize()} from index {i} to {min(i + batch_size, len(descriptions)) - 1}")

            # Skip rows that already have content in the column
            batch_indices = [idx for idx in range(i, i + batch_size) if is_cell_empty(df.at[idx, column])]
            if len(batch_indices) != len(batch_descriptions):  # If some rows were skipped, log it
                logger.info(f"Skipped {len(batch_descriptions) - len(batch_indices)} rows that already had data.")

            batch_descriptions = [descriptions[idx] for idx in batch_indices]  # Filter the descriptions to include only those with empty cells

            if not batch_descriptions:
                logger.info(f"All rows in this batch for {content_type.capitalize()} already contain data, skipping.")
                continue

            # Generate content for the batch (send prompts separately but concurrently)
            batch_results = await generate_content_batch(batch_descriptions, data, content_type, i, batch_indices, executor)

            # Update the DataFrame with the generated results
            for index, result in batch_results:
                if result != "No response generated":
                    df.at[index, column] = result
                else:
                    logger.warning(f"No content generated for {content_type} at index {index}")

            processed_count += batch_size
            if processed_count >= checkpoint_interval:
                await save_dataframe_async(output_file_path)
                logger.info(f"DataFrame saved after processing up to index {i + batch_size - 1}")
                processed_count = 0

    # Final save
    await save_dataframe_async(output_file_path)
    logger.info(f"Final save completed for {content_type.capitalize()}.")

# Asynchronous function to save DataFrame
async def save_dataframe_async(file_path):
    async with aiofiles.open(file_path, 'w') as f:
        await f.write(df.to_csv(index=False))

# Process rows with rationale generation
async def process_rationales():
    await process_batch(df['datasheet_description'].tolist(), rationale_data, 'Rationale', 'rationale')

# # Process rows with Twitter post generation
async def process_twitter_posts():
    await process_batch(df['datasheet_description'].tolist(), twitter_data, 'X', 'x')

# File paths
input_file_path = '/Users/jochem/Desktop/mijn_project2/Camera brands/Final Dataset REAL ONES/dataset_X_LLaMa7.csv'
output_file_path = '/Users/jochem/Desktop/mijn_project2/Camera brands/Final Dataset REAL ONES/dataset_X_LLaMa8.csv'

# Load the CSV file and ensure all columns are treated as string
df = pd.read_csv(input_file_path, dtype=str)
df['Rationale'] = df['Rationale'].astype(str)
df['X'] = df['X'].astype(str)

# Load rationale and Twitter data asynchronously
rationale_data = asyncio.run(load_data_async('Rationale', 'rationale'))
twitter_data = asyncio.run(load_data_async('X', 'tweet'))

# First process all Rationale data
asyncio.run(process_rationales())

# # Run the processing function for Twitter posts
asyncio.run(process_twitter_posts())
