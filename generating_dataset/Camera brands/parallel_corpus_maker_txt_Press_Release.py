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

async def load_file(file_path):
    async with aiofiles.open(file_path, 'r') as f:
        return await f.read()

# Function to load all datasheet descriptions for a specific index (e.g., 1.1, 1.2, etc.)
async def load_all_datasheet_descriptions(folder_path, index):
    tasks = []
    i = 1
    while True:
        file_path = os.path.join(folder_path, f'datasheet_description_{index}.{i}.txt')
        if not os.path.exists(file_path):
            break
        # Append the coroutine (do not await here)
        tasks.append(load_file(file_path))
        i += 1
    return tasks  # Return a list of coroutines (not the results)


async def load_data_async(folder_prefix, item_type):
    tasks = []
    for i in range(1, 7):
        folder_path = f'/Users/jochem/Desktop/mijn_project2/Rationales/Press Release/{folder_prefix} {i}'
        press_release_file = os.path.join(folder_path, f'{item_type}_{i}.txt')

        if os.path.exists(press_release_file):
            # Load all datasheet descriptions for the given folder and index
            datasheet_description_tasks = await load_all_datasheet_descriptions(folder_path, i)
            if datasheet_description_tasks:
                # Create a list of tasks for each datasheet description and the press release
                tasks.append(asyncio.gather(load_file(press_release_file), *datasheet_description_tasks))

    data = []
    for i, results in enumerate(await asyncio.gather(*tasks)):
        press_release_content = results[0]
        datasheet_contents = results[1:]
        # Combine all descriptions into one string
        combined_descriptions = "\n\n".join([desc.strip() for desc in datasheet_contents])
        data.append((f'datasheet_descriptions_{i+1}', combined_descriptions.strip(), press_release_content.strip()))
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

        # Shuffle the data and select 1 random example for each individual prompt
        random.shuffle(data)  # Shuffle the full dataset
        selected_data = data[0]  # Select only 1 random example after shuffling

        ds, ra = selected_data[1], selected_data[2]  # Unpack the datasheet description and rationale/press release
        
        example = f"Datasheet Description: {ds}\n{content_type.capitalize()}: {ra}"

        if content_type == "press_release":
            full_prompt = (
                "You are a content creator generating press releases for various camera models. "
                "In the provided example, a datasheet description is used to create a press release. "
                "Based on this example, generate a new press release for the following camera model description. "
                "**Do not include any introductory sentences like 'Here is a press release for a camera' or 'Press Release:'. Only output the press release content.**\n\n"
                f"**Example:**\n{example}\n\n"
                "Now, focus on the **new camera model** in the Datasheet Description below and create a press release specifically for this model, not for the example.\n\n"
                f"**New Camera Model:**\nDatasheet Description: {description}\nPress Release:"
            )

        else:
            full_prompt = (
                "You are a content creator tasked with explaining the reasoning behind press releases based on the camera datasheet description. "
                "In the provided example, a datasheet description is used to create a press release. "
                "**Do not include any introductory sentences like 'Here is a rationale for the press release'. Only output the rationale content.**\n\n"
                f"**Example:**\n{example}\n\n"
                "Now, focus on the **new camera model** in the Datasheet Description below and create a rationale for this specific model, not for the example.\n\n"
                f"**New Camera Model:**\nDatasheet Description: {description}\nRationale:"
            )


        # print(full_prompt)

        # Log the prompt
        # logger.info(f"Full prompt for index {idx}: {full_prompt[:600]}")

        # Send each prompt concurrently using executor
        task = fetch_responses_concurrently(description, full_prompt, idx, executor)
        tasks.append(task)

    # Gather all the responses for the current batch
    return await asyncio.gather(*tasks)

# Common batch processor function
async def process_batch(descriptions, data, column, content_type):
    batch_size = 1  # Ensuring batch size is 2
    checkpoint_interval = 1 # Save progress every 20 rows
    processed_count = 0

    # Executor for parallel requests
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        # Process in batches
        for i in range(3910, len(descriptions), batch_size):
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

# Process rows with LinkedIn post generation
async def process_press_release_posts():
    await process_batch(df['datasheet_description'].tolist(), press_release_data, 'Press_Release', 'press_release')

# File paths
input_file_path = '/Users/jochem/Desktop/mijn_project2/Camera brands/Final Dataset REAL ONES/dataset_Press_Release_LLaMa_GOOD14.csv'
output_file_path = '/Users/jochem/Desktop/mijn_project2/Camera brands/Final Dataset REAL ONES/dataset_Press_Release_LLaMa_GOOD15.csv'

# Load the CSV file and ensure all columns are treated as string
df = pd.read_csv(input_file_path, dtype=str)
df['Rationale'] = df['Rationale'].astype(str)
df['Press_Release'] = df['Press_Release'].astype(str)

# Load rationale and LinkedIn data asynchronously
rationale_data = asyncio.run(load_data_async('Rationale', 'rationale'))
press_release_data = asyncio.run(load_data_async('Press Release', 'press_release'))

# # First process all Rationale data
asyncio.run(process_rationales())

# Run the processing function for LinkedIn posts
asyncio.run(process_press_release_posts())

# import os
# import pandas as pd
# import asyncio
# import logging
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from concurrent.futures import ThreadPoolExecutor
# import random
# import emoji
# import aiofiles

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load environment variables
# load_dotenv()

# # Initialize LLM
# CHAT_LLM_URL = "http://192.168.209.204/v1"
# model = "neuralmagic/Meta-Llama-3.1-70B-Instruct-FP8-dynamic"

# llm = ChatOpenAI(
#     model=model,
#     openai_api_key="EMPTY",
#     openai_api_base=CHAT_LLM_URL,
#     temperature=0.2,
#     streaming=True,
# )

# # Function to convert emojis to Unicode
# def convert_emojis_to_unicode(text):
#     return emoji.demojize(text)

# async def load_file(file_path):
#     async with aiofiles.open(file_path, 'r') as f:
#         return await f.read()

# # Function to load all datasheet descriptions for a specific index (e.g., 1.1, 1.2, etc.)
# async def load_all_datasheet_descriptions(folder_path, index):
#     tasks = []
#     i = 1
#     while True:
#         file_path = os.path.join(folder_path, f'datasheet_description_{index}.{i}.txt')
#         if not os.path.exists(file_path):
#             break
#         tasks.append(load_file(file_path))
#         i += 1
#     return tasks  # Return a list of coroutines (not the results)

# async def load_data_async(folder_prefix, item_type):
#     tasks = []
#     for i in range(1, 7):
#         folder_path = f'/Users/jochem/Desktop/mijn_project2/Rationales/Press Release/{folder_prefix} {i}'
#         press_release_file = os.path.join(folder_path, f'{item_type}_{i}.txt')

#         if os.path.exists(press_release_file):
#             datasheet_description_tasks = await load_all_datasheet_descriptions(folder_path, i)
#             if datasheet_description_tasks:
#                 tasks.append(asyncio.gather(load_file(press_release_file), *datasheet_description_tasks))

#     data = []
#     for i, results in enumerate(await asyncio.gather(*tasks)):
#         press_release_content = results[0]
#         datasheet_contents = results[1:]
#         combined_descriptions = "\n\n".join([desc.strip() for desc in datasheet_contents])
#         data.append((f'datasheet_descriptions_{i+1}', combined_descriptions.strip(), press_release_content.strip()))
#     return data

# # Function to check if a cell is empty
# def is_cell_empty(value):
#     if pd.isna(value):
#         return True
#     elif isinstance(value, str):
#         stripped_value = value.strip()
#         return stripped_value == '' or stripped_value.lower() == 'nan'
#     return False

# # Function to fetch responses concurrently
# async def fetch_responses_concurrently(description, prompt, index, executor):
#     logger.info(f"Sending prompt for index {index}: {prompt[:100]}")

#     loop = asyncio.get_event_loop()
#     try:
#         response = await loop.run_in_executor(executor, llm.invoke, prompt)
#         response_text = response.text if hasattr(response, 'text') else str(response)

#         if response_text and response_text.strip():
#             logger.info(f"Received response for index {index}: {response_text[:100]}")
#             return (index, response_text.strip())
#         else:
#             raise ValueError(f"Empty response for description at index {index}")
#     except Exception as e:
#         logger.error(f"Error processing description at index {index}: {e}")
#         return (index, "No response generated")

# # Generate content for a batch
# async def generate_content_batch(descriptions, data, content_type, start_index, indices, executor):
#     tasks = []
#     for i, (description, idx) in enumerate(zip(descriptions, indices)):
#         random.shuffle(data)
#         selected_data = data[0]

#         ds, ra = selected_data[1], selected_data[2]

#         example = f"Datasheet Description: {ds}\n{content_type.capitalize()}: {ra}"

#         if content_type == "press_release":
#             full_prompt = (
#                 "You are a content creator generating press releases for various camera models. "
#                 "In the provided example, a datasheet description is used to create a press release. "
#                 "Generate a press release for the following camera model description.\n\n"
#                 f"Example:\n{example}\n\n"
#                 f"New Camera Model:\nDatasheet Description: {description}\nPress Release:"
#             )
#         else:
#             full_prompt = (
#                 "You are a content creator explaining the rationale behind press releases. "
#                 "Create a rationale based on the datasheet description and press release provided.\n\n"
#                 f"Example:\n{example}\n\n"
#                 f"New Camera Model:\nDatasheet Description: {description}\nRationale:"
#             )

#         task = fetch_responses_concurrently(description, full_prompt, idx, executor)
#         tasks.append(task)

#     return await asyncio.gather(*tasks)

# # Common batch processor function
# async def process_batch(descriptions, data, column, content_type):
#     batch_size = 10
#     checkpoint_interval = 1
#     processed_count = 0

#     with ThreadPoolExecutor(max_workers=batch_size) as executor:
#         for i in range(0, len(descriptions), batch_size):
#             batch_descriptions = descriptions[i:i + batch_size]
#             batch_indices = [idx for idx in range(i, i + batch_size) if is_cell_empty(df.at[idx, column])]
#             batch_descriptions = [descriptions[idx] for idx in batch_indices]

#             if not batch_descriptions:
#                 logger.info(f"All rows in this batch for {content_type.capitalize()} already contain data, skipping.")
#                 continue

#             batch_results = await generate_content_batch(batch_descriptions, data, content_type, i, batch_indices, executor)

#             for index, result in batch_results:
#                 if result != "No response generated":
#                     df.at[index, column] = result
#                 else:
#                     logger.warning(f"No content generated for {content_type} at index {index}")

#             processed_count += batch_size
#             if processed_count >= checkpoint_interval:
#                 await save_dataframe_async(output_file_path)
#                 logger.info(f"DataFrame saved after processing up to index {i + batch_size - 1}")
#                 processed_count = 0

#     await save_dataframe_async(output_file_path)
#     logger.info(f"Final save completed for {content_type.capitalize()}.")

# # Asynchronous function to save DataFrame
# async def save_dataframe_async(file_path):
#     async with aiofiles.open(file_path, 'w') as f:
#         await f.write(df.to_csv(index=False))

# # Process rows with rationale generation
# async def process_rationales():
#     await process_batch(df['datasheet_description'].tolist(), rationale_data, 'Rationale', 'rationale')

# # Process rows with press release generation
# async def process_press_release_posts():
#     await process_batch(df['datasheet_description'].tolist(), press_release_data, 'Press_Release', 'press_release')

# # File paths
# input_file_path = '/Users/jochem/Desktop/mijn_project2/Camera brands/Final Dataset REAL ONES/dataset_Press_Release_LLaMa_GOOD13.csv'
# output_file_path = '/Users/jochem/Desktop/mijn_project2/Camera brands/Final Dataset REAL ONES/dataset_Press_Release_LLaMa_GOOD14.csv'

# # Load the CSV file and ensure all columns are treated as string
# df = pd.read_csv(input_file_path, dtype=str)
# df['Rationale'] = df['Rationale'].astype(str)
# df['Press_Release'] = df['Press_Release'].astype(str)

# # Load rationale and press release data asynchronously
# rationale_data = asyncio.run(load_data_async('Rationale', 'rationale'))
# press_release_data = asyncio.run(load_data_async('Press Release', 'press_release'))

# # First process all Rationale data
# asyncio.run(process_rationales())

# # Then process the Press Release data
# asyncio.run(process_press_release_posts())
