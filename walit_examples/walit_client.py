# client.py

import glob
import random
import os
from langserve import RemoteRunnable
from dotenv import load_dotenv

def get_random_prompt_from_txt(folder_path="."):
    """
    Finds all .txt files in the specified folder, chooses one randomly,
    and returns its content.
    """
    try:
        # Find all files ending with .txt
        prompt_files = glob.glob(os.path.join(folder_path, "*.txt"))
        if not prompt_files:
            return "No prompt files (.txt) found."
            
        # Choose a random file
        random_file = random.choice(prompt_files)
        print(f"Selected prompt file: {os.path.basename(random_file)}")
        
        # Read and return the content
        with open(random_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        return f"Error reading prompt: {e}"

# 1. Load environment variables from .env file
load_dotenv()

# 2. Emory server URL (read from .env file)
EMORY_SERVER_URL = os.getenv("EMORY_SERVER_URL")

if not EMORY_SERVER_URL:
    print("Error: EMORY_SERVER_URL environment variable not set in the .env file.")
else:
    try:
        # 3. Initialize the client for the remote service
        emory_service = RemoteRunnable(EMORY_SERVER_URL)

        # 4. Get a random prompt
        prompt_content = get_random_prompt_from_txt("examples")
        print(f"\n--- Sending Prompt ---\n{prompt_content}\n----------------------")

        # 5. Send the prompt to the server and await the response
        response = emory_service.invoke({"input": prompt_content})

        # 6. Print the response
        print(f"\n--- Response from Emory Server ---\n{response.get('output')}\n--------------------------------")

    except Exception as e:
        print(f"\nError: Could not connect to the Emory LangServe server at '{EMORY_SERVER_URL}'.")
        print("Please ensure the server is running.")
        print(f"Details: {e}")