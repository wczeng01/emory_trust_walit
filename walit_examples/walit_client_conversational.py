# walit_client_conversational.py

import os
import sys
from langserve import RemoteRunnable
from dotenv import load_dotenv

def get_initial_prompt(file_path):
    """Reads the initial prompt from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"Error: Prompt file not found at '{file_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading prompt file: {e}")
        sys.exit(1)

# 1. Load environment variables
load_dotenv()
EMORY_SERVER_URL = os.getenv("EMORY_SERVER_URL")

if not EMORY_SERVER_URL:
    print("Error: EMORY_SERVER_URL environment variable not set.")
    sys.exit(1)

# 2. Get the initial prompt file from command-line arguments
if len(sys.argv) > 1:
    initial_prompt_file = sys.argv[1]
else:
    print("Usage: python walit_client_conversational.py <path_to_initial_prompt.txt>")
    sys.exit(1)

# 3. Initialize client and load initial prompt
try:
    emory_service = RemoteRunnable(EMORY_SERVER_URL)
    conversation_history = get_initial_prompt(initial_prompt_file)
    print(f"--- Initial Prompt from {os.path.basename(initial_prompt_file)} ---\n{conversation_history}\n----------------------")

    # 4. Send the initial prompt to the server
    response = emory_service.invoke({"input": conversation_history})
    ai_response = response.get('output', 'No response from server.')
    print(f"\n--- Initial Response from Emory Server ---\n{ai_response}\n--------------------------------")

    # 5. Append AI response to history
    conversation_history += f"\nAI: {ai_response}"

    # 6. Start interactive conversation loop
    print("\n--- Starting Interactive Conversation ---")
    print("Type 'exit' or 'quit' to end the conversation.")
    while True:
        user_input = input("Human: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # Append user message to history
        conversation_history += f"\nHuman: {user_input}"
        
        # Send updated history to the server
        response = emory_service.invoke({"input": conversation_history})
        ai_response = response.get('output', 'No response from server.')
        
        # Print AI response and append to history
        print(f"AI: {ai_response}")
        conversation_history += f"\nAI: {ai_response}"

except Exception as e:
    print(f"\nError: Could not connect to the Emory LangServe server at '{EMORY_SERVER_URL}'.")
    print("Please ensure the server is running.")
    print(f"Details: {e}")
