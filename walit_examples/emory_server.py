# server.py

import uvicorn
import os
from fastapi import FastAPI
from langchain_core.runnables import RunnableLambda
from langserve import add_routes
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# 1. Define the prompt processing logic
def emory_logic(prompt_input: dict) -> dict:
    """
    Custom logic that receives a prompt and returns a response.
    This is the function Emory will replace with their custom model/logic.
    """
    user_prompt = prompt_input.get("input", "")
    
    # Example logic: add a prefix to the response
    response_text = f"[Response from Emory]: Your prompt was '{user_prompt}'"
    
    return {"output": response_text}

# 2. Create the FastAPI app
app = FastAPI(
    title="LangServe Server for Emory",
    version="1.0",
    description="An API server to expose a custom Emory model.",
)

# 3. Add the LangServe route
add_routes(
    app,
    RunnableLambda(emory_logic),
    path="/emory-logic",
)

# 4. Start the server using host and port from the .env file
if __name__ == "__main__":
    server_host = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port = int(os.getenv("SERVER_PORT", 8000))
    print(f"Starting server on {server_host}:{server_port}")
    uvicorn.run(app, host=server_host, port=server_port)