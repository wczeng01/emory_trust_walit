# client_example.py
from langserve import RemoteRunnable
from langchain_core.output_parsers import StrOutputParser

# Connect to your running server:
model = RemoteRunnable("http://localhost:8000/agent/")

# Build whatever chain you want; here we just invoke directly.
payload = {
    "session_id": "demo-001",
    "message": "Hi, I had a bad dream last night and feel on edge.",
    "history": [{"role": "assistant", "content": "Hello, I'm here to listen."}],
    "config": {"temperature": 0.2}
}

# If your runnable returns TrustOutput, you can access fields from the dict.
resp = model.invoke(payload)
print(resp)  # {"text": "...", "state": {...}}
