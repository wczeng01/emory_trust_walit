# How to Run the Emory/Walit Application

This guide explains how to set up and run the server and client components.

### Step 1: Setup and Installation

First, you need to configure your environment and install the required Python libraries.

1.  **Create the environment file.** Copy the example configuration file to a new `.env` file. This file contains the connection details for the server.
    ```bash
    cp .env.example .env
    ```

2.  **Install dependencies.** Open your terminal in the project directory and run the following command:
    ```bash
    pip install -r requirements.txt
    ```
This only needs to be done once.

### Step 2: Start the Server

The server must be running before the client can connect to it.

1.  Open a terminal.
2.  Run the following command to start the Emory server:
    ```bash
    python emory_server.py
    ```
3.  You should see a message indicating that the server has started, like `Starting server on 0.0.0.0:8000`.
4.  **Keep this terminal window open.** The server needs to remain running.

### Step 3: Run the Client

With the server running, you can now run the client.

1.  Open a **new, separate** terminal window.
2.  Run the following command to start the Walit client:
    ```bash
    python walit_client.py
    ```

### Expected Output

When you run the client, it will:
1.  Print the name of the random prompt file it selected from the `examples` folder.
2.  Show the content of the prompt being sent.
3.  Display the response received from the Emory server.

### To Stop the Application

1.  Go back to the first terminal window (the one running `emory_server.py`).
2.  Press `Ctrl+C` to stop the server.
