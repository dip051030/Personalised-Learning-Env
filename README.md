# Personalised Learning System

This project is a sophisticated, AI-powered Personalized Learning System designed to provide students with a dynamic and interactive educational experience. It leverages a modular, multi-LLM architecture with a Retrieval-Augmented Generation (RAG) pipeline to deliver tailored content, answer questions, and guide users through complex topics.

While it can support various Large Language Models, it is pre-configured with helper functions for **Google Gemini**, **Groq**, and **DeepSeek**.

## ⚙️ System Architecture

The system follows a Retrieval-Augmented Generation (RAG) pipeline orchestrated by a LangGraph state machine:

1.  **Data Ingestion & Processing**: Educational content is sourced from local JSON files located in the `data/lessons/` directory. The `db/loader.py` script is responsible for loading this data.

2.  **Embedding & Vector Storage**: The loaded content is processed by `db/vector_db.py`, which uses a sentence-transformer model to convert the text into vector embeddings. These embeddings are then stored in a local ChromaDB vector database for efficient similarity searching.

3.  **Graph Execution**: The main entry point, `main.py`, initializes a predefined user profile and topic. It then invokes the stateful graph defined in `nodes.py`.
    *   **State Initialization**: The graph starts with an initial state containing the user data and the target topic.
    *   **Retriever Node**: This node takes the topic and queries the ChromaDB to find the most relevant document chunks.
    *   **Grader Node**: The retrieved documents are evaluated for their relevance to the query. The graph can decide to end the process if no relevant information is found.
    *   **Generator Node**: If relevant documents are found, this node uses a Large Language Model to synthesize a detailed lesson or explanation. The `models/llm_models.py` file contains functions to initialize different LLMs (e.g., `get_gemini_model`).

4.  **Output Generation**: The `utils/utils.py` script contains helper functions to save the final output. The generated lesson is saved to `generated_content.md`, and the complete final state of the graph is saved to `learning_state.json`.

## 🚀 Features

-   **Multi-LLM Support**: Easily configurable to use different Large Language Models. It includes ready-to-use functions for Google Gemini (`gemini-2.0-flash`), Groq (`llama-4-scout`), and DeepSeek (via OpenRouter).
-   **Stateful Learning Graph**: Built with LangGraph to manage a complex, multi-step workflow that mimics a reasoning process (retrieve, grade, generate).
-   **RAG Architecture**: Utilizes a Retrieval-Augmented Generation pipeline to ground the LLM's output in factual data, leading to more accurate and relevant educational content.
-   **Modular and Extensible**: The code is logically separated into modules for data, database management, models, prompts, and core logic, making it easy to modify or extend.

## 🛠️ Tech Stack

-   **AI Framework**: LangChain, LangGraph
-   **LLM Integrations**: `langchain-google-genai`, `langchain-groq`, `langchain-openai`
-   **Vector Database**: ChromaDB
-   **Embedding Models**: Sentence-Transformers
-   **Core**: Python, asyncio

## 🏁 Getting Started

### Prerequisites

-   Python 3.10+
-   Git

### Installation & Configuration

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/PersonalisedLearningSystem.git
    cd PersonalisedLearningSystem
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up API Keys:**
    Create a file named `.env` in the root directory of the project. You only need to provide the key for the LLM you intend to use.

    *   **For Google Gemini:**
        ```
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
        ```
    *   **For Groq:**
        ```
        GROQ_API_KEY="YOUR_GROQ_API_KEY"
        ```
    *   **For DeepSeek (via OpenRouter):**
        ```
        DEEPSEEK_API_KEY="YOUR_OPENROUTER_API_KEY"
        ```

## 🖥️ Usage

The project is run as a command-line script.

1.  **Customize the Input (Optional):**
    Open the `main.py` file and modify the `user_data` dictionary to change the user profile or the learning `topic`.

2.  **Execute the script:**
    ```bash
    python main.py
    ```

3.  **Check the Output:**
    The script will generate two files:
    -   `generated_content.md`: The final, formatted educational content.
    -   `learning_state.json`: A JSON file containing the complete state of the graph at the end of the run.

## 📂 Project Structure

```
/
├── data/                 # Contains raw JSON data for lessons.
├── db/                   # Manages the ChromaDB vector database and data loading.
│   ├── loader.py
│   └── vector_db.py
├── models/               # Handles initialization for different LLMs and embedding models.
│   ├── llm_models.py     # <== Key file for configuring which LLM to use.
│   └── ...
├── nodes.py              # Defines the nodes and edges of the LangGraph state machine.
├── prompts/              # Stores prompt templates for the LLMs.
├── scrapper/             # Utilities for scraping web content (if needed).
├── utils/                # Helper functions, including saving output files.
├── main.py               # <== Main entry point to run the script.
├── requirements.txt      # Project dependencies.
├── schemas.py            # Pydantic schemas for data validation.
└── README.md             # This file.
```