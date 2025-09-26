# Personalised Learning System

This project is an AI-powered Personalized Learning System that uses a Retrieval-Augmented Generation (RAG) architecture to provide a dynamic and interactive educational experience. It is built with Python, LangGraph, and FastAPI.

## 🚀 Features

- **Dynamic Content Generation:** Automatically generates educational content in Markdown format based on a given topic and user profile.
- **Stateful Learning Graph:** Utilizes LangGraph to manage a complex, multi-step workflow for retrieving, grading, and generating information.
- **RAG Architecture:** Employs a Retrieval-Augmented Generation pipeline to ensure content is relevant and contextually accurate by first fetching information from a knowledge base.
- **Modular Design:** The code is organized into distinct modules for data, database interaction, logic, and models, making it easy to understand and extend.

## Core Technologies

- **Backend:** FastAPI
- **AI Orchestration:** LangGraph & LangChain
- **Database:** ChromaDB for vector storage
- **LLMs:** Designed to be used with models like Google Gemini or OpenAI.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set API Keys:**
    Create a `.env` file in the root directory and add your API keys:
    ```
    GOOGLE_API_KEY="YOUR_API_KEY"
    ```

3.  **Execute the Script:**
    Run the main script from your terminal:
    ```bash
    python main.py
    ```

4.  **Check the Output:**
    The script will generate two files:
    - `generated_content.md`: Contains the generated educational content.
    - `learning_state.json`: Contains the final state of the learning graph.
