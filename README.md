# Different Types of RAG

This project implements a Streamlit application that allows users to interact with a PDF knowledge base using different Retrieval-Augmented Generation (RAG) techniques.

## Features

*   **Multiple RAG Strategies:** Explore and compare different RAG behaviors:
    *   **Corrective RAG:** Runs multiple passes where the LLM validates its generated answer against retrieved documents. If contradictions or unsupported claims are found, it attempts to correct or refine the answer. Great for enhancing factual precision and reducing hallucinations.
    *   **Agentic RAG:** Uses the LLM as an agent capable of planning, reasoning, and calling external tools (like a retriever) based on the query. Best for multi-step logic, dynamic decisions, and sophisticated problem-solving.
    *   **Adaptive RAG:** Adapts its retrieval strategy or answer generation based on factors like chat history, user preferences, or the perceived complexity/type of the query. Useful for providing more personalized and context-aware answers.
    *   **Hybrid RAG:** Combines adaptiveness, validation, and agentic tool use for complex workflows. It can dynamically switch between strategies, offering a robust and flexible solution for diverse information retrieval tasks.
*   **Flexible Knowledge Base:** Use the default PDFs or upload your own.
*   **Easy to Use:** Simple Streamlit interface for interacting with the RAG system.

## Getting Started

### Prerequisites

*   Python 3.10+
*   An API key for Google Gemini

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/priyank766/different-types-of-rag.git
    cd different-types-of-rag
    ```
2.  **Install dependencies:**
    ```bash
    uv pip install -r requirements.txt
    ```
3.  **Set up your environment:**
    Create a `.env` file in the root directory and add your Google Gemini API key:
    ```
    GOOGLE_API_KEY="your-api-key"
    ```

## Usage

1.  **Ingest the data:**
    Before running the app for the first time, you need to process the PDFs and create a vector index.
    
    --data->paths of docs you want to add 
    
    ```bash
    python ingest.py
    ```
2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

## Shoutouts

This project uses content from the following excellent books as its default knowledge base:

*   **Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow** by Aurélien Géron
*   **Hands-On Large Language Models** by Jay Alammar & Maarten Grootendorst

A huge thank you to the authors for their invaluable contributions to the machine learning and NLP communities.
