# Retrieval-Augmented Question Answering (RAG) with URLs

This project is a **Retrieval-Augmented Question Answering (QA)** system built with **LangChain** and **Streamlit**. The app allows users to input URLs, extracts the content from the pages, and enables question-answering based on that content using a language model. The application is powered by **LangChain** for document retrieval and **Hugging Face** embeddings for semantic search.

## Features

- Allows users to input up to 3 URLs.
- Loads content from the URLs and splits them into manageable chunks.
- Indexes the content using FAISS and HuggingFace embeddings for retrieval.
- Answers questions based on the retrieved content using the **Groq Chat** model (`llama-3.1-70b-versatile`).
- Displays both the answer and the source from where the content was retrieved.
- Built with **Streamlit** for an interactive user interface.

## Requirements

- Python 3.7+
- Hugging Face account and token
- Groq API key for language model
- Streamlit for the user interface

### Python Libraries

The following Python libraries are required:

- `streamlit`
- `langchain`
- `langchain_groq`
- `faiss-cpu`
- `huggingface_hub`
- `python-dotenv`
- `sentence-transformers`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-url.git
2. Install the required libraries:

    pip install -r requirements.txt
3. Set up your environment variables by creating a .env file in the root directory:

    touch .env
Add the following keys to the .env file:

HF_TOKEN=your_huggingface_token

GROQ_API_KEY=your_groq_api_key
4. Run the Streamlit app:

    streamlit run app.py
## How to Use
Add up to 3 URLs in the sidebar and click the Add URLs button.
Wait for the URLs to be processed (this might take a few minutes depending on the size of the content).
Once processed, enter a question in the text input field and click Submit to get an answer.
The application will display the answer and the sources from which the content was retrieved.
