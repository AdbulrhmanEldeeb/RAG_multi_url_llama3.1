import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# Load environment variables (API keys, tokens, etc.)
load_dotenv()
hf_token = os.getenv('HF_TOKEN')  # Hugging Face token for embedding models
groq_api_key = os.getenv('GROQ_API_KEY')  # Groq API key for the language model
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token  # Set environment variable for Hugging Face token
os.environ['GROQ_API_KEY'] = groq_api_key  # Set environment variable for Groq API key

# Streamlit app layout settings
icon_image = 'app_data/my_logo_2.png'
st.set_page_config(page_title='RAG Chat with Urls', layout='wide', page_icon=icon_image)  # Set page title and icon
logo_image = 'app_data/my_logo.png'
st.sidebar.image(logo_image)  # Display logo on the sidebar
st.sidebar.markdown('First put urls then hit \'Add URLs\' and wait a bit until urls being processed and ask questions.')

# Function to load the language model (Groq's LLaMA-3.1-70b)
def load_llm():
    """Load the Language Model."""
    try:
        # Load the model using the provided Groq API key
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")
        return llm
    except Exception as e:
        # Display error if loading fails
        st.error(f"Failed to load LLM: {str(e)}")
        return None

# Load the LLM once to avoid reloading every time the user interacts
st.session_state.llm = load_llm()
st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Split text into chunks for efficient processing

# Function to initialize the retrieval chain using the provided URLs
def initialize_chain(urls):
    """Initialize the retrieval chain with provided URLs."""
    try:
        st.session_state.all_documents = []  # List to store all documents from URLs

        # Load documents from all provided URLs
        for url in urls:
            loaders = UnstructuredURLLoader(urls=[url])
            st.session_state.data = loaders.load()  # Load data from URL

            # If documents are successfully loaded, append their content
            for content in st.session_state.data:
                if content.page_content:
                    st.session_state.all_documents.append(Document(page_content=content.page_content, metadata={"source": url}))

        # If no documents were loaded, display an error
        if not st.session_state.all_documents:
            st.error("No content found in the provided URLs.")
            return None

        # Initialize the HuggingFace embedding model
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.all_documents = st.session_state.text_splitter.split_documents(st.session_state.all_documents)  # Split documents for efficient vectorization

        # Create the FAISS index from all documents for retrieval
        st.session_state.vector_index = FAISS.from_documents(st.session_state.all_documents, st.session_state.embeddings)

        # Create the RetrievalQAWithSourcesChain using the LLM and the FAISS retriever
        chain = RetrievalQAWithSourcesChain.from_chain_type(llm=st.session_state.llm, retriever=st.session_state.vector_index.as_retriever())
        return chain
    except Exception as e:
        # Handle errors during chain initialization
        st.error(f"An error occurred during chain initialization: {str(e)}")
        return None

# Function to retrieve the answer based on the user query and initialized chain
def retrieve_answer(query, chain):
    """Retrieve the answer using the provided query and chain."""
    try:
        result = chain({'question': query}, return_only_outputs=True)  # Retrieve answer from chain using the query
        return result
    except Exception as e:
        st.error(f"An error occurred during retrieval: {str(e)}")  # Display error if retrieval fails
        return None

# User interface inputs
st.title('Retrieval-Augmented QA with LangChain')  # Main title
st.sidebar.header('Input')  # Sidebar header for input section

# Input fields for user query and URLs
query = st.text_input('Write your question here:')  # Input field for user query
url1 = st.sidebar.text_input('Add URL 1:')  # Input field for the first URL
url2 = st.sidebar.text_input('Add URL 2 (optional):')  # Optional second URL
url3 = st.sidebar.text_input('Add URL 3 (optional):')  # Optional third URL
add_urls = st.sidebar.button('Add URLs')  # Button to add URLs
urls = []

# If the "Add URLs" button is pressed, gather and store the provided URLs
if add_urls: 
    for url in [url1, url2, url3]:
        if url:
            urls.append(url)

st.session_state['urls'] = urls  # Store URLs in session state
empty = st.empty()  # Placeholder for displaying status messages

# Initialize or retrieve the chain state
if 'chain' not in st.session_state:
    st.session_state.chain = None  # Initialize chain if it doesn't exist in session state

# If the user adds URLs, initialize the chain
if add_urls and (url1 or url2 or url3):
    start = time.time()  # Start timer
    # Create a list of unique URLs
    new_urls = [url for url in [url1, url2, url3] if url and url not in st.session_state.urls]
    
    # Update stored URLs in session state
    if 'urls' not in st.session_state:
        st.session_state.urls = []
    
    st.session_state.urls.extend(new_urls)  # Add new URLs to the session

    empty.info('Processing URLs, wait a moment...')  # Inform user that URLs are being processed
    st.session_state.chain = initialize_chain(st.session_state.urls)  # Initialize the chain with all URLs
    if st.session_state.chain:
        end = time.time()
        st.session_state.chain = st.session_state.chain
        empty.success(f'Chain is ready. Total time to process URLs: {round((end - start) / 60, 2)} minutes.')  # Display success message with processing time
    else:
        empty.text('Failed to initialize chain with the provided URLs.')  # Inform user if chain initialization fails

# Button for submitting the query
submit_query = st.button('Submit')

# When query is submitted, retrieve the answer
if submit_query and query and st.session_state.chain:
    result = retrieve_answer(query, st.session_state.chain)  # Get result based on the query
    if result:
        answer = result.get('answer', 'No answer found.')  # Extract answer from result
        source = result.get('sources', 'No source available.')  # Extract source information
        st.write(f'**Answer:**\n{answer}')  # Display answer
        st.write(f'**Source:**\n{source}')  # Display source
    else:
        st.write('No result to display.')  # Display if no result is found
elif not query:
    st.info('Please enter a question if you added URLs')  # Prompt user to enter a question if missing
elif not st.session_state.chain:
    st.info('Please add at least one URL to initialize the chain.')  # Prompt user to add URLs if the chain isn't initialized
