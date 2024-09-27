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
hf_token = os.getenv('HF_TOKEN')
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ['HUGGINGFACEHUB_API_TOKEN'] = hf_token
os.environ['GROQ_API_KEY'] = groq_api_key

# Streamlit app layout
icon_image='app_data\my_logo_2.png'
st.set_page_config(page_title='RAG Chat with Urls', layout='wide',page_icon=icon_image)
logo_image = 'app_data\my_logo.png'
st.sidebar.image(logo_image)
st.sidebar.markdown('First put urls then hit \'Add URLs\' and wait a bit until urls being processed and ask questions.')
def load_llm():
    """Load the Language Model."""
    try:
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-70b-versatile")
        return llm
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        return None

# Load the LLM once to avoid reloading
st.session_state.llm = load_llm()
st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def initialize_chain(urls):
    """Initialize the retrieval chain with provided URLs."""
    try:
        st.session_state.all_documents = []

        # Load documents from all provided URLs
        for url in urls:
            loaders = UnstructuredURLLoader(urls=[url])
            st.session_state.data = loaders.load()

            # Check if any documents are loaded
            for content in st.session_state.data:
                if content.page_content:
                    st.session_state.all_documents.append(Document(page_content=content.page_content, metadata={"source": url}))

        if not st.session_state.all_documents:
            st.error("No content found in the provided URLs.")
            return None

        # Initialize the HuggingFace embedding model
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.all_documents = st.session_state.text_splitter.split_documents(st.session_state.all_documents)

        # Create the FAISS index from all documents
        st.session_state.vector_index = FAISS.from_documents(st.session_state.all_documents, st.session_state.embeddings)

        # Create the RetrievalQAWithSourcesChain
        chain = RetrievalQAWithSourcesChain.from_chain_type(llm=st.session_state.llm, retriever=st.session_state.vector_index.as_retriever())
        return chain 
    except Exception as e:
        st.error(f"An error occurred during chain initialization: {str(e)}")
        return None


def retrieve_answer(query, chain):
    """Retrieve the answer using the provided query and chain."""
    try:
        result = chain({'question': query}, return_only_outputs=True)
        return result
    except Exception as e:
        st.error(f"An error occurred during retrieval: {str(e)}")
        return None

# User inputs
st.title('Retrieval-Augmented QA with LangChain')
st.sidebar.header('Input')

query = st.text_input('Write your question here:')
url1 = st.sidebar.text_input('Add URL 1:')
url2 = st.sidebar.text_input('Add URL 2 (optional):')
url3 = st.sidebar.text_input('Add URL 3 (optional):')
add_urls = st.sidebar.button('Add URLs')
urls=[]
if add_urls : 
    for url in [url1,url2,url3]: 
        if url : 
            urls.append(url)

st.session_state['urls']=urls   
empty = st.empty()

# Initialize or retrieve the chain state
if 'chain' not in st.session_state:
    st.session_state.chain = None

if add_urls and (url1 or url2 or url3):
    start = time.time()
    # Create a list of URLs, ensuring they are unique
    new_urls = [url for url in [url1, url2, url3] if url and url not in st.session_state.urls]
    
    # Update the stored URLs in the session state
    if 'urls' not in st.session_state:
        st.session_state.urls = []
    
    st.session_state.urls.extend(new_urls)
    
    empty.info('Processing URLs, wait a moment...')
    st.session_state.chain = initialize_chain(st.session_state.urls)  # Initialize with all accumulated URLs
    if st.session_state.chain:
        end = time.time()
        st.session_state.chain = st.session_state.chain
        empty.success(f'Chain is ready. Total time to process URLs: {round((end - start) / 60, 2)} minutes.')
    else:
        empty.text('Failed to initialize chain with the provided URLs.')


submit_query = st.button('Submit')

if submit_query and query and st.session_state.chain:
    result = retrieve_answer(query, st.session_state.chain)
    if result:
        answer = result.get('answer', 'No answer found.')
        source = result.get('sources', 'No source available.')
        st.write(f'**Answer:**\n{answer}')
        st.write(f'**Source:**\n{source}')
    else:
        st.write('No result to display.')
elif not query:
    st.info('Please enter a question if you added URLs')

elif not st.session_state.chain:
    st.info('Please add at least one URL to initialize the chain.')
