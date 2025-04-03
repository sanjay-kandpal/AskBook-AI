import os
import logging
import uuid
import shutil

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from azure.storage.blob import BlobServiceClient

from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstore/db_faiss"

# Improved error handling for environment variables
def get_env_variable(var_name, default=None):
    value = os.getenv(var_name)
    if not value and default is None:
        raise ValueError(f"Environment variable {var_name} is not set")
    return value or default

# Load connection strings from env
try:
    AZURE_CONNECTION_STRING = get_env_variable("AZURE_STORAGE_CONNECTION_STRING")
    HF_TOKEN = get_env_variable("HUGGINGFACE_TOKEN_API")
except ValueError as e:
    logging.error(str(e))
    raise

AZURE_CONTAINER_NAME = "pdf-files"

# Configure logging with more robust settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_processing.log'),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# Blob storage setup with error handling
try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
except Exception as e:
    logger.error(f"Failed to connect to Azure Blob Storage: {e}")
    container_client = None

# Ensure directories exist
for directory in [UPLOAD_FOLDER, DATA_PATH, os.path.dirname(DB_FAISS_PATH)]:
    os.makedirs(directory, exist_ok=True)

# HuggingFace Configuration
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    """Load Language Model with improved error handling"""
    try:
        return HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.5,
            huggingfacehub_api_token=HF_TOKEN,
            model_kwargs={"max_length": 512}
        )
    except Exception as e:
        logger.error(f"Error loading LLM: {e}")
        raise

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Don't provide anything out of the given context.

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt(custom_prompt_template):
    """Create a custom prompt template"""
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_pdf_files(data_path):
    """
    Enhanced PDF loader with comprehensive error handling
    
    Args:
        data_path (str): Path to directory containing PDF files
    
    Returns:
        list: Loaded PDF documents
    """
    try:
        # Verify directory exists and is not empty
        if not os.path.exists(data_path):
            logger.error(f"Directory {data_path} does not exist")
            return []

        pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {data_path}")
            return []

        # Load PDFs
        loader = DirectoryLoader(data_path, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        
        # Log details of loaded documents
        logger.info(f"Loaded {len(documents)} pages from {len(pdf_files)} PDF files")
        for doc in documents:
            logger.info(f"Loaded document: {doc.metadata.get('source', 'Unknown')}")
        
        return documents
    
    except Exception as e:
        logger.error(f"Error loading PDF files from {data_path}: {e}")
        return []

def create_chunks(extracted_data, chunk_size=500, chunk_overlap=50):
    """
    Create text chunks from documents
    
    Args:
        extracted_data (list): List of documents to chunk
        chunk_size (int): Size of text chunks
        chunk_overlap (int): Overlap between chunks
    
    Returns:
        list: Chunked documents
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(extracted_data)

def get_embedding_model(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """
    Get embedding model with optional model name
    
    Args:
        model_name (str): Name of embedding model
    
    Returns:
        HuggingFaceEmbeddings: Embedding model
    """
    return HuggingFaceEmbeddings(model_name=model_name)

@app.route('/process', methods=['POST'])
def process_pdfs():
    """
    Process uploaded PDF files and store in Azure Blob Storage
    
    Returns:
        JSON response with processing status
    """
    if 'pdfFiles' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist('pdfFiles')
    
    uploaded_files = []
    processed_files = []
    blob_uploaded_files = []

    # Create a temporary directory for new uploads
    temp_upload_dir = os.path.join(DATA_PATH, f"upload_{uuid.uuid4()}")
    os.makedirs(temp_upload_dir, exist_ok=True)

    try:
        # Process each uploaded file
        for file in files:
            if not file.filename or not file.filename.endswith('.pdf'):
                continue
            
            try:
                # Secure filename
                filename = secure_filename(file.filename)
                
                # Save file to temporary directory
                local_file_path = os.path.join(temp_upload_dir, filename)
                file.seek(0)
                file.save(local_file_path)
                
                uploaded_files.append(local_file_path)
                logger.info(f"Uploaded file locally: {local_file_path}")
                
                # Upload to Azure Blob Storage
                if container_client:
                    try:
                        with open(local_file_path, "rb") as data:
                            # Generate a unique blob name to prevent overwriting
                            blob_name = f"{uuid.uuid4()}_{filename}"
                            container_client.upload_blob(name=blob_name, data=data)
                            blob_uploaded_files.append(blob_name)
                            logger.info(f"Uploaded file to blob storage: {blob_name}")
                    except Exception as blob_error:
                        logger.error(f"Failed to upload {filename} to blob storage: {blob_error}")
                
            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
                return jsonify({"error": f"Failed to process {filename}"}), 500

        # Verify files were uploaded
        if not uploaded_files:
            return jsonify({"error": "No PDF files uploaded"}), 400

        # Process only the newly uploaded files
        documents = load_pdf_files(temp_upload_dir)
        
        if not documents:
            return jsonify({"error": "No documents could be loaded"}), 400

        # Create text chunks
        text_chunks = create_chunks(documents)
        logger.info(f"Created {len(text_chunks)} text chunks")

        # Get embedding model
        embedding_model = get_embedding_model()

        # Load existing vector store or create new one
        try:
            # Try to load existing vector store
            db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
            
            # Add new documents to existing vector store
            db.add_documents(text_chunks)
            db.save_local(DB_FAISS_PATH)
            logger.info("Added new documents to existing vector store")
        
        except Exception as e:
            # If no existing vector store, create a new one
            logger.warning(f"Creating new vector store: {e}")
            db = FAISS.from_documents(text_chunks, embedding_model)
            db.save_local(DB_FAISS_PATH)

        # Track processed files
        processed_files = [os.path.basename(f) for f in uploaded_files]

        return jsonify({
            "status": "success",
            "message": "PDFs processed and added to vector store",
            "processed_files": processed_files,
            "blob_uploaded_files": blob_uploaded_files
        }), 200

    except Exception as e:
        logger.error(f"Unexpected error in processing: {e}")
        return jsonify({"error": "Failed to process documents", "details": str(e)}), 500
    
    finally:
        # Clean up temporary upload directory
        try:
            shutil.rmtree(temp_upload_dir)
        except Exception as e:
            logger.error(f"Error cleaning up temp directory: {e}")
            
@app.route('/query', methods=['POST'])
def query_documents():
    """
    Query processed documents
    
    Returns:
        JSON response with query results
    """
    try:
        # Extract data from request
        data = request.json
        user_query = data.get('query', '')
        session_id = data.get('session_id', None)
        
        # If session_id is provided, use that specific vector store
        if session_id:
            vector_store_path = os.path.join("vectorstore", f"db_faiss_{session_id}")
            if not os.path.exists(vector_store_path):
                return jsonify({"error": f"Vector store for session {session_id} not found"}), 404
        else:
            # If no session_id, try to find the most recent vector store from registry
            registry_path = os.path.join("vectorstore", "registry.txt")
            if os.path.exists(registry_path):
                with open(registry_path, "r") as registry_file:
                    lines = registry_file.readlines()
                    if lines:
                        # Get the last entry in the registry
                        last_line = lines[-1].strip()
                        session_id, vector_store_path = last_line.split(',')
                    else:
                        return jsonify({"error": "No vector stores available"}), 404
            else:
                return jsonify({"error": "Vector store registry not found"}), 404
        
        # Load embedding model
        embedding_model = get_embedding_model()
        
        # Load the specified FAISS database
        db = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)
        
        # Create QA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=load_llm(HUGGINGFACE_REPO_ID),
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={'k': 3}),
            return_source_documents=True,
            chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
        )
        
        # Process query
        response = qa_chain({'query': user_query})
        
        return jsonify({
            "result": response["result"],
            "source_documents": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                } for doc in response["source_documents"]
            ],
            "session_id": session_id
        }), 200
    
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/sessions', methods=['GET'])
def list_sessions():
    """
    List all available session IDs and their vector store paths
    
    Returns:
        JSON response with list of sessions
    """
    try:
        registry_path = os.path.join("vectorstore", "registry.txt")
        sessions = []
        
        if os.path.exists(registry_path):
            with open(registry_path, "r") as registry_file:
                for line in registry_file:
                    line = line.strip()
                    if line:
                        session_id, vector_store_path = line.split(',')
                        
                        # Check if the vector store still exists
                        if os.path.exists(vector_store_path):
                            sessions.append({
                                "session_id": session_id,
                                "vector_store_path": vector_store_path,
                                "created_at": os.path.getctime(vector_store_path)
                            })
        
        return jsonify({
            "sessions": sessions,
            "count": len(sessions)
        }), 200
    
    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    try:
        app.run(host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Failed to start Flask application: {e}")