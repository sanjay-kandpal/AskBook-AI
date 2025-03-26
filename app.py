import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

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

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)

# HuggingFace Configuration
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN_API")
HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

def load_llm(huggingface_repo_id):
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.5,
            huggingfacehub_api_token=HF_TOKEN,
            model_kwargs={"max_length": 512}
        )
        return llm
    except Exception as e:
        print(f"Error loading LLM: {e}")
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
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def load_pdf_files(data):
    loader = DirectoryLoader(data, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embedding_model

@app.route('/process', methods=['POST'])
def process_pdfs():
    if 'pdfFiles' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist('pdfFiles')
    for file in files:
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            os.rename(os.path.join(UPLOAD_FOLDER, filename), os.path.join(DATA_PATH, filename))

    # Load PDF files
    documents = load_pdf_files(DATA_PATH)
    print("Length of PDF pages:", len(documents))

    # Create chunks
    text_chunks = create_chunks(documents)
    print("Length of text chunks:", len(text_chunks))

    # Get embedding model
    embedding_model = get_embedding_model()

    # Store embeddings in FAISS
    db = FAISS.from_documents(text_chunks, embedding_model)
    db.save_local(DB_FAISS_PATH)

    return jsonify({"status": "success", "message": "PDFs processed and embeddings stored"}), 200

@app.route('/query', methods=['POST'])
def query_documents():
    try:
        # Load embedding model
        embedding_model = get_embedding_model()
        
        # Load FAISS database
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        
        # Extract query from request
        data = request.json
        user_query = data.get('query', '')
        
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
            ]
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)