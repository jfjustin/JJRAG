import os
import logging
import traceback
from typing import List, Union
from pathlib import Path

import streamlit as st
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


#Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_app")




# step1 function for sys running (loadinh->spliting->vectorize store->query)
def load_documents(uploaded_files):
    
    documents = []
    
    try:
        
        if not uploaded_files:
            logger.warning("No files uploaded")
            st.warning("Please upload at least one document file.")
            return []
        
        # Create a temporary directory to store files
        temp_dir = "temp_docs"
        os.makedirs(temp_dir, exist_ok=True)
        
        for uploaded_file in uploaded_files:
            try:
                # Get file info
                file_name = uploaded_file.name
                file_extension = Path(file_name).suffix.lower()
                
                # Save file temporarily
                temp_path = os.path.join(temp_dir, file_name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process based on file type
                if file_extension == '.pdf':
                    logger.info(f"Loading PDF: {file_name}")
                    loader = PyPDFLoader(temp_path)
                    documents.extend(loader.load())
                    
                elif file_extension == '.docx':
                    logger.info(f"Loading DOCX: {file_name}")
                    loader = Docx2txtLoader(temp_path)
                    documents.extend(loader.load())
                    
                elif file_extension == '.txt':
                    logger.info(f"Loading TXT: {file_name}")
                    loader = TextLoader(temp_path)
                    documents.extend(loader.load())
                    
                else:
                    logger.warning(f"Unsupported file type: {file_extension}")
                    st.warning(f"Skipping unsupported file: {file_name}")
                
            except Exception as e:
                logger.error(f"Error processing file {uploaded_file.name}: {str(e)}")
                logger.error(traceback.format_exc())
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
        
        # Log success
        logger.info(f"Successfully loaded {len(documents)} document(s)")
        if documents:
            st.success(f"Successfully loaded {len(documents)} document(s)")
        else:
            st.warning("No documents were loaded.")
            
        return documents

        
    except Exception as e:
        logger.error(f"Error in load_documents: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error loading documents: {str(e)}")
        return []


        
def split_documents(documents, chunk_size=1000, chunk_overlap=200):

    # output: List of document chunks
 
    try:
        # Check if documents exist
        if not documents:
            logger.warning("No documents to split")
            return []
            
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        document_chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(document_chunks)} document chunks")
        
        return document_chunks
        
    except Exception as e:
        logger.error(f"Error in split_documents: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error splitting documents: {str(e)}")
        return []

def create_vector_store(document_chunks, cache_dir="./vector_cache"):
    # cache_dir: Directory to store cached vector stores
    # output: FAISS vector store instance

    try:
        # Check if document chunks exist
        if not document_chunks:
            logger.warning("No document chunks to create vector store")
            st.warning("No document chunks available for creating vector store.")
            return None
            
        # Create embeddings and vector store
        logger.info("Creating vector store")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = FAISS.from_documents(document_chunks, embeddings)
        
        logger.info("Vector store created successfully")
        return vector_store
        
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        logger.error(traceback.format_exc())
        st.error(f"Error creating vector store: {str(e)}")
        return None


# pass in encripted api key thru key.env
from dotenv import load_dotenv


load_dotenv('key.env') 
api_key = os.getenv('OPENAI_API_KEY')  # Use the exact variable name from your file

print(f"API Key loaded: {api_key is not None}")
print(f"First few characters: {api_key[:5]}..." if api_key else "No key found")





#running main app
def main():
    """
    Main function for the Streamlit app
    """
    st.title("Document Q&A System")
    st.write("Upload your PDF, Word (.docx), or text (.txt) files and ask questions about their content.")
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Choose document files",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )
    
    # Initialize session state for vector store
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    # Button to trigger RAG setup
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                try:
                    # Load documents
                    documents = load_documents(uploaded_files)
                    
                    if documents:
                        # Split documents
                        document_chunks = split_documents(documents)
                        
                        if document_chunks:
                            # Create vector store
                            vector_store = create_vector_store(document_chunks)
                            
                            if vector_store:
                                st.session_state.vector_store = vector_store
                                st.success("Documents processed successful.")
                            else:
                                st.error("Failed to create vector store.")
                        else:
                            st.error("Failed to create document chunks.")
                    else:
                        st.error("No documents were loaded.")
                        
                except Exception as e:
                    logger.error(f"Error in document processing: {str(e)}")
                    logger.error(traceback.format_exc())
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please upload at least one doc.")
    
    # Query input
    query = st.text_input("Query to doc:")
    
    # Handle query
    if st.button("Submit Question") and query:
        if st.session_state.vector_store is not None:
            try:
                with st.spinner("Searching..."):
                    # Search for relevant documents
                    docs = st.session_state.vector_store.similarity_search(query, k=4)
                    
                    # Display results
                    if docs:
                        st.subheader("Search Results")
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Result {i+1}**")
                            st.write(doc.page_content)
                            st.divider()
                    else:
                        st.info("No relevant information found. Try again.")
            except Exception as e:
                logger.error(f"Error in query processing: {str(e)}")
                logger.error(traceback.format_exc())
                st.error(f"Error processing query: {str(e)}")
        else:
            st.warning("Please process documents first before asking questions.")




if __name__ == "__main__":
    main()