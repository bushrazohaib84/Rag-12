import streamlit as st
import fitz  # PyMuPDF for PDF processing
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from gtts import gTTS  # Google Text-to-Speech
import os

# Initialize the embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS setup
dimension = 384  # Embedding dimension for the chosen SentenceTransformer model
index = faiss.IndexFlatL2(dimension)  # L2 distance index
pdf_texts = []  # Store texts along with their document IDs


# Function to extract text from PDF (accepting file-like object)
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")  # Open PDF from file-like object
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text += page.get_text("text")
    return text


# Function to split text into chunks
def split_text_into_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks


# Function to add documents to FAISS index
def add_to_faiss_index(text_chunks):
    global pdf_texts
    embeddings = embedder.encode(text_chunks)  # Get embeddings for chunks
    index.add(np.array(embeddings, dtype=np.float32))  # Add embeddings to FAISS index
    pdf_texts.extend(text_chunks)


# Function to retrieve relevant chunks from FAISS
def retrieve(query, k=5):
    if index.ntotal == 0:
        return []  # Return an empty list if no data has been indexed
    
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    
    # Ensure indices are within the range of pdf_texts to avoid IndexError
    relevant_texts = []
    for idx in indices[0]:
        if idx < len(pdf_texts):
            relevant_texts.append(pdf_texts[idx])
    
    return relevant_texts


# Function for text-to-speech


# Streamlit app setup
st.title("PDF-based RAG System with FAISS and Text-to-Speech")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file is not None:
    # Extract text from uploaded PDF
    with st.spinner("Extracting text from the PDF..."):
        extracted_text = extract_text_from_pdf(uploaded_file)
    
    st.success("Text extracted successfully!")

    # Display the extracted text in the Streamlit app
    st.subheader("Extracted Text from PDF:")
    st.text_area("Here is the content of the PDF:", value=extracted_text, height=300)

    # Split text into chunks and add to FAISS index
    chunks = split_text_into_chunks(extracted_text)
    add_to_faiss_index(chunks)
    
    st.write(f"Added {len(chunks)} chunks to the FAISS index.")

# Text-to-Speech optionuse_text_to_speech = st.checkbox("Enable Text-to-Speech")

# Query input
query = st.text_input("Ask a question about the PDF:")
if query:
    with st.spinner("Retrieving relevant information..."):
        results = retrieve(query)
    
    if results:
        st.write("Top relevant chunks:")
        for i, result in enumerate(results):
            st.write(f"**Chunk {i + 1}:** {result}")
            
            # If Text-to-Speech is enabled, speak the first result
         
    else:
        st.write("No relevant information found or the index is empty.")
