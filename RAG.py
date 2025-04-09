import os
import json
import torch
import faiss  
import numpy as np
import fitz  # PyMuPDF
from docx import Document
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Set the folder containing your resume templates
docs_folder = "/Users/raghu/Desktop/Job-applying-bot/Resumes & Cover Letter"  # Folder containing your templates

# Set the device (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"


def extract_text_from_pdf(pdf_path):
    """Extracts full text from a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"
    return full_text.strip()

def extract_text_from_docx(docx_path):
    """Extracts full text from a DOC/DOCX using python-docx.
       If the file is not a valid Word file, returns an empty string."""
    try:
        doc = Document(docx_path)
        full_text = "\n".join([para.text for para in doc.paragraphs])
        return full_text.strip()
    except Exception as e:
        print(f"Error processing DOCX file {docx_path}: {str(e)}")
        return ""

def get_template_files(directory):
    """Scans the given directory for PDF, DOC, and DOCX files, ignoring temporary files."""
    template_files = []
    if os.path.isdir(directory):
        for file in os.listdir(directory):   # Skip the files starting with ~$
            if file.startswith("~$"):
                continue
            if file.lower().endswith(('.pdf', '.doc', '.docx')):
                template_files.append(os.path.join(directory, file))
    else:
        print(f"{directory} is not a valid directory.")
    return template_files

def build_corpus_and_metadata(directory):
    files = get_template_files(directory)
    corpus = []      # List to hold extracted text from each document
    metadata = []    # Store metadata (e.g., file path)
    for file_path in files:
        text = ""
        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(('.doc', '.docx')):
            text = extract_text_from_docx(file_path)
        if text:
            corpus.append(text)
            metadata.append({"file_path": file_path})
    return corpus, metadata

corpus, metadata = build_corpus_and_metadata(docs_folder)
print(f"Extracted {len(corpus)} documents from templates.")

# Build FAISS Index
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
corpus_embeddings = embedder.encode(corpus, convert_to_numpy=True)
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
if torch.cuda.is_available():
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)

print("Building FAISS index...")
index.add(corpus_embeddings)
print(f"FAISS index built with {index.ntotal} vectors.")

# Retrival Function
def retrieve_resume(prompt, top_k=1):
    """
    Given a prompt, encodes it into an embedding, and searches the FAISS index.
    Returns a list of retrieved resume texts.
    """
    query_embedding = embedder.encode([prompt], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved = [corpus[idx] for idx in indices[0]]
    return retrieved

# Text Generation using T5 or LED for resume synthesis
# If T5-base doesn't allow a sufficiently long output, we can switch to LED-base-16384.
gen_model_name = "allenai/led-base-16384"
tokenizer_gen = AutoTokenizer.from_pretrained(gen_model_name)
gen_model = AutoModelForSeq2SeqLM.from_pretrained(gen_model_name).to(device)
generation_pipeline = pipeline("text2text-generation",
                               model=gen_model,
                               tokenizer=tokenizer_gen,
                               device=0 if torch.cuda.is_available() else -1)

def generate_resume(query):
    """
    Retrieves a similar ATS-friendly resume from the vector database using the query,
    summarizes the retrieved context, and then generates a new resume using the LED model.
    """
    retrieved_docs = retrieve_resume(query, top_k=1)
    context = retrieved_docs[0] if retrieved_docs else "No context available."
    
    # Summarize the retrieved context (for example, take the first 300 words)
    def summarize_text(text, max_words=15000):
        words = text.split()
        return " ".join(words[:max_words])
    
    retrieved_docs = retrieve_resume(query, top_k=1)
    context = retrieved_docs[0] if retrieved_docs else "No context available."
    clean_context = summarize_text(context, max_words=15000)
    prompt = (f"Based on the following ATS-friendly resume template:\n\n{clean_context}\n\n"
          "Generate a new ATS-friendly resume for a Data Analyst with the following sections:\n"
          "1. Contact Information\n"
          "2. Professional Summary\n"
          "3. Skills\n"
          "4. Professional Experience\n"
          "5. Education\n")
    
    # Tune generation parameters (adjust temperature and beams as needed)
    output = generation_pipeline(prompt, max_new_tokens=2048, temperature=0.7, num_beams=4)
    return output[0]["generated_text"]
# New Resume with RAG Approach
query = ("Generate an ATS-friendly resume for a Data Analyst with 5 years of experience "
         "in Python, SQL, and Data Visualization tools for Aparna Patnala. Completed B.Tech in Computer Science "
         "from VNR VJIET, Hyderabad.")
generated_resume = generate_resume(query)
print("Generated Resume:")
print(generated_resume)

# Save the trained RAG pipeline (index and metadata)
def save_trained_rag():
    faiss.write_index(index, "Trained_RAG_index.faiss")
    trained_rag_metadata = {
        "corpus": corpus,
        "metadata": metadata,
        "embedder_model_name": "all-MiniLM-L6-v2",
        "generation_model_name": gen_model_name,
    }
    with open("Trained_RAG_metadata.json", "w") as f:
        json.dump(trained_rag_metadata, f)
    print("Trained_RAG pipeline saved (index and metadata).")

# Preview templates
def preview_templates(directory):
    files = get_template_files(directory)
    print(f"Found {len(files)} template files in '{directory}'.")
    for file_path in files:
        print(f"\nProcessing file: {file_path}")
        text = ""
        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        elif file_path.lower().endswith(('.doc', '.docx')):
            text = extract_text_from_docx(file_path)
        print("Extracted text preview:")
        print(text[:500])
        print("=" * 80)

def save_trained_rag():
    """
    Saves the FAISS index and associated metadata to disk.
    """
    faiss.write_index(index, "Trained_RAG_index.faiss")
    trained_rag_metadata = {
        "corpus": corpus,
        "metadata": metadata,
        "embedder_model_name": "all-MiniLM-L6-v2",
        "generation_model_name": gen_model_name,
    }
    with open("Trained_RAG_metadata.json", "w") as f:
        json.dump(trained_rag_metadata, f)
    print("Trained_RAG pipeline saved (index and metadata).")        

# Main function to run the script
def main():
    preview_templates(docs_folder)
    print("\nSynthesizing a new resume using the RAG pipeline...\n")
    generated_resume = generate_resume(query)
    print("Generated ATS-Friendly Resume:")
    print(generated_resume)
    save_trained_rag()

if __name__ == "__main__":
    main()