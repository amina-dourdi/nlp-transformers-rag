# src/rag_engine/document_loader.py
# ==========================================
# PHASE 2 : RAG - Chargement & Découpage
# ==========================================

# - Charger les documents PDF ou TXT
# - Découper les textes en chunks (segments)

from typing import List, Dict
import os
from src.config import CHUNK_SIZE, CHUNK_OVERLAP
from pypdf import PdfReader # pip install pypdf



def load_pdf_documents(folder_path: str) -> List[Dict[str, str]]:
    """
    Scans a folder and extracts text from every page of every PDF file.
    Each page becomes a dictionary containing its text and exact source.
    """
    documents = []
    
    if not os.path.exists(folder_path):
        print(f"⚠️ The folder '{folder_path}' does not exist! Create it or verify the path.")
        return documents
        
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            print(f"📄 Reading: '{filename}'...")
            
            try:
                reader = PdfReader(filepath)
                for page_num, page in enumerate(reader.pages, start=1):
                    text = page.extract_text()
                    
                    # Keep only pages that contain readable text
                    if text and text.strip():
                        documents.append({
                            "text": text.strip(),
                            "source": f"{filename} - Page {page_num}"
                        })
                        
                print(f"  ✅ Extraction successful: {len(reader.pages)} pages found.")
                
            except Exception as e:
                print(f"  ❌ Error while reading '{filename}': {e}")
                
    print(f"\n📚 Total: {len(documents)} pages extracted from the corpus.")
    return documents


def get_chunks(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Splits document pages into homogeneous chunks
    using a sliding window with overlap.
    """
    chunks = []
    
    for doc in documents:
        text = doc["text"]
        source = doc["source"]
        
        # If the text is smaller than the target size, keep it as a single chunk
        if len(text) <= CHUNK_SIZE:
            chunks.append({
                "text": text,
                "source": source
            })
            continue
            
        # Sliding Window chunking algorithm
        start = 0
        chunk_num = 1
        
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk_text = text[start:end]
            
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text.strip(),
                    "source": f"{source} (Chunk {chunk_num})"
                })
                chunk_num += 1
                
            # Move forward while preserving overlap for semantic continuity
            start += (CHUNK_SIZE - CHUNK_OVERLAP)
            
    print(
        f"✂️ Chunking completed: {len(chunks)} chunks created from {len(documents)} pages."
    )
    return chunks