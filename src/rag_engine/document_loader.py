# src/rag_engine/document_loader.py
# ============================================================================
# PHASE 2: DESIGN ARCHITECTURE: DOCUMENT LOADING & GRANULAR CHUNKING
# ============================================================================
# 1. The Problem (Why We Cannot Ingest Full Documents):
# When a user queries the RAG system, the answer often resides within a specific 
# paragraph inside a large corpus (e.g., a 14-page lecture or a 500-page book). 
# Attempting to send raw, unsegmented documents directly to a Large Language Model (LLM) 
# introduces three critical software engineering vulnerabilities:
# 
# - Context Window Saturation: LLMs operate under rigid computational constraints 
#   known as token limits. Injecting full-length academic publications overflows 
#   this boundary, crashing the runtime via memory exhaustion or context overflow.
# - "Lost in the Middle" Phenomenon: Deep learning research proves that LLMs suffer 
#   from severe attention degradation when processing heavy prompts; they retain 
#   information at the absolute margins (beginning and end) while ignoring facts 
#   buried in the middle.
# - Computation & Latency Bottlenecks: Processing thousands of irrelevant words 
#   for a simple query spikes latency, creating a sluggish user experience.
#
# 2. The Solution (Homogeneous Text Segmentation & Overlap Retention):
# To establish an efficient search space, we transform monolithic files into dense, 
# manageable data units through a two-fold engineering strategy implemented below:
#
# A. Strict Size Regularization (CHUNK_SIZE = 600):
# Text extraction standardizes document structures page-by-page. We slice raw strings 
# into uniform packets of 600 characters. This dimension isolates specific definitions, 
# concepts, or equations, ensuring that our FAISS vector index retrieves exactly 
# the relevant context needed, rather than drowning the generator in noise.
#
# B. Sliding Window Semantic Preservation (CHUNK_OVERLAP = 100):
# Arbitrary hard cuts threaten text integrity, often splitting key phrases across 
# separate blocks and breaking their semantic meaning. By implementing a 100-character 
# "Overlap" (chevauchement), trailing characters of a chunk are repeated at the onset 
# of the next. This sliding window guarantees absolute semantic continuity and contextual 
# preservation across boundary layers.
# ============================================================================

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