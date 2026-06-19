# src/rag_engine/vector_store.py

# ============================================================================
# PHASE 2 : DESIGN ARCHITECTURE: INDEX PERSISTENCE & OPTIMIZATION
# ============================================================================
# 1. The Problem (Without Vector Persistence):
# During the document processing phase, text sequences are divided into discrete chunks. 
# To make these fragments searchable by the RAG architecture, they must pass through 
# an Embedding Model. This model performs a computationally heavy mathematical process, 
# transforming natural language strings into high-dimensional numerical dense arrays (Vectors).
# 
# Without a dedicated persistence layer, a severe "Performance Bottleneck" arises:
# - Every time the Streamlit application instantiates or reboots, the CPU is forced 
#   to re-compute the entire vector space for all chunks from scratch.
# - While fast for miniature setups, scaled knowledge bases (e.g., a 500-page book) 
#   would trigger minutes of computational latency at every runtime init, rendering 
#   the production pipeline highly inefficient.
#
# 2. The Solution (With Index Serialization & Metadata Mapping):
# The core architectural goal is "Compute Once, Use Indefinitely". We bypass redundant 
# transformations by storing the finalized states directly into the file system as 
# persistent binary and structural configurations. 
# This is orchestrated via two complementary methods:
#
# A. save_index():
# Meta's FAISS library provides ultra-fast nearest-neighbor lookup matrices, but it is 
# inherently "blind" to original text strings (it strictly indexes index keys and numerical shapes). 
# To circumvent this, we implement a hybrid storage mechanism:
#   - Serializing the pure numerical coordinate matrices into a specialized binary file (faiss.index).
#   - Synchronously writing a descriptive text catalog (source files, page numbers, text content) 
#     into a parallel metadata file formatted in structured JSON (faiss.index.docs.json).
#
# B. load_index():
# When the app reboots, rather than calling the transformer model to run redundant forward passes, 
# the system reaches straight to the storage directories. It reloads both the mathematical index 
# and text dictionaries back into active memory in milliseconds, guaranteeing lightning-fast, 
# zero-latency execution.
# ============================================================================


from sentence_transformers import SentenceTransformer
from typing import List, Dict
from src.config import *
import numpy as np
import faiss
import json



class VectorStoreManager:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, index_path: str = VECTOR_DB_PATH):
        """
        Manages semantic vector index generation, storage, and persistence.
        """
        self.model_name = model_name
        self.index_path = index_path
        self.documents = []  # To cache original string contexts and sources (Metadata)
        self.index = None    # Core FAISS index tracker
        
        print(f"🔄 Loading sentence embedding model: '{model_name}'...")
        self.embedding_model = SentenceTransformer(model_name)
        print("✅ Semantic embedding transformer ready!")

    def build_and_save_index(self, chunks: List[Dict[str, str]]):
        """
        Computes textual embeddings, instantiates an L2 flat FAISS vector database, 
        and automatically persists everything to disk.
        """
        self.documents = chunks
        raw_texts = [c["text"] for c in chunks]
        
        print(f"🔄 Computing semantic vectors for {len(raw_texts)} chunks...")
        embeddings = self.embedding_model.encode(raw_texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")
        
        # Build FAISS index mapped to the specific structural dimension of the model
        vector_dimension = embeddings.shape[1]
        print(f"📐 Vector dimensional space: {vector_dimension}")
        
        # Utilizing Euclidean (L2) Distance for deep semantic retrieval match
        self.index = faiss.IndexFlatL2(vector_dimension)
        self.index.add(embeddings)
        print(f"✅ FAISS index established with {self.index.ntotal} registered vectors.")
        
        # Automatically persist components to local memory fields (Task F5)
        self.save_index()

    def save_index(self):
        """
        [Task F5 Execution] Serializes the binary FAISS index and structural JSON metadata to disk.
        """
        if self.index is None:
            print("⚠️ No structural index initialized to preserve!")
            return
            
        # Guarantee parent directories exist prior to serialization
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        # A. Persist the mathematical vector points inside the binary index file
        faiss.write_index(self.index, self.index_path)
        print(f"💾 Binary FAISS vectors successfully written to: '{self.index_path}'")
        
        # B. Persist raw textual strings since FAISS strictly manages vectors alone
        metadata_json_path = self.index_path + ".docs.json"
        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        print(f"💾 Accompanying JSON text mappings written to: '{metadata_json_path}'")

    def load_index(self) -> bool:
        """
        [Task F5 Execution] Reloads cached FAISS databases and document dictionary catalogs from disk.
        Returns True if operational, False if index files are missing.
        """
        metadata_json_path = self.index_path + ".docs.json"
        
        if not os.path.exists(self.index_path) or not os.path.exists(metadata_json_path):
            print(f"⚠️ Pre-computed database cache not found at target: '{self.index_path}'")
            return False
            
        # A. Reload vector spaces directly from local binarized files
        self.index = faiss.read_index(self.index_path)
        
        # B. Reload matching metadata sequences synchronously
        with open(metadata_json_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
            
        print(f"📂 [Success] Locally persisted FAISS index reloaded ({self.index.ntotal} vectors).")
        print(f"📂 Synchronized {len(self.documents)} context sources from disk metadata cache.")
        return True

    def search_top_k(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        """
        Transforms incoming natural language queries into mathematical arrays, 
        queries the FAISS matrix, and returns the top k semantic matches.
        """
        if self.index is None:
            print("⚠️ Vector storage index not operational! Initialize or run data ingestion first.")
            return []
            
        # Encode user string query into the exact same vector dimension
        query_vector = self.embedding_model.encode([query])
        query_vector = np.array(query_vector).astype("float32")
        
        # Execute mathematical flat spatial L2 nearest-neighbors search
        distances, target_indices = self.index.search(query_vector, k)
        
        retrieved_contexts = []
        for index_point in target_indices[0]:
            if index_point != -1 and index_point < len(self.documents):
                retrieved_contexts.append(self.documents[index_point])
                
        return retrieved_contexts
