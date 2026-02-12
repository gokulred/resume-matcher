import chromadb
import ollama
import uuid
from typing import List

class VectorDB:
    def __init__(self, collection_name="resumes"):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def get_embedding(self, text):
        response = ollama.embeddings(model='nomic-embed-text', prompt=text)
        return response["embedding"]

    def add_resume(self, resume_text, resume_id):
        chunks = [chunk for chunk in resume_text.split('\n\n') if chunk.strip()]
        
        ids = []
        embeddings = []
        metadatas = []
        documents = []

        for idx, chunk in enumerate(chunks):
           
            chunk_id = f"{resume_id}_chunk_{idx}"
            
          
            vector = self.get_embedding(chunk)
            
            ids.append(chunk_id)
            embeddings.append(vector)
            documents.append(chunk)
            metadatas.append({"resume_id": resume_id, "chunk_index": idx})

      
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Stored {len(chunks)} chunks for Resume {resume_id}")

    def query_resume(self, query_text:str, resume_id: str, n_results=3):
      
        query_vector = self.get_embedding(query_text)
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=n_results,
            where={"resume_id": resume_id}
        )
        
        if not results['documents'] or not results ['documents'][0]:
            return ""
        
        context = "\n".join(results['documents'][0])
        return context