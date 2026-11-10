import math
import os
import uuid

import chromadb
import ollama

EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
CHUNK_CHARS = 300


def chunk_text(text: str, max_chars: int = CHUNK_CHARS) -> list[str]:
    # Line windows rather than "\n\n" splits: pypdf output rarely has blank lines,
    # which would leave the whole resume as one chunk and make retrieval a no-op.
    chunks, current = [], ""
    for line in (ln.strip() for ln in text.splitlines()):
        if not line:
            continue
        current = f"{current}\n{line}" if current else line
        if len(current) >= max_chars:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


class VectorDB:
    # One short-lived collection per resume. A shared collection broke after ~20 requests: deletes only
    # mark HNSW entries, and a query filtered to one resume then couldn't find enough live neighbours.
    def __init__(self):
        # in-memory: every resume is deleted right after its request, so nothing needs to persist
        self.client = chromadb.EphemeralClient(settings=chromadb.Settings(anonymized_telemetry=False))

    def get_embeddings(self, texts: list[str]):
        return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]

    def retrieve(self, resume_text: str, query_text: str, fraction: float) -> str:
        """The resume chunks closest to query_text: `fraction` of them, rounded up."""
        chunks = chunk_text(resume_text)
        if not chunks:
            return ""
        # one Ollama request embeds every chunk and the query together
        *chunk_vectors, query_vector = self.get_embeddings([*chunks, query_text])

        name = f"resume-{uuid.uuid4()}"
        collection = self.client.create_collection(name=name)
        try:  # created only after embedding succeeded, and always dropped, so nothing piles up
            collection.add(ids=[str(i) for i in range(len(chunks))], embeddings=chunk_vectors)
            results = collection.query(
                query_embeddings=[query_vector], n_results=math.ceil(len(chunks) * fraction), include=[]
            )
        finally:
            self.client.delete_collection(name=name)
        # Chroma ranks by similarity; put the picks back in resume order so the model reads a coherent resume
        return "\n".join(chunks[i] for i in sorted(int(chunk_id) for chunk_id in results["ids"][0]))
