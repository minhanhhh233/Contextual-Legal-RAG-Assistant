from typing import List, Dict, Tuple
import numpy as np
from openai import OpenAI
import faiss
from rank_bm25 import BM25Okapi

class LegalRetriever:
    def __init__(self, open_api_key: str):
        self.client = OpenAI(api_key=open_api_key)
        self.file_data = {}  # Dict[file_id, {"chunks": List[Dict], "index": faiss.Index, "bm25_index": BM25Okapi}]
        # self.index = None
        # self.bm25_index = None
        # self.chunks = []
        self.dimension = 1536  # Dimension of text-embedding-3-small embeddings

    def create_contextual_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Create embeddings for contextualized segments using OpenAI."""
        print("Debug: Creating contextual embeddings with OpenAI")
        contextual_texts = [f"{chunk.get('contextual', '')}\n{chunk.get('chunk_text', '')}" for chunk in chunks]
        try:
            embeddings = []
            for text in contextual_texts:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                embedding = response.data[0].embedding
                embeddings.append(embedding)
            embeddings = np.array(embeddings, dtype=np.float32)
            print(f"Debug: Generated {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
            return embeddings
        except Exception as e:
            print(f"Debug: OpenAI embedding error: {str(e)}")
            raise Exception(f"Error creating embeddings with OpenAI: {str(e)}")
        
    
    def create_bm25_index(self, chunks: List[Dict]):
        """Create a BM25 index for contextualized segments."""
        print("Debug: Creating BM25 index")
        contextual_texts = [f"{chunk.get('contextual', '')} {chunk.get('chunk_text', '')}".lower().split() for chunk in chunks]
        bm25_index = BM25Okapi(contextual_texts)
        print(f"Debug: Created BM25 index for {len(contextual_texts)} segments")
        return bm25_index

    def store_file_data(self, file_id: str, chunks: List[Dict], embeddings: np.ndarray):
        print(f"Debug: Storing data for file {file_id}")
        # Store chunks
        self.file_data[file_id] = {"chunks": chunks}
        
        # Store FAISS index
        index = faiss.IndexFlatL2(self.dimension)
        index.add(embeddings)
        self.file_data[file_id]["index"] = index
        print(f"Debug: Stored {index.ntotal} embeddings for file {file_id}")

        # Store BM25 index
        self.file_data[file_id]["bm25_index"] = self.create_bm25_index(chunks)


    # def store_embeddings(self, embeddings: np.ndarray):
    #     """Store embeddings in a FAISS index."""
    #     print("Debug: Storing embeddings in FAISS")
    #     self.index = faiss.IndexFlatL2(self.dimension)
    #     self.index.add(embeddings)
    #     print(f"Debug: Stored {self.index.ntotal} embeddings")

    def retrieve_chunks_1file(self, query: str, file_id: str, k: int = 5) -> List[Tuple[str, int, Dict, float]]:
        print(f"Debug: Starting retrieval for file {file_id}")
        if file_id not in self.file_data:
            print(f"Debug: File {file_id} not found in storage")
            return []

        file_data = self.file_data[file_id]
        chunks = file_data["chunks"]
        index = file_data["index"]
        bm25_index = file_data["bm25_index"]

        if index is None or not chunks or bm25_index is None:
            print("Debug: No embeddings, chunks, or BM25 index stored.")
            return []

        # Vector similarity search
        print("Debug: Performing vector similarity search")
        try:
            query_embedding = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            ).data[0].embedding
            query_embedding = np.array([query_embedding], dtype=np.float32)
        except Exception as e:
            print(f"Debug: OpenAI query embedding error: {str(e)}")
            raise Exception(f"Error embedding query with OpenAI: {str(e)}")
        
        distances, vector_indices = index.search(query_embedding, k)
        # vector_results = [self.chunks[idx] for idx in vector_indices[0] if idx < len(self.chunks)]
        vector_results = [(idx, chunks[idx], distances[0][i], rank+1) for i, (rank, idx) in enumerate(enumerate(vector_indices[0])) if idx < len(chunks)]
        print(f"Debug: Retrieved {len(vector_results)} chunks via vector search: {vector_indices[0]}")

        # BM25 search
        print("Debug: Performing BM25 search")
        query_tokens = query.lower().split()
        bm25_scores = bm25_index.get_scores(query_tokens)
        bm25_indices = np.argsort(bm25_scores)[::-1][:k]
        # bm25_results = [self.chunks[idx] for idx in bm25_indices if idx < len(self.chunks)]
        bm25_results = [(idx, chunks[idx], bm25_scores[idx], rank+1) for rank, idx in enumerate(bm25_indices) if idx < len(chunks)]
        print(f"Debug: Retrieved {len(bm25_results)} chunks via BM25: {bm25_indices}")

    
        # Reciprocal Rank Fusion
        print("Debug: Performing Reciprocal Rank Fusion")
        rrf_scores = {}
        for chunk_idx, chunk, _, rank in bm25_results + vector_results:
            chunk_text = chunk.get("chunk_text", "")
            if chunk_text not in rrf_scores:
                rrf_scores[chunk_text] = {"chunk_idx":chunk_idx, "chunk": chunk, "score": 0}
            rrf_scores[chunk_text]["score"] += 1 / (60 + rank)  # RRF with k=60

        # Sort and deduplicate
        fused_results = [(file_id, item["chunk_idx"], item["chunk"], item["score"]) for item in sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:k]]
        print(f"Debug: Fused {len(fused_results)} unique chunks with RRF scores: {[item['score'] for item in rrf_scores.values()][:k]}")

        # print("Debug: Fusing vector and BM25 results")
        # fused_results = []
        # seen_texts = set()
        # for chunk in vector_results + bm25_results:
        #     chunk_text = chunk.get("chunk_text", "")
        #     if chunk_text not in seen_texts:
        #         fused_results.append(chunk)
        #         seen_texts.add(chunk_text)
        # fused_results = fused_results[:k]
        # print(f"Debug: Fused {len(fused_results)} unique chunks")

        # Legal reranking
        # print("Debug: Performing legal reranking")
        # reranked_results = self.rerank_chunks(query, fused_results)
        # print(f"Debug: Reranked {len(reranked_results)} chunks")
        return fused_results

    def retrieve_chunks(self, query: str, k: int = 5) -> List[Tuple[str, int, Dict]]:
        all_retrieved_chunks = [] # List[Tuple[str, int, Dict, float]]
        for file_id in self.file_data:
            results = self.retrieve_chunks_1file(query, file_id, k)
            all_retrieved_chunks.extend(results)
        
        if not all_retrieved_chunks:
            print("Debug: No chunks retrieved from any file")
            return []
        
        all_retrieved_chunks = sorted(all_retrieved_chunks, key=lambda x: x[3], reverse=True)
        final_results = [(item[0], item[1], item[2]) for item in all_retrieved_chunks[:k]] # List[Tuple[str, int, Dict]]
        # final_results = [(item["chunk"], item["score"]) for item in sorted(all_retrieved_chunks.values(), key=lambda x: x["score"], reverse=True)[:k]]
        print(f"Debug: Final results {final_results}")
        return final_results
    
    def generate_answer(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """Generate an answer using Claude with context-aware prompt."""
        print("Debug: Generating answer with context-aware prompt")
        try:
            # client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", "your-anthropic-key"))
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks):
                context = chunk.get("contextual", "")
                chunk_text = chunk.get("chunk_text", "")
                context_parts.append(f"Segment {i+1} (Context: {context}):\n{chunk_text}")
            context = "\n\n".join(context_parts)
            
            prompt = (
                f"Based on the following document segments and their legal context, "
                f"answer the query concisely and accurately, citing relevant details. If the query cannot be answered based on the segments, state so clearly.\n\n"
                f"Document Segments:\n{context}\n\n"
                f"Query: {query}\n\n"
                f"Answer in plain text, ensuring legal accuracy and clarity."
            )
            # response = client.messages.create(
            #     model="claude-3-5-sonnet-20241022",
            #     max_tokens=1000,
            #     messages=[{"role": "user", "content": prompt}]
            # )
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a legal assistant specializing in legal document analysis."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            answer = response.choices[0].message.content.strip()
            print(f"Debug: Generated answer: {answer[:100]}...")
            return answer
        except Exception as e:
            print(f"Debug: OpenAI API error: {str(e)}")
            raise Exception(f"OpenAI API error: {str(e)}")