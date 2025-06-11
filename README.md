# Contextual Legal RAG Assistant

This application presents an advanced **Retrieval-Augmented Generation (RAG)** system specifically designed for legal research. It moves beyond traditional RAG by deeply integrating legal context, aiming to deliver more accurate and relevant responses for complex legal queries.

This idea is inspired by approaches from [Anthropic](https://www.anthropic.com/news/contextual-retrieval) and [TrueLaw.ai](https://www.truelaw.ai/blog/contextual-legal-rag).

---
### Demo
[![Legal RAG Assistant Demo](https://img.youtube.com/vi/UxzdA7gqrr0/hqdefault.jpg)](https://www.youtube.com/watch?v=UxzdA7gqrr0)
---

### How It Works: An Enhanced Legal Retrieval Process

Our Contextual Legal RAG Assistant addresses the inherent limitations of general-purpose RAG in the legal domain by embedding legal-specific context directly into both the document processing and retrieval stages. Here's an overview of its key components:

#### 1. Legal Document Preprocessing

* **a. Contextual Legal Chunking:**
    Documents are intelligently chunked to align with their inherent legal structure (e.g., sections, clauses, logical arguments). This process leverages the advanced capabilities of the **Claude model**, which has demonstrated surprising effectiveness in understanding complex document layouts and semantic meaning for optimal chunking.

* **b. Contextual Legal Embeddings:**
    For each generated chunk, we create a **concise summary of the chunk** to situate it within the broader document. This summary is then prepended to its chunk text. The combined text is then transformed into a semantic dense vector embedding using **OpenAI's `text-embedding-3-small` model**.

* **c. Contextual Legal BM25 Indexing:**
    The  combined text is also used to build an enhanced **BM25 index**. This step boosts the system's ability to find exact matches for critical legal elements like citations, case names, and specific legal terminology, complementing the semantic search capabilities.

#### 2. Enhanced Legal Retrieval

When a legal query is submitted:

* **a. Semantic Query Embedding:** The user's query is first embedded into a vector representation using **OpenAI's `text-embedding-3-small` model**.
* **b. Semantic Similarity Search:** This embedding is used to identify the top relevant chunks based on **semantic embedding similarity** to the query.
* **c. Exact Match Retrieval (BM25):** Simultaneously, the contextualized BM25 index is queried to find top chunks based on **exact keyword matches** and legal terminology.
* **d. Hybrid Rank Fusion:** The results from both semantic and BM25 searches are then combined and deduplicated using **rank fusion techniques**.

#### 3. Response Generation

Finally, the top-ranked retrieved chunks are appended to the user's original query and sent as context to the **GPT-4o mini model** to generate an accurate and relevant legal response.

---

### Important Considerations & Limitations

This application was developed rapidly to demonstrate the core concept. As such, please note the following:

* **User Interface (UI):** The current UI is functional but minimal and may not handle all possible user interactions gracefully.
* **File Input:** For optimal results, please **upload an HTML file** rather than a PDF. Current extracting text from PDF approach can lead to information loss, which impacts the quality of chunking and retrieval.
* **File Length:** We recommend uploading files that are **up to 2 pages long**. The Claude model, used for contextual chunking, has token output limitations that may affect performance on very long documents.


### Sample Documents & Query Examples

You can find sample legal documents within the data folder of this repository.
Below are some example questions:
* **a. DOcument 1:**
. Who are the representatives for Puma SE in this case?
. What is the role of the Chamber determining whether appeals may proceed in this case?
. Who are the parties involved in Case C-4925 P?
. What is the role of EUIPO in this legal proceeding?
. What is the significance of Article 56 of the Statute of the Court of Justice of the European Union in this appeal?

* **a. DOcument 2:**
. Who are the representatives for Bionext SA in Case T-253/25?
. Who are the applicant and defendant in Case T-253/25?
. What is the role of Bionext SA, and where is it established?
. What are the three pleas in law relied upon by Bionext SA to challenge the European Commission’s decision?
. What is the alleged State aid (SA.100547) related to in this case, and how does it connect to COVID-19 Large Scale Testing?
