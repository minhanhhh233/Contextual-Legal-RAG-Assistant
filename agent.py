import streamlit as st
from LegalChunker import LegalChunker
from LegalRetriever import LegalRetriever
import time

try:
    anthropic_api_key = st.secrets["anthropic_api_key"]
    open_api_key = st.secrets["openai_api_key"]
except KeyError:
    st.error("API key not found. Please set it in your Streamlit secrets.")
    st.stop() # Stop the app if key is missing

# Streamlit page configuration
st.set_page_config(page_title="Legal PDF Chunker with Claude", page_icon="📜", layout="wide")

def process_file(uploaded_file, file_id):
    """Process the uploaded file and store chunks and encodings."""
    st.write(f"Debug: Processing file: {uploaded_file.name}, ID: {file_id}")
    try:
        with st.spinner(f"Processing {uploaded_file.name} and generating chunks..."):
            chunks = st.session_state.chunker.chunk_file(uploaded_file)
            st.write(f"Debug: Chunks generated for {file_id}")
            st.write(f"Debug: Number of chunks: {len(chunks)}")

        if chunks:
            embeddings = st.session_state.retriever.create_contextual_embeddings(chunks)
            st.session_state.retriever.store_file_data(file_id, chunks, embeddings)
            st.write(f"Debug: Embeddings, BM25 encodings stored for {file_id}")
            # TODO
            st.session_state.files[file_id] = {
                "name": uploaded_file.name,
                "size": uploaded_file.size,
                "processed": True
            }
            return chunks
        else:
            st.warning(f"No chunks generated for {uploaded_file.name}.")
            st.session_state.files[file_id]["processed"] = False
            return None
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        st.write(f"Debug: Failed to process file {file_id}")
        st.session_state.files[file_id]["processed"] = False
        return None
    
def file_uploader_callback():
    """Callback for file uploader to process file only if it has changed."""
    uploaded_file = st.session_state.file_uploader
    if uploaded_file is None:
        return

    # Generate unique file ID (name + timestamp)
    file_id = f"{uploaded_file.name}_{int(time.time()*1000)}"
    if file_id in st.session_state.files:
        st.write(f"Debug: File {uploaded_file.name} already processed as {file_id}")
        return

    # TODO
    # Initialize file entry
    st.session_state.files[file_id] = {
        "name": uploaded_file.name,
        "size": uploaded_file.size,
        "processed": False
    }

    # Process new file
    st.success(f"File {uploaded_file.name} uploaded successfully!")
    chunks = process_file(uploaded_file, file_id)
    if chunks:
        # st.session_state.files[file_id]["chunks"] = chunks
        # Display chunks
        with st.expander("View Chunks"):
            for i, chunk in enumerate(chunks):
                st.subheader(f"Chunk {i+1}")
                st.write(f"**Start Words**: {chunk.get('start_words', '')}")
                st.write(f"**End Words**: {chunk.get('end_words', '')}")
                st.write(f"**Text**: {chunk.get('chunk_text', '')}")
                st.write(f"**Context**: {chunk.get('contextual', '')}")

def main():
    st.title("Legal RAG Assistant")
    

    # Initialize chunker
    # chunker = LegalChunker(anthropic_api_key)
    # Initialize session state for chunker and retriever
    if 'chunker' not in st.session_state:
        st.session_state.chunker = LegalChunker(anthropic_api_key)
    if 'retriever' not in st.session_state:
        st.session_state.retriever = LegalRetriever(open_api_key)
    if 'files' not in st.session_state:
        st.session_state.files = {}

    # File uploader
    st.file_uploader(
        "Upload a PDF or HTML file",
        type=["pdf", "html", "htm"],
        key="file_uploader",
        on_change=file_uploader_callback
    )

    # Display uploaded files
    if st.session_state.files:
        st.subheader("Uploaded Files")
        for file_id, file_data in st.session_state.files.items():
            st.write(f"**File**: {file_data['name']} (ID: {file_id}, Size: {file_data['size']} bytes)")

    # # File uploader
    # # uploaded_file = st.file_uploader("Upload a PDF or HTML file", type=["pdf", "html", "htm"])
    # if uploaded_file:
    #     st.success("File uploaded successfully!")
    #     try:
    #         with st.spinner("Processing file and generating chunking suggestions..."):
    #             # chunks = chunker.chunk_pdf(uploaded_file)
    #             chunks = st.session_state.chunker.chunk_file(uploaded_file)

    #         if chunks:
    #             st.session_state.retriever.chunks = chunks
    #             embeddings = st.session_state.retriever.create_contextual_embeddings(chunks)
    #             st.session_state.retriever.store_embeddings(embeddings)
    #             st.session_state.retriever.create_bm25_index(chunks)
    #             st.write("Debug: Embeddings and BM25 index stored")

    #             # Display chunks
    #             with st.expander("View Chunks"):
    #                 for i, chunk in enumerate(chunks):
    #                     st.subheader(f"Chunk {i+1}")
    #                     st.write(f"**Start Words**: {chunk.get('start_words', '')}")
    #                     st.write(f"**End Words**: {chunk.get('end_words', '')}")
    #                     st.write(f"**Text**: {chunk.get('chunk_text', '')}")
    #                     st.write(f"**Context**: {chunk.get('contextual', '')}")

                
    #     except Exception as e:
    #         st.error(f"Error processing file: {str(e)}")
    #         st.write("Debug: Failed to process file or generate embeddings")

    # Query input
    st.subheader("Ask a Legal Question")
    query = st.text_input("Enter your question about the document:")
    if query:
        try:
            with st.spinner("Retrieving relevant chunks and generating answer..."):
                # Retrieve relevant chunks
                retrieved_chunks = st.session_state.retriever.retrieve_chunks(query, k=5)
                if not retrieved_chunks:
                    st.warning("No relevant chunks found or no document processed.")
                else:
                    st.write(f"Debug: Retrieved {len(retrieved_chunks)} chunks for query")
                    with st.expander("View Retrieved Chunks"):
                        for file_id, chunk_idx, chunk in retrieved_chunks:
                            st.write(f"**File id {file_id} Chunk {chunk_idx+1} (Context: {chunk.get('contextual', '')})**: {chunk.get('chunk_text', '')[:200]}...")
                    
                    # Generate answer
                    retrieved_chunks = [chunk for _, _, chunk in retrieved_chunks]
                    answer = st.session_state.retriever.generate_answer(query, retrieved_chunks)
                    st.subheader("Answer")
                    st.write(answer)
        except Exception as e:
            st.error(f"Error generating answer: {str(e)}")
            st.write("Debug: Failed to retrieve chunks or generate answer")

if __name__ == "__main__":
    main()