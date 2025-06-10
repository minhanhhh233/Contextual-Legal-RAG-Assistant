import json
import PyPDF2
from anthropic import Anthropic, AnthropicError
import re
import os
from typing import List, Dict
from io import BytesIO
from bs4 import BeautifulSoup


class LegalChunker:
    def __init__(self, anthropic_api_key: str):
        """Initialize the chunker with Anthropic API client."""
        self.client = Anthropic(api_key=anthropic_api_key)

    def extract_text_from_html(self, html_file):
        print("Debug: Starting extract_text_from_html")
        try:
            if hasattr(html_file, 'read'):
                print("Debug: Input is an UploadedFile, reading bytes")
                html_content = html_file.read().decode('utf-8', errors='ignore')
                print(f"Debug: HTML content length: {len(html_content)} characters")
            else:
                print("Debug: Input is not an UploadedFile, using as is")
                html_content = html_file
                
            # Parse HTML with BeautifulSoup
            print("Debug: Initializing BeautifulSoup")
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for element in soup(['script', 'style']):
                element.decompose()
                
            # Extract text from relevant tags
            text = ""
            for element in soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span']):
                element_text = element.get_text(separator=" ", strip=True)
                if element_text:
                    text += element_text + "\n"
                    
            print(f"Debug: Total extracted text length: {len(text)} characters")
            print(f"Debug: First 100 chars of extracted text: {text[:100]}...")
            return text.strip()
        except Exception as e:
            print(f"Debug: Error in extract_text_from_html: {str(e)}")
            raise Exception(f"Error extracting text from HTML: {str(e)}")
        
    def extract_text_from_pdf(self, pdf: str) -> str:
        """Extract text from a PDF file."""
        try:
            # If pdf_file is an UploadedFile, read its bytes
            if hasattr(pdf, 'read'):
                pdf_content = pdf.read()
                pdf = BytesIO(pdf_content)

            pdf_reader = PyPDF2.PdfReader(pdf) # read your PDF file
            # extract the text data from your PDF file after looping through its pages with the .extract_text() method
            text_data= ""
            for page in pdf_reader.pages: # for loop method
                text_data += page.extract_text()
            
            return text_data
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
        # try:
        #     with open(pdf_path, 'rb') as file:
        #         reader = PyPDF2.PdfReader(file)
        #         text = ""
        #         for page in reader.pages:
        #             page_text = page.extract_text()
        #             if page_text:
        #                 text += page_text + "\n"
        #         return text.strip()
        # except Exception as e:
        #     raise Exception(f"Error extracting text from PDF: {str(e)}")

    def clean_text(self, text: str) -> str:
        """Clean extracted text by removing excessive whitespace and artifacts."""
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'\n+', '\n', text)  # Normalize newlines
        return text.strip()

    def clean_and_parse_json(self, raw_response: str) -> List[Dict]:
        """Clean and parse JSON response, removing invalid trailing parts."""
        print("Debug: Starting clean_and_parse_json")
        cleaned_response = raw_response.strip()
        
        # Remove markdown if present
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[7:-3].strip()
        elif cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[3:-3].strip()
        print(f"Debug: Cleaned response (first 500 chars): {cleaned_response[:500]}...")

        # Ensure response is a JSON array
        if not cleaned_response.startswith("["):
            print("Debug: Response is not a JSON array. Attempting to fix.")
            cleaned_response = cleaned_response[cleaned_response.find("["):]
            

        # Initialize variables for parsing
        valid_json = cleaned_response
        last_valid_idx = len(cleaned_response)

        # Try parsing, trimming incomplete objects if necessary
        while valid_json:
            try:
                chunks = json.loads(valid_json)
                if not isinstance(chunks, list):
                    print("Debug: Parsed response is not a list. Invalid JSON.")
                    return []
                print(f"Debug: Successfully parsed {len(chunks)} chunks")
                return chunks
            except json.JSONDecodeError as e:
                print(f"Debug: JSON parsing error: {str(e)}")
                # Find the last complete object by locating the last valid '}'
                last_comma_idx = valid_json.rfind(",", 0, last_valid_idx)
                last_brace_idx = valid_json.rfind("}", 0, last_valid_idx)
                
                # No complete objects found
                if last_brace_idx == -1:
                    print("Debug: No valid JSON objects found (no closing '}').")
                    return []
                
                # If there's a trailing comma or incomplete object, trim to last complete object
                if last_comma_idx > last_brace_idx:
                    valid_json = valid_json[:last_brace_idx + 1] + "]"
                else:
                    # Trim to last complete object
                    valid_json = valid_json[:last_brace_idx + 1]
                    # Ensure array is closed
                    if not valid_json.endswith("]"):
                        valid_json += "]"
                
                last_valid_idx = last_brace_idx - 1
                print(f"Debug: Trimmed response to: {valid_json[:500]}...")

        print("Debug: Failed to parse any valid JSON.")
        return []
    
    def get_chunking_suggestions(self, text: str, max_chunk_size: int = 500) -> str:
        """Send text to Claude for chunking suggestions."""
        prompt = f"""
You are a legal document expert. I have a legal document and need suggestions on how to chunk it for a Retrieval-Augmented Generation (RAG) system. Please:

1. Analyze the document text and suggest logical chunks based on legal document structure (e.g., sections, clauses, paragraphs, or logical breaks in argumentation).
2. For each chunk, specify:
   - The exact start words (first 3-10 words of the chunk).
   - The exact end words (last 3-10 words of the chunk).
   - A brief contextual summary to situate the chunk within the broader document.
3. Do not split sentences between chunks.
5. Return the response in a structured JSON format with fields: `start_words`, `end_words`, and `contextual`.
6. Do not include any markdown, code fences, explanations, or additional text outside the JSON array.

Here is the document text:
```
{text}
```

Please provide the chunking suggestions in JSON format and nothing else.
"""
        try:
            response = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2500,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            response_text = response.content[0].text
            print("response_text: ", response_text)
            # Remove ```json at the start and ``` at the end, allowing for optional whitespace
            # response_text = re.sub(r'^\s*```json\s*\n|\s*```\s*$', '', response_text, flags=re.MULTILINE)
            # chunks = json.loads(response_text)
            #print("JSON is valid!")
            return response_text
        except AnthropicError as e:
            # st.error(f"Error with Anthropic API: {str(e)}")
            return ""
        # except json.JSONDecodeError as e:
            # st.error(f"Error parsing Claude's response as JSON: {str(e)}")
        #     return ""


    def validate_chunks(self, text: str, chunks: List[Dict]) -> List[Dict]:
        """Validate and refine chunks to ensure they match the original text."""
        validated_chunks = []
        for i, chunk in enumerate(chunks):
            start_words = chunk.get("start_words", "")
            end_words = chunk.get("end_words", "")
            contextual = chunk.get("contextual", "")
            print(f"Debug: Validating chunk {i+1}: start_words='{start_words}', end_words='{end_words}'")

            # Find start and end indices in the original text
            start_idx = text.find(start_words)
            if start_idx == -1:
                print(f"Warning: Chunk {i+1} starting with '{start_words}' not found in text. Skipping.")
                continue

            # Find end_words after start_idx
            end_idx = text.find(end_words, start_idx)
            if end_idx == -1:
                print(f"Warning: Chunk {i+1} ending with '{end_words}' not found after start_words. Skipping.")
                continue

            # Extract chunk text
            chunk_text = text[start_idx:end_idx + len(end_words)]
            print(f"Debug: Chunk {i+1} text extracted: {chunk_text[:100]}... (length: {len(chunk_text)})")

            # Verify start and end words match the chunk text
            actual_start = chunk_text[:len(start_words)].strip()
            actual_end = chunk_text[-len(end_words):].strip()
            if actual_start != start_words or actual_end != end_words:
                print(f"Warning: Mismatch in start/end words for chunk {i+1}. Adjusting.")
                # Adjust start_words and end_words based on chunk_text
                words = chunk_text.split()
                if len(words) >= 5:
                    start_words = " ".join(words[:5])
                    end_words = " ".join(words[-5:])
                else:
                    start_words = " ".join(words[:min(5, len(words))])
                    end_words = " ".join(words[-min(5, len(words)):])

            validated_chunks.append({
                "start_words": start_words,
                "end_words": end_words,
                "chunk_text": chunk_text,
                "contextual": contextual
            })
            print(f"Debug: Validated chunk {i+1}: start_words='{start_words}', end_words='{end_words}', chunk_text_length={len(chunk_text)}")

        print(f"Debug: Validation complete. Returning {len(validated_chunks)} chunks.")
        return validated_chunks

    def chunk_file(self, uploaded_file) -> List[Dict]:
        """Main function to chunk a PDF document using Claude."""
        """Chunk text from a PDF or HTML file."""
        
        text = ""
        # Determine file type
        file_name = uploaded_file.name.lower()
        if file_name.endswith('.pdf'):
            print("Debug: Processing PDF file")
            text = self.extract_text_from_pdf(uploaded_file)
        elif file_name.endswith(('.html', '.htm')):
            print("Debug: Processing HTML file")
            text = self.extract_text_from_html(uploaded_file)
        else:
            raise Exception("Unsupported file type. Please upload a PDF or HTML file.")
            
        if not text:
            return []

        text = self.clean_text(text)
    
        raw_response = self.get_chunking_suggestions(text)
        chunks = self.clean_and_parse_json(raw_response)
        validated_chunks = self.validate_chunks(text, chunks)
        return validated_chunks
