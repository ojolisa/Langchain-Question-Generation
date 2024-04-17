# RAG Web Application

RAG is a web application built with Streamlit that allows users to ask questions about a provided PDF document. The system utilizes the following steps:

## Workflow

1. **PDF Processing:** The script reads the uploaded PDF file using PyPDF2 and extracts the text content.
2. **Text Chunking:** The extracted text is split into smaller chunks with a random size between 200 and 3000 characters, along with a 10% overlap to ensure context preservation.
3. **Embedding Generation:** Ollama, a large language model (LLM), is used to generate embeddings (numerical representations) for each text chunk.
4. **Vectorstore Creation:** FAISS, a vector similarity search library, is used to create a vectorstore that indexes the text chunks based on their embeddings.
5. **Document Chain:** A document chain is created using LangChain, a library for building chat-style conversational AI. This chain utilizes Ollama to answer questions based on the provided context (extracted text from the PDF).
6. **Retrieval Chain:** A retrieval chain is built that searches the vectorstore for the most relevant text chunk(s) based on the user's question. The relevant context is then fed to the document chain for answer generation.
7. **Question Answering:** Depending on the button clicked (True-False, MCQ, One-word), the system formulates an appropriate question based on the context and retrieves the answer from the document chain.

## Requirements
A virtual environment should be created in the folder.

The following Python libraries are required:

- `langchain_community.llms`: For accessing Ollama LLM
- `langchain`: Core LangChain library for building conversational AI workflows
- `langchain_core.prompts`: For defining chat prompts
- `langchain_community.vectorstores`: For using FAISS vector similarity search
- `langchain_community.embeddings`: For Ollama embeddings
- `langchain.chains.combine_documents`: For creating document chain
- `langchain.chains`: For creating retrieval chain
- `langchain_text_splitters`: For chunking text
- `streamlit`: For the app
- `faiss-cpu`: For the vector store
- `pypdf2`: To read the pdf

Ollama should be installed with the Llama2 model. The Ollama server should be running.

## Steps

1. **Step 1:** In the terminal, write `python -m venv venv`
2. **Step 2:** In the terminal, write `.\venv\Scripts\activate` 
3. **Step 3:** In the terminal, write `pip install -r requirements.txt`
4. **Step 4:** Start the Ollama server
5. **Step 5:** In the terminal write `streamlit run streamlit_app.py` 
