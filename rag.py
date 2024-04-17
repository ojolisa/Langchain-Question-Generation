# Importing the required libraries
from langchain_community.llms import Ollama
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_text_splitters import CharacterTextSplitter
import streamlit as st
from PyPDF2 import PdfReader
from typing_extensions import Concatenate
import random

# Setting the title of the web app
st.title("RAG")

# Intializing the Ollama model and the OllamaEmbeddings
llm = Ollama(model="llama2")
embeddings = OllamaEmbeddings()

# Reading the PDF file
pdfreader = PdfReader("Big Mac Index.pdf")
text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        text += content

# Setting the chunk size and overlap
chunk_size = random.randint(200,3000)
chunk_overlap = chunk_size//10
print(chunk_size,chunk_overlap)

#Splitting the text into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len
)
texts = text_splitter.split_text(text)

#Creating the vectorstore
vector = FAISS.from_texts(texts, embeddings)

#Creating the document chain
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
                            
<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

#Creating the retrieval chain
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

#Creating the buttons for the different types of questions
button1 = st.button('True-False Question')
button2 = st.button('MCQ Question')
button3 = st.button('One-word Question')

#Displaying the questions and answers based on the button clicked
if button1:
    question = retrieval_chain.invoke(
        {'input': "Write a true-false question based on the given context.Give the question only and do not answer"})
    with st.expander("Question"):
        st.write(question["answer"])
    answer = retrieval_chain.invoke(
        {'input': question["answer"]+"Give the answer in True-False only.Do not give any explanation."})
    with st.expander("Answer"):
        st.write(answer["answer"])

if button2:
    question = retrieval_chain.invoke(
        {'input': "Write a multiple choice question based on the given context.Give the question only and do not answer"})
    with st.expander("Question"):
        st.write(question["answer"])
    answer = retrieval_chain.invoke(
        {'input': question["answer"]+"Give the answer in multiple choice format.Do not give any explanation."})
    with st.expander("Answer"):
        st.write(answer["answer"])

if button3:
    question = retrieval_chain.invoke(
        {'input': "Write a question based on the given context.Give the question only and do not answer"})
    with st.expander("Question"):
        st.write(question["answer"])
    answer = retrieval_chain.invoke(
        {'input': question["answer"]+"Give the answer in one word.Do not give any explanation."})
    with st.expander("Answer"):
        st.write(answer["answer"])
