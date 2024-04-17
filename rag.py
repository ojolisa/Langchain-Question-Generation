# Importing the required libraries
from langchain_community.llms import Ollama
from langchain.chains.llm import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_text_splitters import CharacterTextSplitter
from PyPDF2 import PdfReader
import random

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
chunk_size = random.randint(200, 3000)
chunk_overlap = chunk_size//10

# Splitting the text into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len
)
texts = text_splitter.split_text(text)

# Creating the vectorstore
vector = FAISS.from_texts(texts, embeddings)

# Creating the document chain
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:
                            
<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)

# Creating the retrieval chain
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Getting the user's choice for the type of question
print("Choose the type of question:")
print("1. True-False Question")
print("2. MCQ Question")
print("3. One-word Question")
choice = input("Enter your choice (1-3): ")

# Displaying the questions and answers based on the user's choice
if choice == '1':
    question = retrieval_chain.invoke(
        {'input': "Write a true-false question based on the given context.Give the question only and do not answer"})
    print("Question:")
    print(question["answer"])
    answer = retrieval_chain.invoke(
        {'input': question["answer"]+"Give the answer in True-False only.Do not give any explanation."})
    print("Answer:")
    print(answer["answer"])

elif choice == '2':
    question = retrieval_chain.invoke(
        {'input': "Write a multiple choice question based on the given context.Give the question only and do not answer"})
    print("Question:")
    print(question["answer"])
    answer = retrieval_chain.invoke(
        {'input': question["answer"]+"Give the answer in multiple choice format.Do not give any explanation."})
    print("Answer:")
    print(answer["answer"])

elif choice == '3':
    question = retrieval_chain.invoke(
        {'input': "Write a question based on the given context.Give the question only and do not answer"})
    print("Question:")
    print(question["answer"])
    answer = retrieval_chain.invoke(
        {'input': question["answer"]+"Give the answer in one word.Do not give any explanation."})
    print("Answer:")
    print(answer["answer"])
