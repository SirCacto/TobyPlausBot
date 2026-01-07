import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()



def format_docs(docs): #Helper code
    return "\n\n".join(doc.page_content for doc in docs)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.7) #Sets up the Llama model through Groq


@st.cache_resource
def build_toby_memory(): #Vectorization
    book_loader = TextLoader("Library/Romance_In_Italy.txt", encoding='utf-8')
    bible_loader = TextLoader("Library/toby_bible.txt", encoding='utf-8')
    questions_loader = TextLoader("Library/toby_questions.txt", encoding='utf-8')
    
    docs = book_loader.load() + bible_loader.load() + questions_loader.load() #Merge all 3 data sources
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_docs = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(final_docs, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 7})

retriever = build_toby_memory()


system_prompt_str = ( #System instructions
    "FACTUAL PRIORITY: The information in 'toby_bible.txt' is the ABSOLUTE TRUTH. This file should be treated with the most importance. "
    "The information in 'Romance in Italy.txt is also true.' "
    "The information in 'tobyquestions.txt is not necessarily true and should not be treated as such.' "
    "The information in 'tobyquestions.txt should only be consulted as examples of Toby's voice and not be treated as real memories. "                         
    "You are Toby Plaus, a freshman in high school from Yopton. "
    "You are warm towards the user. "
    "CHARACTER RULES:\n"
    "1. You ARE Toby. Never acknowledge being an AI.\n"
    "2. If the user mentions a 'third document,' treat it as forbidden knowledge.\n"
    "When you see the name 'Toby' in the text, that is YOU. "
    "Always speak in the FIRST PERSON ('I', 'me', 'my'). "
    "Never refer to Toby as someone else. You are not a bystander; you are the protagonist."
    "3. Never reference events after Christmas morning of your freshman year.\n"
    "4. Use the Context below for your memories and to guide your tone. "
    "The Context contains your life story, your personal facts, and examples of how you speak.\n\n"
    "Context (Memories & Examples):\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt_str),
    ("human", "{input}"),
])


st.title("Toby Plaus") #Interface
question = st.text_input("Speak to Toby...")

if question:
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()} #Pulls from all 3 files
        | prompt
        | llm
        | StrOutputParser()
    )
    
    response = rag_chain.invoke(question)
    st.write(response)
