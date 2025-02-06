from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st


def extract_texts(files):
    text = ""
    for file in files:
        ext = os.path.splitext(file.name)[-1]
        match ext:
            case '.pdf':
                text += extract_pdf_text(file)
                break
            case '.docx':
                text += extract_docx_text(file)
                break
            case '.csv':
                text += extract_csv_text(file)
                break
    return text

def extract_pdf_text(file):
    from pypdf import PdfReader

    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_docx_text(file):
    from docx import Document

    doc = Document(file)
    text = []
    for para in doc.paragraphs:
        text.append(para.text)
    return ' '.join(text)

def extract_csv_text(file):
    from csv import reader
    
    text = []
    with open(file, 'r', newline='') as f:
        csv_reader = reader(file, delimiter=',')
        for row in csv_reader:
            text.append(', '.join(row))
    return text

def create_chunks(texts):
    from langchain_text_splitters import CharacterTextSplitter

    splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_text(texts)

def make_vec_db(chunks):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    return FAISS.from_texts(chunks, HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'))

def make_conversation_chain(vecDB):
    from langchain_ollama import ChatOllama
    
    from langchain.chains.history_aware_retriever import create_history_aware_retriever
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain

    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    
    llm = ChatOllama(model='deepseek-r1:1.5b', temperature=0)
    rephrase_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use this context to answer the question at the end:\nSTART CONTEXT\n{context}\nEND CONTEXT\n"),
        ("human", "Question: {input}")
    ])

    retriever = create_history_aware_retriever(llm, vecDB.as_retriever(), rephrase_prompt)

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    retrieval_chain = create_retrieval_chain(retriever, qa_chain)
    
    return retrieval_chain

def handle_query(q):
    from streamlit_chat import message
    from langchain_core.messages import HumanMessage, AIMessage

    resp = st.session_state['chain'].invoke({'input': q, 'chat_history': st.session_state['chat_history']})
    st.session_state['chat_history'].extend([
        HumanMessage(content=q),
        AIMessage(content=resp['answer'])
    ])

    cont = st.container()
    with cont:
        for i, msg in enumerate(st.session_state['chat_history']):
            message(msg.content, is_user=(i % 2 == 0), key=str(i))

def main():
    st.set_page_config(page_title="Chat with Documents")
    st.header("DocGPT")

    for key, default in [('chain', None), ('chat_history', []), ('doneProcessing', False)]:
        if key not in st.session_state:
            st.session_state[key] = default
    
    with st.sidebar:
        uploaded = st.file_uploader("Upload Files", type=['pdf', 'docx', 'csv'], accept_multiple_files=True)
        
        if st.button("Process Files"):
            texts = extract_texts(uploaded)
            st.write("Loaded Files")
            chunks = create_chunks(texts)
            st.write("Done Chunking")
            vecDB = make_vec_db(chunks)
            st.write("VectorDB Created")
            
            st.session_state['chain'] = make_conversation_chain(vecDB)
            st.session_state['doneProcessing'] = True
    
    if st.session_state['doneProcessing']:
        q = st.chat_input("Ask a Question")
        if q:
            handle_query(q)

if __name__ == "__main__":
    main()