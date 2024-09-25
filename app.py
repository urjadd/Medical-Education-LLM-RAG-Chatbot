import random
import streamlit as st
from langchain import hub
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter
import csv
import random

def pull_random_questions():
    file_path = 'output.csv'
    questions = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            question = row[0]  # Assuming the question is in the first column
            questions.append(question)
    
    random_question = random.choice(questions)
    return random_question


llama = LlamaCppEmbeddings(model_path="D:\Models\TheBloke\Llama-2-7B-Chat-GGML\llama-2-7b-chat.ggmlv3.q8_0.bin")

# Load, chunk and index the contents of the blog.
loader = CSVLoader(file_path='output.csv')
docs = loader.load()

print(docs)

text_splitter = CharacterTextSplitter(
    separator=",",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False,
)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=llama)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llama
    | StrOutputParser()
)



st.title("⚕️ Medicine Learning Chatbot")
st.caption("🚀 A chatbot powered by Mistral & Chroma")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you study today?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if st.button("Generate Random Question from Flashcards"):
    st.session_state.rand_question = pull_random_questions()
    st.session_state["messages"] = [{"role": "assistant", "content": st.session_state.rand_question}]
    st.chat_message("assistant").write(st.session_state.rand_question)


if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    print(st.session_state.messages)
    messages = st.session_state.messages
    rag_content = rag_chain.run({"question": messages})
    answer = rag_content["output"]
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)