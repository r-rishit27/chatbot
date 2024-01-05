import streamlit as st
from lang import get_qa_chain, create_vector_db

st.title(" Personalised Assistant ‍💻")
btn = st.button("Create Knowledgebase")
uploaded_file = st.file_uploader("Choose an document...", type=["pdf"])
if btn:
    create_vector_db(uploaded_file)

question = st.text_input(" Ask Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Response: ")
    st.write(response["result"])
