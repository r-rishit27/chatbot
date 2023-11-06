import streamlit as st
from lang import get_qa_chain, create_vector_db

st.title(" Your Personalised AI/ML Guide 👨‍💻")
btn = st.button("Create Knowledgebase")
if btn:
    create_vector_db()

question = st.text_input(" Ask Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])