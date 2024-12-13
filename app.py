import pickle

with open('db.pkl', 'rb') as f:
    db = pickle.load(f)

class RAG_Chatbot:
    def __init__(self, db, llm):
        self.db=db
        self.llm=llm
    def generate(self, question):
        docs=self.db.similarity_search(question,k=3)
        context='\n'.join([doc.page_content for doc in docs])
        context_metadata='\n'.join([doc.metadata['source'] for doc in docs])
        prompt=f'''Use the following pieces of information to answer the user's question.
                  If you don't know the answer, just say that you don't know, don't try to make up an answer.
                  Context: {context}
                  Question: {question}
                  Only return the helpful answer below and nothing else'''
        response=self.llm(prompt)
        return response, context_metadata    

import streamlit as st
api_key='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'

from langchain.llms import HuggingFaceHub
llm = HuggingFaceHub(
    huggingfacehub_api_token=api_key,
    repo_id="google/flan-t5-large",
    model_kwargs={
        "temperature": 0.5,
        "top_p": 0.85,
        "max_length": 150  # Increase max_length for longer outputs
    }
)

rag=RAG_Chatbot(db,llm)

st.title('RAG Chatbot')
question=st.text_input('Question')
if question:
    response, context_metadata=rag.generate(question)
    st.write(response)