import streamlit as st
import os
from embedding import PDFEmbeddingProcessor
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

def main():
    model_list = ["tinyllama"]
    st.title("PDF Chat Application")
    st.write("Upload a PDF file to interact with its content.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if "selected_model" not in st.session_state:
        st.session_state.selected_model = model_list[0]

    selected_model = st.selectbox("Select a model", model_list, index=model_list.index(st.session_state.selected_model))
    st.session_state.selected_model = selected_model

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = os.path.join("/tmp", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Get absolute path
        abs_file_path = os.path.abspath(temp_file_path)

        processor = PDFEmbeddingProcessor(abs_file_path, selected_model)
        processor.process()

        st.write("You can now ask questions about the content of the PDF.")
        query = st.text_input("Enter your question:")

        if query:
            retriever = processor.get_retriever()

            template = """Answer the question based only on the following context:
            {context}

            Question: {question}
            """
            prompt = ChatPromptTemplate.from_template(template)
            model = ChatOllama(model=selected_model,temperature=0,)

            retrieval_chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
            )

            answer = retrieval_chain.invoke(query)
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()