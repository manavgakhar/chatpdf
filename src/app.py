import streamlit as st
from embedding import PDFEmbeddingProcessor
from langchain_ollama.llms import OllamaLLM
import os

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

        llm = OllamaLLM(model=selected_model)
        processor = PDFEmbeddingProcessor(abs_file_path, selected_model)
        processor.process()

        st.write("You can now ask questions about the content of the PDF.")
        query = st.text_input("Enter your question:")

        if query:
            answer = llm.invoke(query)
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()