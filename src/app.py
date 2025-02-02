import streamlit as st
from utils import load_docs, split_documents, get_embeddings, create_rag_chain

def main():
    st.title("PDF Chat Application")
    st.write("Upload a PDF file to interact with its content.")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        
        documents = load_docs(uploaded_file)
        chunks = split_documents(documents)
        embeddings = get_embeddings(chunks)
        rag_chain = create_rag_chain(embeddings)

        st.write("You can now ask questions about the content of the PDF.")
        query = st.text_input("Enter your question:")

        if query:
            answer = rag_chain.invoke(query)
            st.write("Answer:", answer)

if __name__ == "__main__":
    main()