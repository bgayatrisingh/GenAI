import os
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"

def create_langchain_qa():
    # Step 1: Load the LangChain website content
    # Remove the 'features' from bs_kwargs to avoid the conflict
    loader = WebBaseLoader(
        web_paths=["https://langchain.com/",
                  "https://langchain.com/about",
                  "https://langchain.com/use-cases",
                  "https://python.langchain.com/docs/concepts/",
                  "https://www.edyoda.com/faq",
                  "https://python.langchain.com/docs/get_started"],
        bs_kwargs={},  # Empty dictionary to avoid conflicts
        header_template={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}
    )
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        verbose=True
    )

    return qa_chain

def ask_question(qa_chain, question):
    """Ask a question to the QA system"""
    result = qa_chain({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.page_content[:100] + "..." for doc in result["source_documents"]]
    }

# Example usage
if __name__ == "__main__":
    qa_system = create_langchain_qa()

    # Ask some questions
    questions = [
        "What is LangChain?",
        "How do I use LangChain with RAG?",
        "What are the key components of LangChain?",
        "How do I schedule my batch?"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        response = ask_question(qa_system, question)
        print(f"Answer: {response['answer']}")
        print("\nSources:")
        for i, source in enumerate(response['sources']):
            print(f"{i+1}. {source}")
