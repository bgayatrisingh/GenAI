import os
from langchain_community.document_loaders import SitemapLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

#!pip install langchain-community langchain-text-splitters langchain-openai langchain-chroma openai requests beautifulsoup4

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Set USER_AGENT environment variable
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"

def extract_urls_from_domain(domain):
    """Extracts all URLs from a given domain."""
    urls = set()
    try:
        response = requests.get(domain)
        soup = BeautifulSoup(response.content, 'html.parser')

        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(domain, href)
            if full_url.startswith(domain):
                urls.add(full_url)

    except requests.exceptions.RequestException as e:
        print(f"Error while fetching {domain}: {e}")

    return list(urls)

def create_langchain_qa(urls):
    """Creates a LangChain QA system based on the extracted URLs."""

    loader = WebBaseLoader(
        web_paths=urls,
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
    """Asks a question to the QA system."""
    result = qa_chain({"query": question})
    return {
        "answer": result["result"],
        "sources": [doc.page_content[:100] + "..." for doc in result["source_documents"]]
    }

if __name__ == "__main__":
    domain = "https://www.urbanvault.in/"  # Replace with your domain

    urls = extract_urls_from_domain(domain)
    print(f"Extracted URLs: {urls}")
    qa_system = create_langchain_qa(urls)

    # Ask some questions
    questions = [
        "What are the best features of urbanvault?",
        "Why is urbanvault special?"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        response = ask_question(qa_system, question)
        print(f"Answer: {response['answer']}")
        print("\nSources:")
        for i, source in enumerate(response['sources']):
            print(f"{i+1}. {source}")

