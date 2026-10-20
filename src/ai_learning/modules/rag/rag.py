import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from ai_learning.core.config import config
from ai_learning.core.logging import setup_logging

logger = setup_logging()

# Configuration
PERSIST_DIRECTORY = config.PROJECT_ROOT / "data" / "chroma_db"
DEFAULT_PDF_PATH = "/home/gokult/Documents/study/java/Effective Java (2017, Addison-Wesley).pdf"

def initialize_rag(pdf_path=DEFAULT_PDF_PATH):
    """Initializes the RAG system: extracts text, indexes it, and returns a chain."""

    embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")

    if PERSIST_DIRECTORY.exists() and any(PERSIST_DIRECTORY.iterdir()):
        logger.info("Found existing vector database. Loading from disk...")
        vectorstore = Chroma(
            persist_directory=str(PERSIST_DIRECTORY), embedding_function=embeddings
        )
    else:
        logger.info("Vector store not found. Ingesting document...")

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Could not find the target PDF file at: {pdf_path}")

        reader = PdfReader(pdf_path)
        raw_documents = []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text.strip():
                doc = Document(
                    page_content=text, metadata={"source": pdf_path, "page": page_num + 1}
                )
                raw_documents.append(doc)

        logger.info(f"Successfully extracted text from {len(raw_documents)} pages.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(raw_documents)

        logger.info("Generating embeddings and indexing to ChromaDB...")
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=embeddings, persist_directory=str(PERSIST_DIRECTORY)
        )
        logger.info("Embeddings successfully saved to disk.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOllama(model="qwen3.5:9b", temperature=0)

    prompt_template = """
    You are a strict QA assistant. Your task is to answer the user's question using ONLY the provided text context fragments below.

    CRITICAL INSTRUCTIONS:
    1. If the context fragments do not contain the direct facts or answer to the question, you MUST reply exactly with: "I don't know."
    2. Do NOT use your own outside knowledge, pre-training data, or assumptions to fabricate an answer.
    3. Keep your answers brief, factual, and directly sourced from the text.

    Context fragments:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def run_rag_cli():
    """Interactive loop for the RAG bot."""
    try:
        chain = initialize_rag()
        print("\n" + "=" * 50)
        print("📚 Local PDF RAG Bot Initialized!")
        print("Type your question below. Type 'exit' or 'quit' to close the app.")
        print("=" * 50 + "\n")

        while True:
            user_query = input("\n🧑 Ask a question: ").strip()
            if user_query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            if not user_query:
                continue

            print("🔍 Searching database and generating response...")
            response = chain.invoke(user_query)
            print(f"\n🤖 Answer:\n{response}")
            print("-" * 40)

    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"RAG Error: {e}")

if __name__ == "__main__":
    run_rag_cli()
