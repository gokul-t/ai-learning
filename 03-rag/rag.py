import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Define local storage path for the vector store
PERSIST_DIRECTORY = "./chroma_db"
PDF_PATH = "/home/gokult/Documents/Study/Effective Java (2017, Addison-Wesley).pdf"

# Initialize the Ollama embedding model
embeddings = OllamaEmbeddings(model="nomic-embed-text:v1.5")

# Check if the database already exists on your hard drive
if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
    print("Found existing vector database. Loading from disk...")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings
    )
else:
    print("Vector store not found. Ingesting document independently...")

    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Could not find the target PDF file at: {PDF_PATH}")

    reader = PdfReader(PDF_PATH)
    raw_documents = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text.strip():
            doc = Document(
                page_content=text, metadata={"source": PDF_PATH, "page": page_num + 1}
            )
            raw_documents.append(doc)

    print(f"Successfully extracted text from {len(raw_documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(raw_documents)

    print("Generating embeddings and indexing to ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=embeddings, persist_directory=PERSIST_DIRECTORY
    )
    print("Embeddings successfully saved to disk.")

# 4. Create retriever (fetches top 3 matching chunks based on vector similarity)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 5. Initialize LLM (temperature=0 ensures deterministic, non-creative answers)
llm = ChatOllama(model="qwen3.5:9b", temperature=0)

# 6. Guardrailed RAG prompt template
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


# 7. Construct the execution chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 8. Interactive loop for the user
print("\n" + "=" * 50)
print("📚 Local PDF RAG Bot Initialized!")
print("Type your question below. Type 'exit' or 'quit' to close the app.")
print("=" * 50 + "\n")

while True:
    try:
        user_query = input("\n🧑 Ask a question: ").strip()

        # Exit condition
        if user_query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_query:
            continue

        print("🔍 Searching database and generating response...")

        # This triggers similarity search -> formats context -> executes LLM
        response = rag_chain.invoke(user_query)

        print(f"\n🤖 Answer:\n{response}")
        print("-" * 40)

    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
