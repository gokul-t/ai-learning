from pathlib import Path
from embeddings import create_embedding
from chunker import chunk_text


def load_documents(directory="documents"):
    documents = []

    files = Path(directory).glob("**/*.txt")

    for file in files:
        content = file.read_text(encoding="utf-8")

        documents.append({"source": file.name, "text": content})

    return documents


def build_chunks(documents):

    chunks = []

    for document in documents:
        document_chunks = chunk_text(document["text"], chunk_size=1000, overlap=200)

        for chunk in document_chunks:
            chunks.append({"source": document["source"], "text": chunk})

    return chunks


def build_vector_store(chunks):

    vector_store = []

    for chunk in chunks:
        embedding = create_embedding(chunk["text"])

        vector_store.append(
            {"source": chunk["source"], "text": chunk["text"], "embedding": embedding}
        )

    return vector_store


def main():
    documents = load_documents()
    chunks = build_chunks(documents)
    vector_store = build_vector_store(chunks)

    print(vector_store)


if __name__ == "__main__":
    main()
