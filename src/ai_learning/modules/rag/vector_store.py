from loader import load_documents, build_chunks, build_vector_store

documents = load_documents()

chunks = build_chunks(documents)

vector_store = build_vector_store(chunks)
