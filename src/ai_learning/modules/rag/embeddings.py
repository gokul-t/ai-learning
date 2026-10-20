import ollama

EMBEDDING_MODEL = "nomic-embed-text:v1.5"


def create_embedding(text):
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response["embeddings"][0]
