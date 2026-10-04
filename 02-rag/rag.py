import ollama
from vector_store import vector_store
from embeddings import create_embedding
from similarity import cosine_similarity


def generate_answer(query, vector_store):

    results = search(
        vector_store,
        query,
        top_k=3,
    )

    context = "\n\n".join(result["text"] for result in results)

    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{query}

If the answer cannot be found in the context,
say that you don't know.

Answer:
"""

    print("\nPrompt:\n", prompt)

    response = ollama.chat(
        model="qwen3.5:0.8b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response["message"]["content"]


def search(vector_store, query, top_k=3):

    query_embedding = create_embedding(query)

    results = []

    for item in vector_store:
        similarity = cosine_similarity(query_embedding, item["embedding"])

        if similarity > 0.0:
            results.append(
                {
                    "source": item["source"],
                    "text": item["text"],
                    "similarity": similarity,
                }
            )

    results.sort(key=lambda item: item["similarity"], reverse=True)

    return results[:top_k]


def main():

    while True:
        query = input("\nYou: ")

        if query.lower() == "exit":
            break

        answer = generate_answer(
            query,
            vector_store,
        )

        print("\nAI:", answer)


if __name__ == "__main__":
    main()
