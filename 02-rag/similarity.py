import math
from vector_store import vector_store


def cosine_similarity(a, b):
    dot_product = sum(x * y for x, y in zip(a, b))

    magnitude_a = math.sqrt(sum(x * x for x in a))

    magnitude_b = math.sqrt(sum(x * x for x in b))

    return dot_product / (magnitude_a * magnitude_b)


def main():
    texts = [
        "Kafka consumer groups allow consumers to share partitions.",
        "Consumers in a Kafka group cooperate to process messages.",
        "Amazon S3 is an object storage service.",
    ]
    embeddings = []

    from embeddings import create_embedding

    for text in texts:
        vector = create_embedding(text)
        embeddings.append(vector)

        print(text)
        print("Dimensions:", len(vector))
        print()
    similarity = cosine_similarity(
        embeddings[0],
        embeddings[1],
    )

    print(similarity)
    similarity = cosine_similarity(
        embeddings[0],
        embeddings[2],
    )
    print(similarity)


if __name__ == "__main__":
    main()
