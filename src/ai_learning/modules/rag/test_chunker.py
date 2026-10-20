import unittest
from chunker import chunk_text

text = """
Kafka is a distributed event streaming platform.
A Kafka topic is divided into partitions.
Each partition contains an ordered sequence of records.
Consumers read records from partitions.
A consumer group is a group of consumers cooperating
to consume records from Kafka topics.
"""


class TestChunkText(unittest.TestCase):
    def test_chunk_text(self):
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        for i, chunk in enumerate(chunks):
            print(f"\n========== CHUNK {i} ==========")
            print(chunk)


if __name__ == "__main__":
    unittest.main()
