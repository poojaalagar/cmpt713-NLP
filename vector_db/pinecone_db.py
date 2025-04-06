import pinecone
import json
from vector_db.create_embeddings import embed_text
from api_keys import get_pinecone_api_key

# Initialize Pinecone with the new API
index_name = "dnd-embeddings"
pc = pinecone.Pinecone(api_key=get_pinecone_api_key())
try:
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec={
            "serverless": {
                "cloud": "aws",
                "region": "us-east-1"
            }
        }
    )
except pinecone.PineconeApiException as e:
    if "ALREADY_EXISTS" in str(e):
        print(f"Index '{index_name}' already exists.")
    else:
        raise

index = pc.Index(index_name)

describe = index.describe_index_stats()
if describe["total_vector_count"] == 0:
    print("Index is empty. Populating it...")

    vectors = []
    with open("../data/embedded_chunks.jsonl", "r") as f:
        for i, line in enumerate(f):
            chunk = json.loads(line)
            vector = {
                "id": f"chunk-{i}",
                "values": chunk["embedding"],
                "metadata": {
                    "text": chunk["text"],
                    "page_numbers": [str(int(p)) for p in chunk.get("metadata", {}).get("page_numbers", [])],
                    "section_titles": chunk.get("metadata", {}).get("section_titles", [])
                }
            }
            vectors.append(vector)

    BATCH_SIZE = 100

    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i + BATCH_SIZE]
        index.upsert(vectors=batch)
        print(f"Upserted batch {i // BATCH_SIZE + 1} ({len(batch)} vectors)")
else:
    print("Index already contains data. Skipping upsert.")


def query_embedding(query_text, top_k=5):
    query_vector = embed_text(query_text)
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    return results


# Test the query with some sample text
sample_results = query_embedding("Please tell me about the Barbarian class.")
print(sample_results)
