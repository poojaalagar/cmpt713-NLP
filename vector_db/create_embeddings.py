import json
from tqdm import tqdm
from query_processing import embed_text


def main():
    # Load your chunks
    with open("../data/merged_chunks_by_size_output.jsonl", "r") as f:
        chunks = [json.loads(line) for line in f]

    # Create embeddings
    for chunk in tqdm(chunks):
        chunk["embedding"] = embed_text(chunk["text"])

    # Optionally save to file
    with open("../data/embedded_chunks.jsonl", "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")


if __name__ == "__main__":
    main()
