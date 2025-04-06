import json
import statistics
from pathlib import Path

# Change this to the path of your file
file_path = Path("../data/merged_chunks_by_size_output.jsonl")

char_lengths = []
word_lengths = []

with file_path.open("r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        text = data.get("text", "")
        char_lengths.append(len(text))
        word_lengths.append(len(text.split()))


def print_stats(name, values):
    print(f"{name} stats:")
    print(f"  Count: {len(values)}")
    print(f"  Min: {min(values)}")
    print(f"  Max: {max(values)}")
    print(f"  Mean: {statistics.mean(values):.2f}")
    print(f"  Median: {statistics.median(values)}")
    print()


print_stats("Character length", char_lengths)
print_stats("Word count", word_lengths)
