import json


def merge_chunks_by_size(input_path, output_path, min_chars=500):
    merged_chunks = []
    buffer = {"text": "", "metadata": {"page_numbers": [], "section_titles": [], "element_ids": []}}
    current_page = None

    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            chunk = json.loads(line)
            page = chunk["metadata"].get("page_numbers", [None])[0]

            # If page changes or buffer has enough characters, flush it
            if current_page is not None and (page != current_page or len(buffer["text"]) >= min_chars):
                merged_chunks.append(buffer)
                buffer = {"text": "", "metadata": {"page_numbers": [], "section_titles": [], "element_ids": []}}

            current_page = page

            # Append to buffer
            buffer["text"] += ("\n\n" if buffer["text"] else "") + chunk["text"]
            buffer["metadata"]["page_numbers"] = list(
                set(buffer["metadata"]["page_numbers"] + chunk["metadata"].get("page_numbers", [])))
            section_title = chunk["metadata"].get("section_title")
            if section_title:
                buffer["metadata"]["section_titles"].append(section_title)
            buffer["metadata"]["element_ids"].extend(chunk["metadata"].get("element_ids", []))

        # Final flush
        if buffer["text"]:
            merged_chunks.append(buffer)

    # Save merged chunks
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for chunk in merged_chunks:
            json.dump(chunk, outfile)
            outfile.write('\n')


if __name__ == "__main__":
    merge_chunks_by_size("../data/combined_chunks_output.jsonl", "../data/merged_chunks_by_size_output.jsonl", min_chars=500)
