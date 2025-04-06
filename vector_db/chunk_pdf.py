import json
import os
import asyncio
from unstructured.partition.pdf import partition_pdf
from unstructured_client import UnstructuredClient
from unstructured_client.models import operations, shared
from rapidfuzz import fuzz
from collections import defaultdict
from typing import List


def is_junk_element(el):
    text = getattr(el, "text", "").strip()
    el_category = getattr(el, "category", None)
    if el_category == "UncategorizedText":
        try:
            if isinstance(int(text), int):
                return True
        except ValueError:
            pass
    if text == "Not for resale. Permission granted to print or photocopy this document for personal use only.":
        return True
    if text == "System Reference Document 5.1":
        return True
    return False


def split_and_sort_elements_by_page(elements, split_x=850):
    pages = defaultdict(list)
    for el in elements:
        page = getattr(el.metadata, "page_number", 0)
        pages[page].append(el)

    sorted_elements = []

    for page_num in sorted(pages.keys()):
        page_elements = pages[page_num]

        left = []
        right = []

        for el in page_elements:
            coords = getattr(el.metadata, "coordinates", None)
            coords = coords.points if coords else None
            if coords:
                x0 = coords[0][0]
                if x0 < split_x:
                    left.append(el)
                else:
                    right.append(el)

        def y0(el):
            coords = getattr(el.metadata, "coordinates", None)
            coords = coords.points if coords else None
            return coords[0][1] if coords else float("inf")

        left_sorted = sorted(left, key=y0)
        right_sorted = sorted(right, key=y0)

        sorted_elements.extend(left_sorted + right_sorted)

    return sorted_elements


async def get_all_table_html_from_api(file_path):
    client = UnstructuredClient(api_key_auth=os.getenv("UNSTRUCTURED_API_KEY"))
    with open(file_path, "rb") as f:
        files = shared.Files(
            content=f.read(),
            file_name=os.path.basename(file_path)
        )

    params = shared.PartitionParameters(
        files=files,
        strategy=shared.Strategy.VLM,
        vlm_model="gpt-4o",
        vlm_model_provider="openai",
        split_pdf_page=True,
        split_pdf_allow_failed=True,
    )

    request = operations.PartitionRequest(partition_parameters=params)
    result = await client.general.partition_async(request=request)

    return [
        el
        for el in result.elements
        if el["type"] == "Table" and "text_as_html" in el["metadata"]
    ]


def enrich_tables_with_html(elements, file_path, threshold=70):
    loop = asyncio.get_event_loop()
    api_tables = loop.run_until_complete(get_all_table_html_from_api(file_path))

    tables = [ele for ele in elements if getattr(ele, "category") == "Table"]
    for el in tables:
        el_text = el.text.strip()

        best_match = None
        best_score = 0

        for api_el in api_tables:
            api_text = api_el.get("text", "").strip()

            score = fuzz.partial_ratio(el_text, api_text)
            if score > best_score:
                best_match = api_el
                best_score = score

        if best_match and best_score >= threshold:
            setattr(el, "text_as_html", best_match.get("metadata", {}).get("text_as_html"))


def combine_elements_for_rag(elements: List) -> List[dict]:
    combined_chunks = []
    current_chunk = None

    def flush_chunk():
        nonlocal current_chunk
        if current_chunk and current_chunk["texts"]:
            combined_chunks.append({
                "text": "\n".join(current_chunk["texts"]),
                "metadata": {
                    "page_numbers": sorted(current_chunk["metadata"]["page_numbers"]),
                    "section_title": current_chunk["metadata"]["section_title"],
                    "element_ids": current_chunk["metadata"]["element_ids"],
                }
            })
            current_chunk = None

    prev_was_title = False

    for el in elements:
        el_category = getattr(el, "category", None)
        text = getattr(el, "text", "").strip()
        page_num = getattr(el.metadata, "page_number", None)
        el_id = getattr(el, "element_id", None)

        if el_category == "Table" and hasattr(el, "text_as_html"):
            text = f"\n\n[HTML_TABLE]\n{el.text_as_html}\n[/HTML_TABLE]"

        # Heuristic: reclassify title-like elements that are actually narrative text
        if el_category == "Title" and "." in text and len(text.split()) > 4:
            el_category = "NarrativeText"

        if el_category == "Title":
            if current_chunk is None or not prev_was_title:
                flush_chunk()
                current_chunk = {
                    "texts": [text],
                    "metadata": {
                        "page_numbers": set(),
                        "section_title": text,
                        "element_ids": [],
                    }
                }
            else:
                current_chunk["texts"].append(text)
            prev_was_title = True
        else:
            if current_chunk is None:
                current_chunk = {
                    "texts": [],
                    "metadata": {
                        "page_numbers": set(),
                        "section_title": None,
                        "element_ids": [],
                    }
                }
            current_chunk["texts"].append(text)
            prev_was_title = False

        if current_chunk:
            if page_num:
                current_chunk["metadata"]["page_numbers"].add(page_num)
            if el_id:
                current_chunk["metadata"]["element_ids"].append(el_id)

    flush_chunk()
    return combined_chunks


def main():
    file_path = "../data/SRD-OGL_V5.1.pdf"
    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        include_metadata=True,
        pdf_infer_table_structure=True
    )
    all_elements = split_and_sort_elements_by_page(elements)
    all_elements = [el for el in all_elements if not is_junk_element(el)]
    enrich_tables_with_html(all_elements, file_path)
    combined_chunks = combine_elements_for_rag(all_elements)
    for chunk in combined_chunks:
        print(chunk["metadata"].get("section_title"))
        print(chunk["text"])
        print("---")

    with open("../data/combined_chunks_output.jsonl", "w") as f:
        for chunk in combined_chunks:
            json.dump(chunk, f)
            f.write("\n")


if __name__ == "__main__":
    main()
