from openai import OpenAI
from src.api_keys import get_openai_api_key
from src.state import State


def embed_text(text: str, model="text-embedding-ada-002") -> list:
    client = OpenAI(
        api_key=get_openai_api_key()
    )
    response = client.embeddings.create(input=[text], model=model)
    return response.data[0].embedding


def fetch_query_vector(state: State):
    """Generate a query vector from the user's question."""
    query = state["question"]
    query_vector = embed_text(query)
    return {"query_vector": query_vector}


def fetch_matches_from_vectorstore(state: State, vectorstore, top_k=8):
    """Retrieve relevant context using vector similarity search."""
    query_vector = state["query_vector"]
    matches = vectorstore.query(vector=query_vector, top_k=top_k, include_metadata=True)

    context_chunks = []
    for match in matches["matches"]:
        chunk_id = match.get("id", "unknown-chunk")
        metadata = match["metadata"]
        section_titles = metadata.get("section_titles", [])
        page_numbers = metadata.get("page_numbers", [])
        text = metadata.get("text", "")
        formatted_chunk = \
            f"""
            Pages: {', '.join(page_numbers) if page_numbers else 'N/A'}
            Chunk ID: {chunk_id}
            Section: {', '.join(section_titles) if section_titles else 'N/A'}
            {text}"""

        context_chunks.append(formatted_chunk)

    context = "\n\n---\n\n".join(context_chunks)  # clean separation between chunks
    return {"non_parametric_data": context}


def generate_answer(state: State, llm):
    """Answer question using retrieved context."""
    prompt = (
        "You are a helpful Dungeons & Dragons rules assistant. "
        "Use the numbered context chunks below to answer the user's question. "
        "Please note the page(s) used to justify your answer.\n\n"
        f"Context:\n{state['non_parametric_data']}\n\n"
        f"Question: {state['question']}"
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}
