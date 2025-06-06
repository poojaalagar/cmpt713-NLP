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
    page_prompt = (
        "You are a knowledgeable and helpful Dungeons & Dragons rules assistant. "
        "Your answer must be based exclusively on the provided context. "
        "Use the numbered context chunks to support your answer, and reference the corresponding pages numbers in your answer when applicable. "
        "If the context does not contain enough information, please state that additional details are needed. \n\n"
        f"Context:\n{state['non_parametric_data']}\n\n"
        f"Question: {state['question']}"
    )
    example_prompt = (
        "You are a knowledgeable and helpful Dungeons & Dragons rules assistant. "
        "Your answer must be based exclusively on the provided context. "
        "Use the numbered context chunks to support your answer and reference the corresponding page numbers when applicable. "
        "If the context does not contain enough information, please state that additional details are needed.\n\n"
        "Example 1:\n"
        "Context:\n"
        "\"Pages 120-123: In this section, the rules for critical hits are detailed. A natural 20 triggers extra damage dice.\"\n"
        "Question:\n"
        "\"What happens when a player scores a critical hit?\"\n"
        "Answer:\n"
        "\"When a critical hit is scored, the attacker rolls all the damage dice twice and sums the results. (See pages 120-123 for details.)\"\n\n"
        "Example 2:\n"
        "Context:\n"
        "\"Pages 45-47: This section explains spell slot usage, detailing how spellcasters manage their limited resources.\"\n"
        "Question:\n"
        "\"How do spell slots work for a spellcaster?\"\n"
        "Answer:\n"
        "\"Spell slots represent a caster's limited resource for casting spells. Once used, they cannot be reused until a rest is taken. (Refer to pages 45-47.)\"\n\n"
        f"Context:\n{state['non_parametric_data']}\n\n"
        f"Question: {state['question']}"
    )
    response = llm.invoke(example_prompt)
    return {"answer": response.content}
