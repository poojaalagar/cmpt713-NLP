import requests
from functools import partial
from langgraph.graph import START, StateGraph
from src.state import State
from src.config import config_langsmith, config_pinecone, config_openai
from src.query_processing import generate_answer, fetch_query_vector, fetch_matches_from_vectorstore


def initialize_dependencies():
    """Initialize external dependencies and return configured instances."""
    config_langsmith()
    index = config_pinecone()
    llm = config_openai()
    return index, llm


def build_graph(vectorstore, llm, top_k=8):
    """Construct the StateGraph for query execution."""
    graph_builder = StateGraph(State).add_sequence([
        ("User_query", lambda state: {"question": state["question"]}),
        ("Fetch_query_vector", partial(fetch_query_vector)),
        ("Fetch_matches_from_vectorstore", partial(fetch_matches_from_vectorstore,
                                                   vectorstore=vectorstore,
                                                   top_k=top_k)),
        ("Generate_answer", partial(generate_answer, llm=llm))
    ])
    graph_builder.add_edge(START, "User_query")
    return graph_builder.compile()


def run_query(graph, question):
    """Run the StateGraph with a given question and display the results."""
    for step in graph.stream({"question": question}, stream_mode="updates"):
        print(step)


def create_and_save_langchain_diagram(graph):
    # Save the Mermaid diagram
    graph_image_path = "graph.png"
    try:
        mermaid_image = graph.get_graph().draw_mermaid_png()
    except requests.exceptions.ReadTimeout:
        print("Mermaid diagram generation timed out. Skipping...")
        mermaid_image = None

    # Save the image to a file
    with open(graph_image_path, "wb") as f:
        f.write(mermaid_image)
