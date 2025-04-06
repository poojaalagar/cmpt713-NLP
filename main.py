from src.pipeline import initialize_dependencies, build_graph, run_query, create_and_save_langchain_diagram


def main():
    """Main entry point for the program."""
    query = "Can you tell me about Meteor Swarm? Who can use this ability? What does it do?"
    index, llm = initialize_dependencies()
    graph = build_graph(vectorstore=index, llm=llm, top_k=8)
    run_query(graph=graph, question=query)

    create_and_save_langchain_diagram(graph)


if __name__ == "__main__":
    main()
