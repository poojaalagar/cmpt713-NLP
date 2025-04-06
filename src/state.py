from typing_extensions import TypedDict


class State(TypedDict):
    question: str
    query_vector: list
    non_parametric_data: str
    answer: str
