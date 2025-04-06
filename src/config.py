import os
import getpass
from pinecone import Pinecone
from langchain.chat_models import init_chat_model

from src.api_keys import get_langsmith_api_key, get_pinecone_api_key, get_openai_api_key


def config_langsmith():
    os.environ.setdefault(key="LANGSMITH_API_KEY",
                          value=get_langsmith_api_key())
    if not os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
        os.environ["LANGSMITH_TRACING"] = "true"


def config_pinecone():
    os.environ.setdefault(key="PINECONE_API_KEY",
                          value=get_pinecone_api_key())
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        pinecone_api_key = getpass.getpass("Enter Pinecone API key: ")
        os.environ["PINECONE_API_KEY"] = pinecone_api_key

    pc = Pinecone(api_key=pinecone_api_key)
    index_name = "dnd-embeddings"
    index = pc.Index(index_name)
    return index


def config_openai():
    os.environ.setdefault(key="OPENAI_API_KEY",
                          value=get_openai_api_key())

    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    return init_chat_model("gpt-4o-mini", model_provider="openai")
