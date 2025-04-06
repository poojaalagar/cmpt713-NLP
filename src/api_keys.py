import os

def get_openai_api_key():
    return os.environ.get("OPENAI_API_KEY", "")

def get_pinecone_api_key():
    return os.environ.get("PINECONE_API_KEY", "")

def get_langsmith_api_key():
    return os.environ.get("LANGSMITH_API_KEY", "")
