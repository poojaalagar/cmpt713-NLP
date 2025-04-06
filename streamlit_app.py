import streamlit as st
import base64
from src.pipeline import (
    initialize_dependencies,
    build_graph,
    run_query,
    create_and_save_langchain_diagram,
)

@st.cache_resource(show_spinner=False)
def get_dependencies():
    return initialize_dependencies()

st.markdown(
    """
    <style>
    /* Increase the font size for all labels (including the text area label) */
    label {
        font-size: 48px !important;
    }
    /* Make the submit button full width with increased font size and height */
    div.stButton > button {
        width: 100% !important;
        height: 50px;
        font-size: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def get_base64_of_bin_file(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

image_path = "data/peakpx.jpg"
bin_str = get_base64_of_bin_file(image_path)

page_bg_img = f"""
<style>
.stApp {{
    background-image: url("data:image/jpeg;base64,{bin_str}");
    background-size: cover;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

def main():
    st.title("D&D RAG Assistant")
    
    query = st.text_area("Enter your question about D&D rules:", value="")
    placeholder = st.empty()
    
    if st.button("Submit"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            try:
                st.info("Initializing dependencies...")
                index, llm = get_dependencies()
                graph = build_graph(vectorstore=index, llm=llm, top_k=8)
                st.info("Running query...")
                
                for step in graph.stream({"question": query}, stream_mode="updates"):
                    placeholder.write(step)
                
                create_and_save_langchain_diagram(graph)
                st.success("Query completed and diagram saved.")
            except Exception as e:
                st.error(f"Error during query execution: {e}")

if __name__ == "__main__":
    main()