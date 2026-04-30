from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
