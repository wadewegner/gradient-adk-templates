from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_classic.tools.retriever import create_retriever_tool


def load_pdfs_from_folder(pdf_folder_path: str):
    """Load all PDF files from a specified folder."""
    pdf_path = Path(pdf_folder_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"Folder {pdf_folder_path} does not exist")

    pdf_files = list(pdf_path.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(f"No PDF files found in {pdf_folder_path}")

    print(f"Found {len(pdf_files)} PDF files")

    docs = []
    for pdf_file in pdf_files:
        print(f"Loading: {pdf_file.name}")
        loader = PyPDFLoader(str(pdf_file))
        docs.extend(loader.load())

    return docs


def create_document_splits(
    folder_path: str, chunk_size: int = 100, chunk_overlap: int = 50
):

    # Load documents from PDF folder
    docs_list = load_pdfs_from_folder(folder_path)

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    doc_splits = text_splitter.split_documents(docs_list)
    return doc_splits


def create_retriever(folder_path: str, chunk_size: int = 100, chunk_overlap: int = 50):
    doc_splits = create_document_splits(folder_path, chunk_size, chunk_overlap)
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever()

    # Create retriever tool
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_documents",
        "Search and return information from the document collection.",
    )

    return retriever_tool
