import os
from pypdf import PdfReader
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

load_dotenv()

DATA_DIR = "data"
INDEX_DIR = "indices"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def get_pdf_text(filepath: str) -> List[Document]:
    """Extracts text from a single PDF file and returns it as a list of Document objects."""
    texts = []
    filename = os.path.basename(filepath)
    try:
        reader = PdfReader(filepath)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                sanitized_text = text.encode("utf-8", "ignore").decode("utf-8")
                texts.append(
                    Document(
                        page_content=sanitized_text,
                        metadata={
                            "source_file": filename,
                            "page": page_num + 1,
                        },
                    )
                )
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return texts


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Splits documents into chunks of a specified size with overlap using RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks


def build_and_save_vectorstore(chunks: List[Document], index_path: str):
    """Builds a FAISS vector store and saves it locally."""
    if not os.path.exists(index_path):
        os.makedirs(index_path)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", google_api_key=os.environ.get("GEMINI_API_KEY")
    )

    if not chunks:
        print(f"No chunks provided for {index_path}. Skipping vector store creation.")
        return

    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(index_path)
    print(f"Vector store saved to {index_path}")


def ingest_pdfs():
    """Full pipeline for ingesting specific PDFs and building separate vector stores."""
    print("Starting PDF ingestion...")

    target_pdfs = [
        "Hands-On_Large_Language_Models.pdf",
        "Hands-On_Machine_Learining.pdf",
    ]

    for pdf_filename in target_pdfs:
        pdf_filepath = os.path.join(DATA_DIR, pdf_filename)
        if not os.path.exists(pdf_filepath):
            print(f"Warning: {pdf_filepath} not found. Skipping.")
            continue

        print(f"\nProcessing {pdf_filename}...")

        raw_documents = get_pdf_text(pdf_filepath)
        if not raw_documents:
            print(f"No text could be extracted from {pdf_filename}. Skipping.")
            continue
        print(f"Extracted text from {len(raw_documents)} pages of {pdf_filename}.")

        chunks = chunk_documents(raw_documents)
        print(f"Created {len(chunks)} text chunks for {pdf_filename}.")

        pdf_name_without_ext = os.path.splitext(pdf_filename)[0]
        index_subdir = os.path.join(INDEX_DIR, pdf_name_without_ext)
        build_and_save_vectorstore(chunks, index_subdir)

    print("\nIngestion complete for all target PDFs.")


if __name__ == "__main__":
    if not os.path.exists(DATA_DIR) or not os.listdir(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        print(f"Created '{DATA_DIR}' directory. Please add your PDF files there.")
    elif not os.environ.get("GEMINI_API_KEY"):
        print("Please set the GEMINI_API_KEY environment variable.")
    else:
        ingest_pdfs()
