import os
import fitz
import chromadb
import argparse
import ollama

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

OLLAMA_MODEL = "llama3.2"


def extract_pdf_text(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"Failed to extract text from {file_path}: {e}")
        return ""


def transcribe_video(file_path: str) -> str:
    try:
        import whisper
        whisper_model = whisper.load_model("base")
        result = whisper_model.transcribe(file_path)
        return result["text"]
    except Exception as e:
        print(f"Failed to transcribe video {file_path}: {e}")
        return ""


def chunk_text(text: str, source: str) -> list[dict]:
    chunker = SemanticChunker(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    try:
        docs = chunker.create_documents([text])
        if not docs:
            raise ValueError("No chunks produced")
    except Exception:
        return [{"text": text, "source": source, "id": f"{source}_chunk_0"}]
    return [
        {"text": doc.page_content, "source": source, "id": f"{source}_chunk_{i}"}
        for i, doc in enumerate(docs)
    ]


def create_vector_store(collection_name: str = "rag_kb"):
    client = chromadb.PersistentClient(path="./chroma_db")
    return client.get_or_create_collection(collection_name)


def add_chunks(collection, chunks: list[dict]) -> None:
    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[{"source": c["source"]} for c in chunks],
    )


def query_store(collection, query: str, top_k: int = 10) -> list[str]:
    results = collection.query(query_texts=[query], n_results=top_k)
    return results["documents"][0]


def generate_answer(question: str, context_chunks: list[str]) -> str:
    context = "\n\n".join(context_chunks)
    prompt = (
        "Use the following context to answer the question. "
        "Base your answer only on the provided context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    client = ollama.Client()
    response = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.message.content


def ingest(collection, pdf_paths: list[str], video_paths: list[str]) -> None:
    for path in pdf_paths:
        print(f"Processing PDF: {path}")
        text = extract_pdf_text(path)
        chunks = chunk_text(text, os.path.basename(path))
        add_chunks(collection, chunks)

    for path in video_paths:
        print(f"Processing video: {path}")
        text = transcribe_video(path)
        chunks = chunk_text(text, os.path.basename(path))
        add_chunks(collection, chunks)


def ask(collection, question: str) -> str:
    chunks = query_store(collection, question)
    return generate_answer(question, chunks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Chatbot")
    parser.add_argument("--pdfs", nargs="*", default=[], help="PDF file paths")
    parser.add_argument("--videos", nargs="*", default=[], help="MP4 video file paths")
    parser.add_argument("--model", default=None, help="Ollama model name")
    args = parser.parse_args()

    if args.model:
        OLLAMA_MODEL = args.model

    collection = create_vector_store()
    ingest(collection, pdf_paths=args.pdfs, video_paths=args.videos)

    print("Knowledge base loaded. Ask questions (type 'quit' to exit):")
    while True:
        question = input("\nQ: ")
        if question.lower() in ("quit", "exit", "q"):
            break
        answer = ask(collection, question)
        print(f"A: {answer}")
