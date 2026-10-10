from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from embedding import create_embedding
from similarity import cosine_similarity
from chunking import chunk_text
from vector_store import VectorStore


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not set")

client = genai.Client(api_key=api_key)

MODEL = "gemini-3.6-flash"

DOCUMENTS_DIR = Path("documents")

VECTOR_STORE_PATH="data/vector_store.json"

TOP_K=3


def load_documents(
    directory=DOCUMENTS_DIR,
    pattern="*.txt",
):
    """
    Load all text documents from the documents folder.
    """

    if not directory.exists():
        raise FileNotFoundError(
            f"Folder not found: {directory}"
        )

    documents = []

    for file_path in directory.glob(pattern):

        try:
            text = file_path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError:
            print(
                f"Skipping {file_path.name}: "
                "not valid UTF-8"
            )
            continue

        documents.append(
            {
                "filename": file_path.name,
                "text": text,
            }
        )

    return documents


def prepare_chunks(documents):
    """
    Chunk all documents and attach metadata.

    Documents with section headings use structure-aware
    chunking.

    Documents without section headings fall back to
    paragraph-based chunks.
    """

    all_chunks = []

    for document in documents:

        text = document["text"]

        # Try structure-aware chunking.
        chunks = chunk_text(text)

        # Fallback for documents without recognized sections.
        if not chunks:

            paragraphs = [
                paragraph.strip()
                for paragraph in text.split("\n\n")
                if paragraph.strip()
            ]

            chunks = [
                {
                    "chunk_id": index,
                    "section": "General",
                    "text": paragraph,
                }
                for index, paragraph in enumerate(
                    paragraphs,
                    start=1,
                )
            ]

        for chunk in chunks:

            all_chunks.append(
                {
                    "filename": document["filename"],
                    "chunk_id": chunk["chunk_id"],
                    "section": chunk["section"],
                    "text": chunk["text"],
                }
            )

    return all_chunks

def build_vector_store(documents):
    chunks=prepare_chunks(documents)
    store=VectorStore(storage_path=VECTOR_STORE_PATH)

    print(f"\n Indexing {len(chunks)} chunks>>>>")

    for chunk in chunks:
        record_id=(f"{chunk['filename']}_{chunk['chunk_id']}")
        embedding=create_embedding(chunk['text'])

        store.add(
            record_id=record_id,
            text=chunk['text'],
            embedding=embedding,
            metadata={
                "document":chunk['filename'],
                'section':chunk['section'],
                'chunk_id':chunk['chunk_id']
            }
        )
    print(f"Indexed:{record_id}")
    store.save()
    return store

def get_vector_store(documents):
    store=VectorStore(storage_path=VECTOR_STORE_PATH)
    if Path(VECTOR_STORE_PATH).exists():
        print("\nLoading existing vector store...")
        store.load()

    else:
        print("\nNo existing vector store found.")
        print("Creating a new vector store...")
        store=build_vector_store(documents)
    return store


def retrive_chunks(
    question,
    store,
    top_k=TOP_K,
):
    """
    Retrieve the Top-K chunks using embedding similarity.
    """

    question_embedding = create_embedding(
        question
    )

    results=store.search(query_embedding=question_embedding,
                         top_k=top_k)

    return results
    

def generate_answer(
    question,
    retrived_chunks,
):
    """
    Generate an answer using the retrieved context.
    """
    if not retrived_chunks:
        return (
            "I cannot find that information "
            "in the provided documents."
        )

    context_parts = []

    for chunk in retrived_chunks:
        metadata=chunk['metadata']
        context_parts.append(
            f"""
SOURCE: {metadata['document']}
SECTION: {metadata['section']}
CHUNK: {metadata['chunk_id']}

CONTENT:
{chunk['text']}
"""
   )

    context = "\n".join(context_parts)

    prompt = f"""
You are a helpful assistant answering questions
using company documents.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context,
say exactly:

"I cannot find that information in the provided documents."

Do not invent information.
Do not assume that the context supports facts it does not contain.

CONTEXT:
{context}

QUESTION:
{question}

Answer clearly and concisely.
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0
        ),
    )

    return (
        response.text
        or "The model returned an empty response."
    )


def main():

    # Step 1: Load documents.
    documents = load_documents()

    if not documents:
        print("No documents found.")
        return

    print("Documents loaded:")

    for document in documents:
        print(document["filename"])

    store = get_vector_store(
        documents
    )

    print(
        f"\nVector store contains {store.count()} records."
    )

    # Step 2: Ask a question.
    question = input(
        "\nEnter your question: "
    ).strip()

    if not question:
        print("Question cannot be empty.")
        return

    print("\nQuestion:")
    print(question)

    # Step 3: Retrieve relevant chunks.
    retrived_chunks = retrive_chunks(
        question,
        store,
        top_k=TOP_K,
    )

    print("\nTop-K Retrieved Chunks")
    print("-" * 60)

    for index, chunk in enumerate(
        retrived_chunks,
        start=1,
    ):
        metadata = chunk["metadata"]

        print(f"\nRank: {index}")
        print(f"Document: {metadata['document']}")
        print(f"Section: {metadata['section']}")
        print(f"Chunk ID: {metadata['chunk_id']}")


        print(
            f"Similarity: {chunk['similarity']:.4f}"
        )

        print(f"Text: {chunk['text']}")

    # Step 4: Generate an answer.
    answer = generate_answer(
        question,
        retrived_chunks,
    )

    print("\nGemini Answer")
    print("-" * 60)
    print(answer)


if __name__ == "__main__":
    main()