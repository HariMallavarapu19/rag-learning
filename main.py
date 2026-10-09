from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from embedding import create_embedding
from similarity import cosine_similarity
from chunking import chunk_text


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not set")

client = genai.Client(api_key=api_key)

MODEL = "gemini-3.6-flash"

DOCUMENTS_DIR = Path("documents")


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


def retrive_chunks(
    question,
    documents,
    top_k=3,
):
    """
    Retrieve the Top-K chunks using embedding similarity.
    """

    question_embedding = create_embedding(
        question
    )

    # Create structured chunks from every document.
    chunks = prepare_chunks(documents)

    results = []

    for chunk in chunks:

        chunk_embedding = create_embedding(
            chunk["text"]
        )

        similarity = cosine_similarity(
            question_embedding,
            chunk_embedding,
        )

        results.append(
            {
                "filename": chunk["filename"],
                "chunk_id": chunk["chunk_id"],
                "section": chunk["section"],
                "similarity": similarity,
                "text": chunk["text"],
            }
        )

    # Sort from highest to lowest similarity.
    results.sort(
        key=lambda item: item["similarity"],
        reverse=True,
    )

    return results[:top_k]


def generate_answer(
    question,
    retrived_chunks,
):
    """
    Generate an answer using the retrieved context.
    """

    context_parts = []

    for chunk in retrived_chunks:

        context_parts.append(
            f"""
SOURCE: {chunk['filename']}
SECTION: {chunk['section']}
CHUNK: {chunk['chunk_id']}

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
        documents,
        top_k=3,
    )

    print("\nTop-K Retrieved Chunks")
    print("-" * 60)

    for index, chunk in enumerate(
        retrived_chunks,
        start=1,
    ):

        print(f"\nRank: {index}")
        print(f"Document: {chunk['filename']}")
        print(f"Section: {chunk['section']}")
        print(f"Chunk ID: {chunk['chunk_id']}")

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