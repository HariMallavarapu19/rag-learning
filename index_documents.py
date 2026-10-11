from pathlib import Path
from chunking import chunk_text
from chroma_store import ChromaVectorStore

DOCUMENTS_DIR=Path("documents")

def load_documents()->list[dict]:
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents folder not found:{DOCUMENTS_DIR}"
        )
    documents=[]
    for file_path in DOCUMENTS_DIR.glob("*.txt"):

        text = file_path.read_text(
            encoding="utf-8"
        )

        documents.append(
            {
                "filename": file_path.name,
                "text": text,
            }
        )

    return documents

def prepare_chunks(
    documents: list[dict],
) -> list[dict]:
    """
    Convert documents into chunks with metadata.
    """

    all_chunks = []

    for document in documents:

        chunks = chunk_text(
            document["text"]
        )

        # Fallback for documents without recognized headings.
        if not chunks:

            paragraphs = [
                paragraph.strip()
                for paragraph in document["text"].split("\n\n")
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

            record_id = (
                f"{document['filename']}_{chunk['chunk_id']}"
            )

            all_chunks.append(
                {
                    "id": record_id,
                    "text": chunk["text"],
                    "metadata": {
                        "document": document["filename"],
                        "section": chunk["section"],
                        "chunk_id": chunk["chunk_id"],
                    }
                }
            )
    return all_chunks

def main():
    documents=load_documents()
    if not documents:
        print("no documents found")
        return 
    print("Documents loaded")

    for document in documents:
        print(f"-{document['filename']}")
    chunks=prepare_chunks(documents)
    print(f"\n Total chunks to index:{len(chunks)}")

    store=ChromaVectorStore()
    print(f"Existing Chroma records:{store.count()}")
    store.add_chunks(chunks)
    print("\n Indexing complete")
    print(f"Total records inn chroma:{store.count()}")

if __name__=="__main__":
    main()

