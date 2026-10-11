from chroma_store import ChromaVectorStore

def main():
    store=ChromaVectorStore()
    print(f"Chroma records:{store.count()}")

    if store.count()==0:
        print(
            "No records found. run index document.py first"
        )
        return
    question=input("\n Enter your question:").strip()

    results=store.search(question=question,top_k=3)
    print("\n Retrieved Chunks")
    print("="*60)
    for rank,chunk in enumerate(results,start=1):
        metadata=chunk['metadata']
        print(f"\nRank: {rank}")
        print(f"Document: {metadata['document']}")
        print(f"Section: {metadata['section']}")
        print(f"Chunk ID: {metadata['chunk_id']}")

        print(
            f"Similarity: {chunk['similarity']:.4f}"
        )

        print(f"Text: {chunk['text']}")


if __name__ == "__main__":
    main()

