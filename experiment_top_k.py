from chroma_store import ChromaVectorStore

def main():
    store=ChromaVectorStore()
    if store.count()==0:
        print("Chromdb is empty")
        print("Run python index documents.py first")
        return

    question="Does the company provide the health insurance?"
    top_k_values=[1,2,3,5]

    print("="*70)
    print("Top-k retireval experiment")
    print("="*70)

    print(f"\nQuestion:{question}")
    print(f"Total ChromaDB records:{store.count()}")

    for top_k in top_k_values:
        results=store.search(question,top_k)
        print("\n"+"-"*70)
        print(f"TOP_K={top_k}")
        print("-"*70)

        for rank,result in enumerate(results,start=1):
            metadata=result['metadata']
            print(f"\n Rank:{rank}")
            print(f"Document:{metadata['document']}")
            print(f"Section:{metadata['section']}")
            print(f"Similarity:{result['similarity']:.4f}")
            print(f"Text:{result['text']}")
        print(f"\n Results returned:{len(results)}")

if __name__=="__main__":
    main()


