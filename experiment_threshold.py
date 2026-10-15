from chroma_store import ChromaVectorStore

def main():
    store=ChromaVectorStore()
    if store.count()==0:
        print("ChromaDB is empty.")
        print("Run python index_documents.py first.")
        return

    question = "Does the company provide health insurance?"

    top_k = 5

    thresholds=[0.50,0.60,0.65,0.70,0.75]
    results=store.search(question,top_k)
    print("=" * 70)
    print("SIMILARITY THRESHOLD EXPERIMENT")
    print("=" * 70)

    print(f"\nQuestion: {question}")
    print(f"Top K: {top_k}")

    print("\nAll retrieved results:")

    for rank,result in enumerate(results,start=1):
        print(f"Rank{rank} | "
              f"Similarity:{result['similarity']:4f} |"
              f"Section:{result['metadata']['section']}")

    for threshold in thresholds:
        filtered_results=[result for result in results if result['similarity']>=threshold]
        print("\n"+"-"*70)
        print(f"Threshold:{threshold:.2f}")
        print(f"Chunks retained:{len(filtered_results)}")

        for result in filtered_results:
            print(f"-{result['metadata']['section']}"
                  f"({result['similarity']:.4f})")

        if not filtered_results:
            print("No chunks passed the theshold")

    print('\n Experiment completed')

if __name__=='__main__':
    main()

        




