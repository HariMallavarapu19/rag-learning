from chroma_store import ChromaVectorStore
from evaluation_dataset import EVALUATION_DATASET
 
TOP_K=3

def precision_at_k(retrieved_ids:list[str],
                  relevant_ids:list[str],
                  k:int)-> float:

    if k<=0:
        raise ValueError("K must be greater than zero")

    retrieved=retrieved_ids[:k]

    if not retrieved:
        return 0.0

    relevant_set=set(relevant_ids)

    relevant_count=sum(1 for chunk_id in retrieved if chunk_id in relevant_set)

    return relevant_count/len(retrieved)

def recall_at_k(retrieved_ids:list[str],
                relevant_ids:list[str],
                k:int):
    relevant_set=set(relevant_ids)
    if not relevant_set:
        return 0.0
    retrieved_set=set(retrieved_ids[:k])
    relevant_count=len(retrieved_set&relevant_set)

    return relevant_count/len(relevant_set)

def reciprocal_rank(retrieved_ids:list[str],
                    relevant_ids:list[str])->float:

    relevant_set=set(relevant_ids)
    for rank,chunk_id in enumerate(retrieved_ids,start=1):
        if chunk_id in relevant_set:
            return 1.0/rank

    return 0.0

def hit_rate_at_k(
        retrieved_ids:list[str],
        relevant_ids:list[str],
        k:int
    ):
    if k <=0:
        raise ValueError("K must be greater than zero")
    relevant_set = set(relevant_ids)

    return float(any(
        chunk_id in relevant_set
        for chunk_id in retrieved_ids[:k]
    ))


def evaluate_question(store:ChromaVectorStore,
                      question:str,
                      relevant_ids:list[str],
                      k:int) -> dict:
    results=store.search(question=question,
                         top_k=k)

    retrieved_ids=[result['id'] for result in results] 

    return {
        "question":question,
        "retrieved_ids":retrieved_ids,
        "results":results,
        "precision":precision_at_k(
            retrieved_ids,
            relevant_ids,
            k
        ),
        "recall":recall_at_k(retrieved_ids,
                             relevant_ids,
                             k
        ),
        "reciprocal_rank":reciprocal_rank(retrieved_ids,
                                          relevant_ids
        ),
        "hit_rate":hit_rate_at_k(retrieved_ids,
                                 relevant_ids,
                                 k
        )

    }

def evaluate_retrieval()->None:
    store=ChromaVectorStore()
    if store.count()==0:
        print("ChromDB is empty")
        print("Run python index_documens.py first")
        return

    stored_records=store.collection.get(include=['metadatas'])

    stored_ids=set(stored_records['ids'])

    missing_ids=set()

    for item in EVALUATION_DATASET:
        for chunk_id in item['relevant_chunk_ids']:
            if chunk_id not in stored_ids:
                missing_ids.add(chunk_id)

    if missing_ids:
        print("\n Error:Some expected chunk IDs do not exist")
        print("Verify the IDs before evaluatin retrieval")

        for chunk_id in sorted(missing_ids):
            print(f"Missing ID:{chunk_id}")
            print("\n Evaluation stopped to prevent misleading scores")

            return

    precision_scores=[]
    recall_scores=[]
    reciprocal_rank_scores=[]
    hit_scores=[]

    print("="*70)
    print("RAG Retrieval Evaluation")
    print("="*70)

    print(f"ChromaDB records:{store.count()}")
    print(f"Evaluation questions:{len(EVALUATION_DATASET)}")
    print(f"Top K:{TOP_K}")

    for index,item in enumerate(EVALUATION_DATASET,start=1):
        question=item['question']
        relevant_ids=item['relevant_chunk_ids']
        evaluation=evaluate_question(
            store=store,
            question=question,
            relevant_ids=relevant_ids,
            k=TOP_K
        )

        precision=evaluation['precision']
        recall=evaluation['recall']
        rr=evaluation['reciprocal_rank']
        hit=evaluation['hit_rate']

        precision_scores.append(precision)
        recall_scores.append(recall)
        reciprocal_rank_scores.append(rr)
        hit_scores.append(hit)

        print("\n"+"-"*70)
        print(f"Test {index}:{question}")

        print("\n Retrieve chunks:")

        for rank,result in enumerate(evaluation['results'],start=1):
            chunk_id=result['id']
            if chunk_id in relevant_ids:
                status=" expected relevant"
            else:
                status='Not in expected set'

            print(f"Rank{rank}:{chunk_id} | "
                  f"Similarity:{result['similarity']:.4f}|"
                  f"{status}")

        print(f"\n Precision@{TOP_K}:{precision:.3f}")
        print(f"Recall@{TOP_K}:{recall:.3f}")
        print(f"Reciprocal Rank:{rr:.3f}")
        print(f"Hit Rate@{TOP_K}:{hit:.3f}")



    total_questions=len(EVALUATION_DATASET)
    if total_questions==0:
        print("\n Evaluation dataset is empty")
        return

    print("Overall evalution results")

    print(f"Mean Precision@{TOP_K}:"
            f"{sum(precision_scores)/total_questions:.3f}"
    )

    print(f"Mean Recall@{TOP_K}"
            f"{sum(recall_scores)/total_questions:.3f}"
    )
    print(
        "Mean Reciprocal Rank (MRR): "
        f"{sum(reciprocal_rank_scores) / total_questions:.3f}"
    )

    print(
        f"Hit Rate@{TOP_K}: "
        f"{sum(hit_scores) / total_questions:.3f}"
    )


if __name__ == "__main__":
    evaluate_retrieval()
        

        