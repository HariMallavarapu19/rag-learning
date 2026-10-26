from main_chroma import retrieve_chunks,generate_answer
TOP_K=3

def build_context(results):
    return "\n\n".join(result['text'] for result in results)


def evaluate_faithfulness():
    print("RAG faithfulness evaluation")

    question=input("Enter your question").strip()

    if not question:
        print("Question cannot be empty")
        return

    results=retrieve_chunks(question,top_k=TOP_K)

    context=build_context(results)

    if not context:
        print("No relevant context was retrieved")

    else:
        for index,result in enumerate(results,start=1):
            print(f"\nCHunk{index}")
            print(f"similarity:{result['similarity']:.3f}")
            print(result['text'])

    print("\n"+"=*60")
    print("Generated answer")

    if not context:
        answer=(
            "I couldnt find relevant information"
            "in the available company documents"
        )
    else:
        answer=generate_answer(question,context)
    print(answer)

    print("manual faithfulness evaluation")
    print( "\nBreak the generated answer into individual claims."
           "\nFor each claim, decide whether the retrieved context " "supports it." )

    supported_claims=0
    unsupported_claims=0
    skipped_claims=0

    while True:
        claim=input(
            "\n Enter one claim from the answwer"
            "(or type done to finish)"
        ).strip()

        if claim.lower()=="done":
            break
        if not claim:
            print("Please enter a claim")
            continue

        print("\n Does the retrieved context support this claim")
        print("1=Supported")
        print("2=Unsupported")
        print("s=skip")

        label=input("Your judgment").strip().lower()

        if label=="1":
            supported_claims+=1
            print("Recorded:Supported")

        elif label=="0":
            unsupported_claims+=1
            print("Recorded:Unsupported")

        elif label=="s":
            skipped_claims+=1
            print("Claim skipped")    

        else:
            print("Invalid choice.please try aagain")


    evaluated_claims=supported_claims+unsupported_claims

    print("faithfulness results")

    print(f"Supported Claims:{supported_claims}")

    print(f"Unsupported claims:{unsupported_claims}")

    print(f"Skipped claims:{skipped_claims}")

    if evaluated_claims>0:
        faithfulness=supported_claims/evaluated_claims

        print(f"Evaluated claims:{evaluated_claims}")

        print(f"Faithfulness score:{faithfulness:.3f}")

        print(f"Faithfulness:{faithfulness*100:.1f}%")

    else:
        print("No claims were evaluated score cannot be calculated")


if __name__=="__main__":

    evaluate_faithfulness()

    
        

  
