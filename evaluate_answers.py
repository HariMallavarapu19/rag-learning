from evaluation_answers import ANSWER_EVALUATION_DATASET
from main_chroma import retrieve_chunks, generate_answer
from google.genai.errors import APIError


TOP_K = 3


def build_context(results: list[dict]) -> str:
    """
    Combine retrieved document chunks into one context string.
    """

    context_parts = []

    for result in results:
        context_parts.append(result["text"])

    return "\n\n".join(context_parts)


def evaluate_answers():
    print("=" * 70)
    print("RAG ANSWER EVALUATION")
    print("=" * 70)

    total_questions = len(ANSWER_EVALUATION_DATASET)

    print(f"Evaluation questions: {total_questions}")
    print(f"Top K: {TOP_K}")

    # Track evaluation results.
    evaluation_results = []

    successful_generations = 0
    generation_failures = 0

    # Process each evaluation question.
    for index, item in enumerate(
        ANSWER_EVALUATION_DATASET,
        start=1,
    ):
        question = item["question"]
        expected_answer = item["expected_answer"]

        print("\n" + "-" * 70)
        print(f"Test {index}: {question}")

        # --------------------------------------------------
        # Step 1: Retrieve relevant chunks
        # --------------------------------------------------

        results = retrieve_chunks(
            question=question,
            top_k=TOP_K,
        )

        # --------------------------------------------------
        # Step 2: Build context
        # --------------------------------------------------

        context = build_context(results)

        print("\nRetrieved chunks:")

        if not results:
            print("No chunks passed the similarity threshold.")

        for rank, result in enumerate(results, start=1):
            print(
                f"Rank {rank}: "
                f"{result['id']} | "
                f"Similarity: {result['similarity']:.4f}"
            )

        # --------------------------------------------------
        # Step 3: Generate answer
        # --------------------------------------------------

        generated_answer = None
        generation_error = None

        try:
            generated_answer = generate_answer(
                question=question,
                context=context,
            )

        except APIError as error:
            generation_error = (
                f"{type(error).__name__}: {error}"
            )

            print("\nGeneration failed!")
            print(generation_error)

        except Exception as error:
            generation_error = (
                f"{type(error).__name__}: {error}"
            )

            print("\nUnexpected error during generation!")
            print(generation_error)

        # --------------------------------------------------
        # Step 4: Display expected answer
        # --------------------------------------------------

        print("\nExpected answer:")

        if expected_answer is None:
            print(
                "UNANSWERABLE: The documents do not "
                "establish an answer."
            )
        else:
            print(expected_answer)

        # --------------------------------------------------
        # Step 5: Display generated answer
        # --------------------------------------------------

        print("\nGenerated answer:")

        if generated_answer is None:
            print(
                "EVALUATION SKIPPED: "
                "No answer was generated."
            )

        else:
            generated_answer = generated_answer.strip()

            if not generated_answer:
                generated_answer = None

                print(
                    "EVALUATION SKIPPED: "
                    "The model returned an empty answer."
                )
            else:
                print(generated_answer)

        # --------------------------------------------------
        # Step 6: Record the evaluation result
        # --------------------------------------------------

        if generated_answer is None:
            generation_failures += 1

            status = "generation_failed"

        else:
            successful_generations += 1

            status = "generated"

        result_record = {
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "retrieved_chunks": [
                result["id"] for result in results
            ],
            "status": status,
            "generation_error": generation_error,
            "correct": None,
        }

        evaluation_results.append(result_record)

        # --------------------------------------------------
        # Step 7: Manual evaluation checklist
        # --------------------------------------------------

        print("\nManual evaluation checklist:")

        if generated_answer is None:
            print("Not applicable: generation failed.")

        else:
            print("[ ] Is the answer correct?")
            print(
                "[ ] Is every factual claim supported "
                "by the retrieved context?"
            )
            print(
                "[ ] Does the answer invent "
                "unsupported information?"
            )

    # ======================================================
    # Final evaluation summary
    # IMPORTANT: This section is outside the for loop.
    # ======================================================

    print("\n" + "=" * 70)
    print("RAG ANSWER EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Total questions: {total_questions}")
    print(f"Successful generations: {successful_generations}")
    print(f"Generation failures: {generation_failures}")

    if total_questions > 0:
        success_rate = (
            successful_generations / total_questions
        ) * 100

        print(
            f"Generation success rate: "
            f"{success_rate:.1f}%"
        )
    else:
        print("Generation success rate: N/A")

    print("\nCorrectness score: Not calculated yet.")
    print("Faithfulness score: Not calculated yet.")

    print(
        "Reason: Generated answers still require "
        "correctness and faithfulness labels."
    )

    print("=" * 70)

    return evaluation_results


if __name__ == "__main__":
    evaluate_answers()