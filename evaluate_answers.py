
from evaluation_answers import ANSWER_EVALUATION_DATASET
from main_chroma import retrieve_chunks, generate_answer
from google.genai.errors import APIError


TOP_K = 3


def build_context(results: list[dict]) -> str:
    """Combine retrieved chunks into one context string."""
    return "\n\n".join(
        result["text"] for result in results
    )


def evaluate_answers():
    print("=" * 70)
    print("RAG ANSWER EVALUATION")
    print("=" * 70)

    total_questions = len(ANSWER_EVALUATION_DATASET)

    print(f"Evaluation questions: {total_questions}")
    print(f"Top K: {TOP_K}")

    evaluation_results = []

    successful_generations = 0
    generation_failures = 0

    for index, item in enumerate(
        ANSWER_EVALUATION_DATASET,
        start=1,
    ):
        question = item["question"]
        expected_answer = item["expected_answer"]
        answerable = item["answerable"]

        print("\n" + "-" * 70)
        print(f"Test {index}: {question}")

        # Step 1: Retrieve chunks
        results = retrieve_chunks(
            question=question,
            top_k=TOP_K,
        )

        # Step 2: Build context
        context = build_context(results)

        print("\nRetrieved chunks:")

        if not results:
            print("No chunks passed the similarity threshold.")

        for rank, result in enumerate(results, start=1):
            print(
                f"Rank {rank}: {result['id']} | "
                f"Similarity: {result['similarity']:.4f}"
            )

        # Step 3: Generate answer
        generated_answer = None
        generation_error = None

        try:
            generated_answer = generate_answer(
                question=question,
                context=context,
            )

            # Treat an empty response as a failed generation.
            if generated_answer is not None:
                generated_answer = generated_answer.strip()

                if not generated_answer:
                    generated_answer = None

        except APIError as error:
            generation_error = (
                f"{type(error).__name__}: {error}"
            )

        except Exception as error:
            generation_error = (
                f"{type(error).__name__}: {error}"
            )

        # Step 4: Display expected answer
        print("\nExpected answer:")

        if answerable:
            print(expected_answer)
        else:
            print(
                "UNANSWERABLE: The available documents "
                "do not establish an answer."
            )

        # Step 5: Display generated answer
        print("\nGenerated answer:")

        if generated_answer is None:
            print("Generation failed or returned an empty answer.")

            if generation_error:
                print(f"Error: {generation_error}")

            generation_status = "failed"
            generation_failures += 1

        else:
            print(generated_answer)

            generation_status = "success"
            successful_generations += 1

        # Step 6: Ask for a manual correctness label
        correct = None

        if generated_answer is not None:
            if answerable:
                print("\nEvaluate the generated answer:")
                print("1 = Correct")
                print("0 = Incorrect")
                print("s = Skip evaluation")

                while True:
                    label = input(
                        "Enter your label (1/0/s): "
                    ).strip().lower()

                    if label == "1":
                        correct = True
                        break

                    elif label == "0":
                        correct = False
                        break

                    elif label == "s":
                        correct = None
                        break

                    else:
                        print(
                            "Invalid input. Enter 1, 0, or s."
                        )

            else:
                print(
                    "\nCheck whether the answer appropriately "
                    "acknowledges that the information is unavailable."
                )
                print("1 = Appropriate abstention")
                print("0 = Unsupported answer")
                print("s = Skip evaluation")

                while True:
                    label = input(
                        "Enter your label (1/0/s): "
                    ).strip().lower()

                    if label == "1":
                        correct = True
                        break

                    elif label == "0":
                        correct = False
                        break

                    elif label == "s":
                        correct = None
                        break

                    else:
                        print(
                            "Invalid input. Enter 1, 0, or s."
                        )

        # Step 7: Save this result in memory
        result_record = {
            "question": question,
            "expected_answer": expected_answer,
            "answerable": answerable,
            "generated_answer": generated_answer,
            "retrieved_chunks": [
                result["id"] for result in results
            ],
            "generation_status": generation_status,
            "generation_error": generation_error,
            "correct": correct,
        }

        evaluation_results.append(result_record)

        print(
            "Recorded correctness label:",
            correct if correct is not None else "Not evaluated",
        )

    # Step 8: Calculate the final summary
    print("\n" + "=" * 70)
    print("RAG ANSWER EVALUATION SUMMARY")
    print("=" * 70)

    print(f"Total questions: {total_questions}")
    print(f"Successful generations: {successful_generations}")
    print(f"Generation failures: {generation_failures}")

    if total_questions:
        success_rate = (
            successful_generations / total_questions
        ) * 100

        print(f"Generation success rate: {success_rate:.1f}%")

    # Calculate accuracy for answerable questions only.
    evaluated_answerable = [
        result
        for result in evaluation_results
        if result["answerable"]
        and result["generation_status"] == "success"
        and result["correct"] is not None
    ]

    correct_answers = sum(
        result["correct"] is True
        for result in evaluated_answerable
    )

    incorrect_answers = sum(
        result["correct"] is False
        for result in evaluated_answerable
    )

    total_evaluated = len(evaluated_answerable)

    print("\nAnswerable-question evaluation:")
    print(f"Evaluated answers: {total_evaluated}")
    print(f"Correct answers: {correct_answers}")
    print(f"Incorrect answers: {incorrect_answers}")

    if total_evaluated:
        accuracy = (
            correct_answers / total_evaluated
        ) * 100

        print(f"Answer accuracy: {accuracy:.1f}%")
    else:
        print("Answer accuracy: N/A")

    # Report unanswerable-question results separately.
    evaluated_unanswerable = [
        result
        for result in evaluation_results
        if not result["answerable"]
        and result["generation_status"] == "success"
        and result["correct"] is not None
    ]

    correct_abstentions = sum(
        result["correct"] is True
        for result in evaluated_unanswerable
    )

    print("\nUnanswerable-question evaluation:")
    print(f"Evaluated answers: {len(evaluated_unanswerable)}")
    print(f"Appropriate abstentions: {correct_abstentions}")

    if evaluated_unanswerable:
        abstention_accuracy = (
            correct_abstentions
            / len(evaluated_unanswerable)
        ) * 100

        print(
            f"Appropriate abstention rate: "
            f"{abstention_accuracy:.1f}%"
        )
    else:
        print("Appropriate abstention rate: N/A")

    print("\nFaithfulness: Not evaluated yet.")
    print("Evaluation labels are manual, not automatically verified.")

    print("=" * 70)

    return evaluation_results


if __name__ == "__main__":
    evaluate_answers()

