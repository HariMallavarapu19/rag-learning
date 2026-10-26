import json
import re

from google import genai
from google.genai import types

from main_chroma import (
    retrieve_chunks,generate_answer,MODEL_NAME,client
)

TOP_K=3

def build_context(results):
    return "\n\n".join(
        result['text'] for result in results
    )

def generate_rag_answer(question,context):
    if not context.strip():
        return (
            "I couldnt find relevant information"
            "in the available company documents"
        )

    return generate_answer(question,context)

def extract_json(text):
    text=text.strip()
    text=re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text=re.sub(r"\s*'''$","",text)

    return json.loads(text)

def evaluate_with_gemini(question,context,answer):
    prompt = f""" You are a careful evaluator of a Retrieval-Augmented Generation (RAG) system. 
    Your task is to evaluate whether the factual claims in an answer are supported by the provided context.
    Do not use outside knowledge. Do not assume a claim is supported merely because it sounds plausible.
    Distinguish between supported claims, partially supported claims, and unsupported claims.
    If the answer says that the documents do not provide enough information,
    check whether the supplied context actually supports that conclusion.
    If the context is empty, an appropriate statement that no relevant information was retrieved should be treated as an abstention,
    not as an unsupported company-policy claim.
    
    QUESTION: {question} 
    RETRIEVED CONTEXT: {context if context.strip() else "[No context retrieved]"} 
    GENERATED ANSWER: {answer}

    Return ONLY valid JSON using this exact structure:
    {{
          "claims": [
            {{ 
                "claim": "A factual claim from the answer",
                "verdict": "supported", 
                "reason": "Explain what context supports the claim."
             }} 
          ],
          "abstention": {{ 
            "is_abstention": false, 
            "appropriate": null, 
            "reason": "Explain whether the answer appropriately abstains." 
        }} 
    }}

    Rules:
      1. The verdict must be one of: "supported", "partially_supported", "unsupported".
      2. Extract the factual claims from the answer. Do not invent claims that are not in the answer. 
      3. If the answer contains no factual claims, return an empty claims list. 
      4. For an answer that appropriately abstains because information is missing, set is_abstention to true and appropriate to true. 
      5. If the answer abstains but the context clearly contains the answer, set is_abstention to true and appropriate to false. 
      6. If the answer does not abstain, set is_abstention to false and appropriate to null. 
      7. If the context does not support a claim, do not label it supported. 
      8. Return JSON only. Do not include Markdown or additional commentary. 
      
    """

    response=client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0
        )
    )

    if not response.text:
        raise ValueError(
            "The evaluator model returned an empty response"
        )

    evaluation=extract_json(response.text)

    if not isinstance(evaluation,dict):
        raise ValueError("Evaluator response must be json object")
    claims=evaluation.get("claims")
    abstention=evaluation.get("abstention")

    if not isinstance(claims,list):
        raise ValueError("claims must be a json list")

    if not isinstance(abstention,dict):
        raise ValueError('abstention must be a json object')

    allowed_verdicts=(
        "supported",
        "partially_supported"
        "unsupported"
    )

    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("Each claim must be a JSON object.") 
        if claim.get("verdict") not in allowed_verdicts: 
            raise ValueError("Evaluator returned an invalid verdict.") 
        if not isinstance(claim.get("claim"), str): 
            raise ValueError("Each claim must contain text.") 
        if not isinstance(claim.get("reason"), str): 
            raise ValueError("Each claim must contain a reason.") 
        if not isinstance(abstention.get("is_abstention"), bool): 
            raise ValueError("'is_abstention' must be true or false.") 
        if abstention.get("appropriate") not in (True, False, None): 
            raise ValueError("'appropriate' must be true, false, or null.") 
        if not isinstance(abstention.get("reason"), str): 
            raise ValueError("Abstention evaluation must include a reason.") 
        
        return evaluation

def calculate_faithfulness(evaluation):
    """ Calculate a claim-support score. 
    Supported claim = 1.0 
    Partially supported claim = 0.5 
    Unsupported claim = 0.0 
    This is a simple educational scoring convention, 
    not a universal standard for RAG evaluation. """ 
    
    claims = evaluation["claims"] 
    if not claims: 
        return None 
    scores = { "supported": 1.0, "partially_supported": 0.5, "unsupported": 0.0, } 
    total_score = sum( scores[claim["verdict"]] for claim in claims ) 
    
    return total_score / len(claims)

def print_report(question,context,answer,evaluation):
    print("Automated rag faithfulness evaluation")
    print("Question:")
    print(question)

    print(context if context.strip() else "No relevant context retrieved")

    print(answer)

    claims=evaluation["claims"]

    if not claims:
        print("no factual claims were identified")
    else:
        for index,claim in enumerate(claims,start=1):
            print(f"\nClaim {index}: {claim['claim']}") 
            print(f"Verdict: {claim['verdict']}") 
            print(f"Reason: {claim['reason']}")

    score = calculate_faithfulness(evaluation)

    print("\nFAITHFULNESS SCORE")

    if score is None:
        print("Not applicable: no factual claims were identified.")
    else:
        print(f"Score: {score:.3f}")
        print(f"Percentage: {score * 100:.1f}%")

    abstention = evaluation.get("abstention") or {}

    print("\nABSTENTION EVALUATION")

    is_abstention = abstention.get("is_abstention", False)

    print(f"Is abstention: {is_abstention}")

    if is_abstention:
        print(f"Appropriate: {abstention.get('appropriate')}")

    print(f"Reason: {abstention.get('reason', 'No reason provided.')}")

def run_evalution():
    print("Automated rag faithfulness evaluator")
    question=input("Enter your question:").strip()

    if not question:
        print("question cannot be empty")
        return

    try:
        results=retrieve_chunks(question,top_k=TOP_K)
        context=build_context(results)
        if not results:
            print("No relevant context retrieved")

        else:
            for index,result in enumerate(results,start=1):
                print(f"\nChunk {index}")
                print(f"Similarity:{result['similarity']:.3f}")
                print(result['text'])

        answer=generate_rag_answer(question,context)

        if not answer or not answer.strip():
            raise ValueError("The RAG system returned an empty answer ")

        evaluation=evaluate_with_gemini(question,context,answer)

        print_report(question,context,answer,evaluation)
    except Exception as error:
        print("\nEvaluation failed.") 
        print(f"Error type: {type(error).__name__}") 
        print(f"Error details: {error}")
if __name__=="__main__":
    run_evalution()


