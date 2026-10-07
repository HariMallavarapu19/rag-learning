from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from embedding import create_embedding
from similarity import cosine_similarity


load_dotenv()
api_key=os.getenv("GEMINI_API_KEY")

if not api_key:
     raise ValueError("Gemini api key not set")

client=genai.Client(api_key=api_key)

MODEL='gemini-3.6-flash'

D0CUMENTS_DIR=Path('documents',)

def load_documents(directory=D0CUMENTS_DIR,pattern="*.txt"):
    documentss=[]

    for file_path in directory.glob(pattern):
        if not directory.exists():
            raise FileNotFoundError(f"Folder not found:{directory}")
        try:
            text=file_path.read_text(encoding="utf-8",)
        except  UnicodeDecodeError:
            print(f"Skipping{file_path.name}: not valid UTF-8")
            continue
        documentss.append(
            {
                "filename":file_path.name,
                "text":text
            }
        )
    return documentss

def retrive(question,documentss):
    question_words=set(question.lower().split())
    results=[]

    for document in documentss:
        document_words=set(document['text'].lower().split())

        matching_words=question_words.intersection(document_words)

        score=len(matching_words)

        if score>0:
            results.append(
                {
                    "filename":document['filename'],
                    'score':score,
                    "text":document['text']
                }
            )
    results.sort(
        key=lambda item:item['score'],reverse=True
    )

    return results


def generate_answer(question,context):
    prompt=f"""
    You are a helpful assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context,
say:

"I cannot find that information in the provided documents."

Do not invent information.

CONTEXT:
{context}

QUESTION:
{question}
"""
    response=client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.0
        )

    )

    return response.text

def main():
    documents=load_documents()
    print("Documents loaded")

    

    questions = ["How many vacation days do employees receive?",
        "Does the company provide health insurance?",
        "How many days can I work from home?"]

    for question in questions:

        print("\n" + "=" * 60)
        print(f"QUESTION: {question}")
        print("=" * 60)
        question_embedding = create_embedding(question)

        results=[]

        for document in documents:
            print(f"\n Creating embedding for "
                f"{document['filename']}")

            document_embedding=create_embedding(document['text'])

            similarity=cosine_similarity(question_embedding,document_embedding)

            results.append({
                "filename":document['filename'],
                'text':document['text'],
                'similarity':similarity
            })

        results.sort(
            key=lambda item: item['similarity'],reverse=True,
        )

        print("Ranking--------")
        for index,result in enumerate(results,start=1):
            print(f"{index}."
                f"{result['filename']}"
                f"->{result['similarity']:.4f}")
        if not results:

            print("No relevant documents found")
            return 
        
        

        best_result=results[0]

        print(f"file:{best_result['filename']}")
        print(f"Similarity:"
            f"{best_result['similarity']:.4f}")

        # answer=generate_answer(question,context)

        # print("\nGeminin answer")
        # print(answer)

if __name__=="__main__":
    main()
