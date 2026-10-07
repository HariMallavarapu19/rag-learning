from pathlib import Path
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

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

    for document in documents:
        print(document['filename'])

    question = "How many annual leave days do employees receive?"

    print("\nQuestion:")
    print(question)
    results=retrive(question,documents)
    print('/nRetrived documents')

    if not results:
        print("No relevant documents found")
        return 
    
    for result in results:
        print(f"File:{result['filename']}")
        print(f'Score:{result['score']}')

    best_result=results[0]
    context=best_result['text']
    print(context)

    answer=generate_answer(question,context)

    print("\nGeminin answer")
    print(answer)

if __name__=="__main__":
    main()