import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key=os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("gemini key is not set")

client=genai.Client(api_key=api_key)

EMBEDDING_MODEL='gemini-embedding-001'

def create_embedding(text:str) ->list[float]:
    response=client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )
    return response.embeddings[0].values



