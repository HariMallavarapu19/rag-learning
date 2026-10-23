from pathlib import Path
import chromadb
from embedding import create_embedding

CHROMA_PATH=Path("data/chroma_db")

COLLECTION_NAME="company_documents"

class ChromaVectorStore:
    def __init__(self):
        CHROMA_PATH.mkdir(
            parents=True,
            exist_ok=True
        )

        self.client=chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection=self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description":"Company policy document chunks",
                      "hnsw:space":"cosine"}
        )
        
    def add_chunks(self,chunks:list[dict]):
        for chunk in chunks:
            embedding=create_embedding(chunk['text'])
            self.collection.upsert(
                ids=[chunk['id']],
                documents=[chunk['text']],
                embeddings=[embedding],
                metadatas=[chunk['metadata']]
            )

        print(f"Stored chunk:{chunk['id']}")

    def search(self,question:str,top_k:int=3)->list[dict]:
        if top_k<1:
            raise ValueError("Top_K must be at least 1")
        if self.collection.count()==0:
            return []
        question_embedding=create_embedding(question)

        results=self.collection.query(
            query_embeddings=[question_embedding],
            n_results=min(top_k,self.collection.count()),
            include=["documents","metadatas","distances"]
        )

        retrieved_chunks=[]
        for index,document in enumerate(results["documents"][0]):
            metadata=results['metadatas'][0][index]
            distance=results['distances'][0][index]
                
            similarity=1.0-distance
            retrieved_chunks.append({
                "id":results["ids"][0][index],
                "text":document,
                "metadata":metadata,
                'similarity':similarity
            })
        return retrieved_chunks
    def count(self)->int:
        return self.collection.count()
    
    

