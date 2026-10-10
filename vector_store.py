import json
from pathlib import Path
from similarity import cosine_similarity

class VectorStore:
    def __init__(self,storage_path="data/vector_store.json"):
        self.storage_path=Path(storage_path)
        self.records={}

    def add(self,record_id:str,
            text:str,
            embedding:list[float],
            metadata:dict):
        if not record_id:
            raise ValueError("Record id cannot be empty")
        if not text.strip():
            raise ValueError("Text cannot be empty")
        if not embedding:
            raise ValueError("Embedding cannot be empty")
        if record_id in self.records:
            raise ValueError(
                f"Record Id already exixts:{record_id}"
            )
        self.records[record_id]={
            "id":record_id,
            "text":text,
            "embedding":embedding,
            "metadata":metadata
        }

    def save(self):
        self.storage_path.parent.mkdir(parents=True,exist_ok=True)
        with self.storage_path.open("w",encoding='utf-8') as file:
            json.dump(
                list(self.records.values()),
                file,
                ensure_ascii=False,
                indent=2
            )
        print(f"Saved {len(self.records)}records"
              f"to {self.storage_path}")

    def load(self):
        if not self.storage_path.exists():
            print("not exisiting vector store found"
                  "Starting with an empty store")
            return 
        with self.storage_path.open("r",encoding="utf-8") as file:
            stored_records=json.load(file)
        loaded_records={}
        for record in stored_records:
            record_id=record['id']
            embedding=record['embedding']

            if not embedding:
                raise ValueError(f"Empty embedding for record {record_id}")

            loaded_records[record_id]=record

        self.records=loaded_records
        print(f"Loaded {len(self.records)} records"
              f"from {self.storage_path}")

    def search(self,query_embedding:list[float],top_k:int=3)->list[dict]:
        if top_k<1:
            raise ValueError("top k must be at least 1")
        results=[]
        for record in self.records.values():
            score=cosine_similarity(query_embedding,record["embedding"])
            results.append(
                {
                    "id":record['id'],
                    "text":record['text'],
                    'metadata':record['metadata'],
                    'similarity':score
                }
            )

        results.sort(key=lambda item:item['similarity'],reverse=True)

        return results[:top_k]

    def count(self)->int:
        return len(self.records)
    def clear(self):
        self.records.clear()


    
        

