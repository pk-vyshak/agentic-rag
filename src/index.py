import fitz,uuid,time,os
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance,Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from loguru import logger as lg
from basevector import BaseVectorStore 

class PDFIndexer(BaseVectorStore):
    def __init__(self):
        super().__init__()

    def extract_paragraphs(self, pdf_path: str) -> List[Dict]:
        try:
            st = time.time()
            doc = fitz.open(pdf_path)
            data = []

            for page_num, page in enumerate(doc, start=1):
                blocks = page.get_text("blocks")
                blocks.sort(key=lambda b: b[1])
                for line_num, block in enumerate(blocks, start=1):
                    text = block[4].strip()
                    if text:
                        data.append({
                            "text": text,
                            "page": page_num,
                            "line": line_num
                        })

            en = time.time()
            lg.info(f"Extracted {len(data)} paragraphs from {pdf_path} in {en-st:.2f} seconds.")
            return data
        except Exception as e:
            lg.exception(f"Error processing PDF: {e}")
            return []
    
    def index_paragraphs(self, data: List[Dict], pdf_name: str, batch_size: int = 40) -> bool:
        try:
            total = len(data)
            for start in range(0, total, batch_size):
                batch = data[start:start + batch_size]
                texts = [d['text'] for d in batch]
                vectors = self.model.encode(texts).tolist()

                points = []
                for record, vector in zip(batch, vectors):
                    points.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": record["text"],
                            "page_number": record["page"],
                            "line_number": record["line"],
                            "docid": pdf_name
                        }
                    ))

                self.client.upsert(collection_name=self.collection_name, points=points)

            lg.info(f"Indexed {total} paragraphs under docid='{pdf_name}' in batches of {batch_size}.")
            return True
        except Exception as e:
            lg.exception(f"Error indexing paragraphs in batches: {e}")
            return False

    def main(self, pdf_path: str):
        try:
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            data=self.extract_paragraphs(pdf_path)
            index_flag=self.index_paragraphs(data, pdf_name)
            return index_flag
        except Exception as e:
            lg.exception(f"Error in main method: {e}")
            return True

    def search_paragraphs(self, query_text: str, pdf_name: str, top_k: int = 5) -> List[Dict]:
        try:
            vector = self.model.encode([query_text])[0]

            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=top_k,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="docid",
                            match=MatchValue(value=pdf_name)
                        )
                    ]
                )
            )

            return [
                {
                    "text": r.payload.get("text"),
                    "page_number": r.payload.get("page_number"),
                    "line_number": r.payload.get("line_number"),
                    "score": r.score
                }
                for r in results
            ]

        except Exception as e:
            lg.exception(f"Error during search for '{query_text}' in '{pdf_name}': {e}")
            return []

    
if __name__=="__main__":
    indexer=PDFIndexer()
    # pdf_path = "/data2/vysakh/alphagrid/docs/Tiger - Senior Facilities Agreement [03.09.2024].pdf"  
    # indexer.main(pdf_path)
    indexer.search_paragraphs("What is the interest rate?", "Tiger - Senior Facilities Agreement [03.09.2024]")
    # data = extract_paragraphs(pdf_path)
    # index_paragraphs(data)