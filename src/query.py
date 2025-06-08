from qdrant_client.models import Filter, FieldCondition, MatchValue
from typing import List, Dict
from loguru import logger as lg
from basevector import BaseVectorStore
import os
from agno.agent import Agent,RunResponse
from agno.tools.reasoning import ReasoningTools
from agno.models.azure import AzureOpenAI
from agno.tools import tool
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

class ResponseModel(BaseModel):
    response :str= Field(..., description="The response from the agent.")
    confidence: float = Field(..., description="The confidence level of the response.")
    retry: bool = Field(..., description="Indicates if the agent should retry the response.")
    
class QueryDecompositionModel(BaseModel):
    query: str = Field(..., description="The original query to be decomposed.")
    confidence: float = Field(..., description="Confidence level of the decomposition.")

class QueryEngine(BaseVectorStore):
    def search_paragraphs(self, query_text: str, pdf_name: str, top_k: int = 5) -> List[Dict]:
        try:
            super().__init__()
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

query_engine = QueryEngine()

@tool
def search_tool(query_text: str, pdf_name: str, top_k: int = 5) -> List[Dict]:
    """
    Search for paragraphs in a specific PDF document based on the query text.
    
    :param query_text: The text to search for in the PDF.
    :param pdf_name: The name of the PDF document to search within.
    :param top_k: The number of top results to return.
    :return: A list of dictionaries containing the search results.
    """
    try:
        lg.info(f"Searching for '{query_text}' in PDF '{pdf_name}' with top_k={top_k}.")
        result= query_engine.search_paragraphs(query_text, pdf_name, top_k)
        if result:result=result[:5]
        return result
    except Exception as e:  
        lg.exception(f"Error in search_tool: {e}")
        return []

class RAGAgent(Agent):
    def __init__(self):
        self.model=AzureOpenAI(
                id=os.getenv("AZURE_GPT_4o_MINI_ID"),   
                temperature=0.01,
                api_version=os.getenv("AZURE_GPT_4o_MINI_API_VERSION"),
                api_key=os.getenv("AZURE_GPT_4o_MINI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_GPT_4o_MINI_ENDPOINT")
            )
        self.rag_agent= Agent(
            description="You are a legal Assistant AI that specializes in summarizing tables of legal documents. Given the image and text of a page from a legal document, your task is to summarize the content of the tables present in the page. If no table is present, return an empty string.",
            model=self.model,
            tools=[ReasoningTools(add_instructions=True,think=True,analyze=True),search_tool],response_model=ResponseModel,
            tool_call_limit=1)
        self.query_decompoer = Agent(
            description="You are a query decomposer AI that specializes in breaking down complex queries into simpler sub-queries. Given a complex query, your task is to decompose it into simpler sub-queries that can be answered by the RAG agent.",
            model=self.model,
            tools=[ReasoningTools(add_instructions=True,think=True,analyze=True)],
            response_model=QueryDecompositionModel,
            tool_call_limit=1
        )
        
    def preprocess(self, pdf_name: str, query_text: str, top_k: int = 5) -> str:
            try:
                prompt_text=f"""Retrieve the most relevant paragraphs from the PDF document '{pdf_name}' based on the query: '{query_text}' for the top {top_k} paragraphs.
                The paragraphs should be relevant to the query and should provide sufficient context for answering the query."""
                return prompt_text  
            except Exception as e:
                lg.exception(f"Error in preprocess method: {e}")
                return ""

    def retrieve(self, query_text: str, pdf_name: str, top_k: int = 4, max_retries: int = 3) -> List[Dict]:
        """
        Retrieve paragraphs from the PDF document based on the query text.
        
        :param query_text: The text to search for in the PDF.
        :param pdf_name: The name of the PDF document to search within.
        :param top_k: The number of top results to return.
        :return: A list of dictionaries containing the search results.
        """
        try:
            default_output={"query":query_text,"response":"Sorry I could not find any relevant information.","confidence":0.0,"retry":False}
            retries=0
            lg.info(f"Starting retrieval for query with RAG agent.")
            while retries < max_retries:
                lg.info(f"Attempt {retries + 1} for query: {query_text}")
                retries += 1
                result:RunResponse=self.query_decompoer.run(query_text)
                lg.info(f"Decomposed query: {result.content}")
                query_text = result.content.query
                prompt=self.preprocess(pdf_name=pdf_name, query_text=query_text, top_k=top_k)
                rag_result:RunResponse=self.rag_agent.run(prompt)
                rag_result_content:ResponseModel=rag_result.content
                if not rag_result_content.retry or retries >=max_retries:
                    if not rag_result_content.retry:
                        lg.info(f"RAG agent returned response: {rag_result_content.response} with confidence {rag_result_content.confidence}")
                    else:
                        lg.warning(f"Max retries reached ({max_retries}). Returning results.")
                    break
            output={"query":query_text,"response":rag_result_content.response,"confidence":rag_result_content.confidence,"retry":rag_result_content.retry}
            return output if rag_result_content.response else default_output
        except Exception as e:
            lg.exception(f"Error in retrieve method: {e}")
            return default_output

if __name__ == "__main__":
    pdf_name='Tiger - Senior Facilities Agreement [03.09.2024]'
    rag_agent = RAGAgent()
    response = rag_agent.retrieve(
        query_text="What is the interest rate?",
        pdf_name="Tiger - Senior Facilities Agreement [03.09.2024]",
        top_k=5
    )
    print(response)