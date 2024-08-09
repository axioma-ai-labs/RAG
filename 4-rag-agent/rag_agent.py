import os
import ast
import logging
from typing import List, Dict, Tuple, Optional

from langchain.text_splitter import NLTKTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI

# configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class RAGAgent:
    """
    RAG Agent with semantic chunking, document relevance grading and re-runking w/ RRF
    """
    def __init__(self):
        self.init_components()

    def init_components(self):
        """Initializes all necessary components for the agent."""
        self.text_splitter = NLTKTextSplitter()
        self.embedding_model = SentenceTransformerEmbeddings(model_name="thenlper/gte-large")
        self.output_parser = JsonOutputParser()
        self.vector_store = Chroma
        self.azure_openai = AzureChatOpenAI(
            openai_api_version=os.getenv('AZURE_OPENAI_VERSION', 'default_version'),
            azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'default_deployment'),
            temperature=0
        )
        self.llama = ChatOllama(model="llama3", format="json", temperature=0)

    def load_documents(self, data_path: str) -> List[str]:
        """Loads and prepares PDF documents from a directory."""
        try:
            pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
            docs = [PyPDFLoader(os.path.join(data_path, file)).load() for file in pdf_files]
            logger.info(f"Loaded {len(docs)} files")
            return docs
        except Exception as e:
            logger.error("Failed to load documents", exc_info=True)
            return []

    def preprocess_documents(self, docs: List[str]) -> List[str]:
        """Preprocesses and chunks documents into manageable parts."""
        try:
            docs_list = [item for sublist in docs for item in sublist]
            chunks = self.text_splitter.split_documents(docs_list)
            logger.info(f"Documents chunked. Total no. of chunks: {len(chunks)}")
            return chunks
        except Exception as e:
            logger.error("Failed to preprocess documents", exc_info=True)
            return []

    def setup_vector_store(self, chunks: List[str]) -> Optional[Chroma]:
        """Initializes and returns a ChromaDB vector store with documents and embedding model."""
        try:
            db = self.vector_store.from_documents(documents=chunks, embedding=self.embedding_model, collection_metadata={"hnsw:space": "cosine"})
            logger.info("Vector store successfully created")
            return db
        except Exception as e:
            logger.error("Failed to initialize vector store", exc_info=True)
            return None

    def retrieve_documents(self, db, user_query: str) -> List[str]: 
        """Retrieves documents relevant to the given user query."""
        try:
            retriever = db.as_retriever(search_type="mmr")
            documents = retriever.invoke(user_query)
            logger.info(f"Successfully retrieved {len(documents)} documents")
            return documents
        except Exception as e:
            logger.error("Failed to retrieve documents", exc_info=True)
            return []

    def grade_document_relevance(self, user_query: str, documents: List[str]) -> List[str]: 
        """Determines the relevance of each document to the user query."""
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, 
            grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
            Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
            Here is the retrieved document: \n\n {document} \n\n
            Here is the user question: {user_query} \n
            """,
            input_variables=["user_query", "document"],
        )

        retrieval_grader = prompt | self.azure_openai | JsonOutputParser()
        relevant_docs = []

        for doc in documents:
            input_data = {"user_query": user_query, "document": doc.page_content}
            try:
                result = retrieval_grader.invoke(input=input_data)
                if result.get('score') == 'yes':
                    relevant_docs.append(doc)
            except Exception as e:
                logger.error("Error processing document", exc_info=True)

        logger.info(f"Grader: {len(relevant_docs)} relevant documents out of {len(documents)}")
        return relevant_docs

    def generate_similar_search_queries(self, user_query: str, documents: List[str]) -> List[str]:
        """Generates similar search queries based on the provided user query and documents."""
        message = HumanMessage(
            content=f""" 
                You are a helpful search assistant. Your task is to generate four similar search queries in relation to the provided documents based on a single input query.
                Always use provided output for your response. Be concise and constructive. Do not deviate from the context of the provided documents.
                Initial single input query: {user_query}
                Documents: {documents}
                Output structure: ["{user_query}", search query 1, search query 2, search query 3, search query 4]
            """
        )
        try:
            response = self.azure_openai.invoke([message])
            queries = ast.literal_eval(response.content)
            logger.info("Successfully generated similar search queries.")
            return queries
        except Exception as e:
            logger.error("Failed to generate similar search queries", exc_info=True)
            return []

    def simulate_search_results(self, db, queries: List[str]) -> Dict[str, List[Tuple[str, float]]]:
        """Simulates search results for given queries using a vector store."""
        simulated_results = {}
        try:
            for query in queries:
                results = db.similarity_search_with_score(query, k=4)
                simulated_results[query] = results
            logger.info("Successfully simulated search results.")
        except Exception as e:
            logger.error("Failed to simulate search results", exc_info=True)
        return simulated_results

    def reciprocal_rank_fusion(self, search_results_dict: Dict[str, List[Tuple[str, float]]], k: int = 5) -> Dict[str, Tuple[float, str]]:
        """Applies reciprocal rank fusion to re-rank search results."""
        fused_scores = {}
        try:
            for query, doc_scores in search_results_dict.items():
                for rank, (doc, score) in enumerate(sorted(doc_scores, key=lambda x: x[1], reverse=True)):
                    doc_identifier = doc.page_content
                    doc_source = doc.metadata.get('source', 'unknown')
                    if doc_identifier not in fused_scores:
                        fused_scores[doc_identifier] = [0, doc_source]
                    fused_scores[doc_identifier][0] += 1 / (rank + k)
            reranked_results = {doc: (score, source) for doc, (score, source) in sorted(fused_scores.items(), key=lambda x: x[1][0], reverse=True)}
            logger.info("Documents re-ranked successfully using reciprocal rank fusion.")
        except Exception as e:
            logger.error("Failed to apply reciprocal rank fusion", exc_info=True)
            reranked_results = {}
        return reranked_results

    def generate_answer(self, reranked_results: Dict[str, Tuple[float, str]], user_query: str, top_n_results: int = 3):
        """Generates an answer to the user query using the top N reranked results."""
        prompt = PromptTemplate(
            template="""
                You are an assistant for question-answering tasks. 
                Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
                Be as concise and as precise as possible. Use maximum of 400 completion tokens for your response.
                Question: {question} 
                Context: {context} 
                Answer: 
            """,
            input_variables=["question", "context"],
        )
        try:
            top_reranked_results = dict(list(reranked_results.items())[:top_n_results])
            context = '. '.join([doc for doc, _ in top_reranked_results.items()])
            rag_chain = prompt | self.azure_openai | StrOutputParser()
            final_response = rag_chain.invoke({"context": context, "question": user_query})
            logger.info("Answer generated successfully.")
            return final_response
        except Exception as e:
            logger.error("Failed to generate answer", exc_info=True)
            return "Unable to generate an answer at this time."