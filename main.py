import os
import logging
import time
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration class for RAG system"""
    qdrant_url: str
    qdrant_api_key: str
    groq_api_key: str
    # embedding_model_name: str = 'all-MiniLM-L6-v2'
    # embedding_model_name: str = "hkunlp/instructor-large"
    embedding_model_name: str = 'mixedbread-ai/mxbai-embed-large-v1'
    vector_size: int = 1024
    distance_metric: Distance = Distance.COSINE
    timeout: int = 300
    default_temperature: float = 0.1
    default_max_tokens: int = 2048
    default_search_limit: int = 10

class RAGSystem:
    """Production-ready RAG system with comprehensive error handling and logging"""
    
    def __init__(self, config: RAGConfig):
        """Initialize the RAG system with configuration"""
        self.config = config
        self.embedding_model = None
        self.qdrant_client = None
        self.groq_client = None
        
        # Default prompt template
        self.prompt_template = """
        You are a helpful AI assistant. Use the following context to answer the question accurately.

        Context: {context}

        Question: {question}

        Instructions:
        - Provide specific, data-driven answers based on the context
        - If the context contains numerical data, include relevant numbers in your response
        - If you cannot find the answer in the context, say so clearly
        - Be concise but comprehensive
        - For questions about trends or patterns, explain what the data shows
        - Do not make up data, only give answers based on relevant context

        Answer:
        """
        
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all system components with error handling"""
        try:
            # Initialize embedding model
            logger.info(f"Loading embedding model: {self.config.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
            logger.info("Embedding model loaded successfully")
            
            # Initialize Qdrant client
            logger.info("Initializing Qdrant client")
            self.qdrant_client = QdrantClient(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                timeout=self.config.timeout
            )
            
            # Test connection
            collections = self.qdrant_client.get_collections()
            logger.info(f"Qdrant connection successful. Found {len(collections.collections)} collections")
            
            # Initialize Groq client
            logger.info("Initializing Groq client")
            self.groq_client = Groq(api_key=self.config.groq_api_key)
            logger.info("Groq client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            raise
    
    def create_collection(self, collection_name: str, vector_size: Optional[int] = None) -> bool:
        """Create a new collection if it doesn't exist"""
        try:
            if self.qdrant_client.collection_exists(collection_name):
                logger.info(f"Collection '{collection_name}' already exists")
                return True
            
            vector_size = vector_size or self.config.vector_size
            
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size, 
                    distance=self.config.distance_metric
                )
            )
            
            logger.info(f"Collection '{collection_name}' created successfully with vector size {vector_size}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection '{collection_name}': {str(e)}")
            return False
    
    def add_documents(self, collection_name: str, documents: List[str], 
                     metadata: Optional[List[Dict]] = None) -> bool:
        """Add documents to a collection with error handling"""
        try:
            if not self.qdrant_client.collection_exists(collection_name):
                logger.warning(f"Collection '{collection_name}' doesn't exist. Creating it...")
                if not self.create_collection(collection_name):
                    return False
            
            points = []
            failed_docs = []
            
            for i, doc in enumerate(documents):
                try:
                    # Generate embedding
                    embedding = self.embedding_model.encode(doc)
                    
                    # Prepare payload
                    payload = {"text": doc}
                    if metadata and i < len(metadata):
                        payload.update(metadata[i])
                    
                    # Create point
                    point = PointStruct(
                        id=int(time.time() * 1000000) + i,  # Unique timestamp-based ID
                        vector=embedding.tolist(),
                        payload=payload
                    )
                    points.append(point)
                    
                except Exception as e:
                    logger.error(f"Failed to process document {i}: {str(e)}")
                    failed_docs.append(i)
                    continue
            
            if not points:
                logger.error("No valid documents to insert")
                return False
            
            # Batch upsert
            result = self.qdrant_client.upsert(
                collection_name=collection_name,
                wait=True,
                points=points
            )
            
            success_count = len(points)
            logger.info(f"Successfully added {success_count} documents to '{collection_name}'")
            
            if failed_docs:
                logger.warning(f"Failed to process {len(failed_docs)} documents: {failed_docs}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to '{collection_name}': {str(e)}")
            return False
    
    def search_vectors(self, collection_name: str, query: str, 
                      limit: int = 10, score_threshold: float = 0.0) -> Optional[List[Any]]:
        """Search for similar vectors with error handling"""
        try:
            if not self.qdrant_client.collection_exists(collection_name):
                logger.error(f"Collection '{collection_name}' doesn't exist")
                return None
            
            limit = limit or self.config.default_search_limit
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Search vectors
            results = self.qdrant_client.search(
                collection_name=collection_name,
                query_vector=query_embedding.tolist(),
                with_payload=True,
                limit=limit,
                score_threshold=score_threshold
            )
            
            logger.info(f"Found {len(results)} results for query in '{collection_name}'")
            
            # Log search results for debugging
            for i, result in enumerate(results):
                logger.debug(f"Result {i+1}: Score={result.score:.4f}, Text={result.payload.get('text', '')[:50]}...")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search vectors in '{collection_name}': {str(e)}")
            return None
    
    def generate_response(self, context: str, question: str, 
                         model: str = "llama3-70b-8192",
                         temperature: float = None,
                         max_tokens: int = None,
                         custom_prompt_template: str = None) -> Optional[str]:
        """Generate response using Groq with error handling"""
        try:
            temperature = temperature or self.config.default_temperature
            max_tokens = max_tokens or self.config.default_max_tokens
            prompt_template = custom_prompt_template or self.prompt_template
            
            # Format the prompt
            filled_prompt = prompt_template.format(
                context=context,
                question=question
            )
            
            # Generate response
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI assistant that provides accurate answers based on the given context."
                    },
                    {
                        "role": "user",
                        "content": filled_prompt
                    }
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            answer = response.choices[0].message.content
            logger.info(f"Generated response using model '{model}'")
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            return None
    
    def query(self, collection_name: str, question: str, 
              search_limit: int = None,
              score_threshold: float = 0.0,
              model: str = "llama3-70b-8192",
              temperature: float = None,
              max_tokens: int = None,
              custom_prompt_template: str = None) -> Dict[str, Any]:
        """Complete RAG query with comprehensive error handling"""
        
        start_time = time.time()
        result = {
            "success": False,
            "answer": None,
            "context": None,
            "search_results": None,
            "execution_time": 0,
            "error": None
        }
        
        try:
            logger.info(f"Processing query: '{question}' in collection '{collection_name}'")
            
            # Step 1: Search for relevant vectors
            search_results = self.search_vectors(
                collection_name=collection_name,
                query=question,
                limit=search_limit,
                score_threshold=score_threshold
            )
            
            if not search_results:
                result["error"] = f"No relevant documents found in collection '{collection_name}'"
                logger.warning(result["error"])
                return result
            
            # Step 2: Build context
            context_texts = [point.payload.get('text', '') for point in search_results]
            context = "\n".join(context_texts)
            
            if not context.strip():
                result["error"] = "Retrieved documents contain no text content"
                logger.warning(result["error"])
                return result
            
            # Step 3: Generate response
            answer = self.generate_response(
                context=context,
                question=question,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                custom_prompt_template=custom_prompt_template
            )
            
            if not answer:
                result["error"] = "Failed to generate response from LLM"
                return result
            
            # Success
            result.update({
                "success": True,
                "answer": answer,
                "context": context_texts,
                "search_results": [
                    {
                        "score": point.score,
                        "text": point.payload.get('text', ''),
                        "metadata": {k: v for k, v in point.payload.items() if k != 'text'}
                    }
                    for point in search_results
                ],
                "execution_time": time.time() - start_time
            })
            
            logger.info(f"Query completed successfully in {result['execution_time']:.2f}s")
            return result
            
        except Exception as e:
            result["error"] = f"Unexpected error during query: {str(e)}"
            result["execution_time"] = time.time() - start_time
            logger.error(result["error"])
            return result
    
    def list_collections(self) -> Optional[List[str]]:
        """List all available collections"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            logger.info(f"Found collections: {collection_names}")
            return collection_names
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            return None
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """Get information about a specific collection"""
        try:
            if not self.qdrant_client.collection_exists(collection_name):
                logger.warning(f"Collection '{collection_name}' doesn't exist")
                return None
            
            info = self.qdrant_client.get_collection(collection_name)
            return {
                "name": collection_name,
                "points_count": info.points_count,
                "vector_size": info.config.params.vectors.size,
                "distance_metric": info.config.params.vectors.distance
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for '{collection_name}': {str(e)}")
            return None

def main():
    """Example usage of the RAG system"""
    
    # Configuration
    config = RAGConfig(
        qdrant_url=os.getenv('QDRANT_URL'),
        qdrant_api_key=os.getenv('QDRANT_API_KEY'),
        groq_api_key=os.getenv('GROQ_API_KEY')
    )
    
    # Initialize RAG system
    try:
        rag = RAGSystem(config)
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        return
    
    # Example data
    # collection_name = "test_collection"
    collection_name = "test_sales_data"
    # sales_df = pd.read_csv('data/final/final_sales_data.csv')
    # documents = sales_df['text'].tolist()

    
    # Add documents
    # if rag.add_documents(collection_name, documents):
    #     logger.info("Documents added successfully")
    # else:
    #     logger.error("Failed to add documents")
    #     return
    
    
    # Query the system
    query = "What is the satus of Order on 04-30-22 ?"
    # query = "What is the highest number of orders shipped in Bengaluru?"

    result = rag.query(collection_name, query)
    
    if result["success"]:
        print(f"\nQuestion: {query}")
        print(f"Answer: {result['answer']}")
        print(f"Execution time: {result['execution_time']:.2f}s")
        print("\nSearch Results:")
        for i, res in enumerate(result['search_results']):
            print(f"{i+1}. Score: {res['score']:.4f} - {res['text']}")
    else:
        print(f"Query failed: {result['error']}")
    
    # List collections
    collections = rag.list_collections()
    if collections:
        print(f"\nAvailable collections: {collections}")
    
    # Get collection info
    info = rag.get_collection_info(collection_name)
    if info:
        print(f"\nCollection info: {info}")

if __name__ == "__main__":
    main()