import os
import re
import logging
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, Range
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_sales_system.log'),
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
    embedding_model_name: str = "hkunlp/instructor-large"  # Better for instruction-based tasks
    vector_size: int = 768  # Instructor-large uses 768 dimensions
    distance_metric: Distance = Distance.COSINE
    timeout: int = 300
    default_temperature: float = 0.1
    default_max_tokens: int = 1024
    default_search_limit: int = 10
    min_score_threshold: float = 0.3  # Minimum relevance score to consider

class RAGSystem:
    """Production-ready RAG system for structured sales data with comprehensive error handling"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_model = None
        self.qdrant_client = None
        self.groq_client = None
        
        # Enhanced prompt template for sales data
        self.prompt_template = """
        You are a sales data analyst AI. Use the following context to answer the question accurately.

        Context: 
        {context}

        Question: {question}

        Instructions:
        1. Provide specific, data-driven answers based EXCLUSIVELY on the context
        2. For orders data:
           - Include Order IDs when available
           - Report exact amounts with currency
           - Specify dates in MM-DD-YY format
        3. For summary questions (totals, counts), calculate from context
        4. If context contains multiple records, aggregate when appropriate
        5. If answer can't be determined, say "I couldn't find relevant data"
        6. NEVER invent order details or amounts

        Answer:
        """
        
        try:
            self._initialize_components()
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.critical(f"Critical failure during initialization: {str(e)}")
            raise RuntimeError("System initialization failed") from e
    
    def _initialize_components(self) -> None:
        """Initialize all system components with robust error handling"""
        try:
            # Initialize embedding model
            logger.info(f"Loading embedding model: {self.config.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
            logger.info(f"Embedding model loaded. Vector size: {self.embedding_model.get_sentence_embedding_dimension()}")
            
            # Initialize Qdrant client
            logger.info("Initializing Qdrant client")
            self.qdrant_client = QdrantClient(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                timeout=self.config.timeout
            )
            
            # Test connection
            collections = self.qdrant_client.get_collections()
            logger.info(f"Qdrant connected. Collections: {len(collections.collections)}")
            
            # Initialize Groq client
            logger.info("Initializing Groq client")
            self.groq_client = Groq(api_key=self.config.groq_api_key)
            logger.info("Groq client initialized")
            
        except Exception as e:
            logger.exception("Component initialization failed")
            raise
    
    def create_collection(self, collection_name: str) -> bool:
        """Create a new collection with proper vector size"""
        try:
            if self.qdrant_client.collection_exists(collection_name):
                logger.info(f"Collection '{collection_name}' already exists")
                return True
            
            # Get actual vector size from model
            vector_size = self.embedding_model.get_sentence_embedding_dimension()
            
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size, 
                    distance=self.config.distance_metric
                )
            )
            
            logger.info(f"Collection '{collection_name}' created (vector size: {vector_size})")
            return True
            
        except Exception as e:
            logger.error(f"Collection creation failed: {str(e)}")
            return False
    
    def create_metadata_indexes(self, collection_name: str) -> bool:
        """Create required metadata indexes for filtering"""
        try:
            if not self.qdrant_client.collection_exists(collection_name):
                logger.error(f"Collection '{collection_name}' not found")
                return False
            
            # Define index configurations
            indexes = [
                ("metadata.date", "keyword"),
                ("metadata.status", "keyword"),
                ("metadata.amount", "float"),
                ("metadata.ship_city", "keyword"),
                ("metadata.order_id", "keyword")
            ]
            
            for field, schema_type in indexes:
                try:
                    self.qdrant_client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema=schema_type,
                        wait=True
                    )
                    logger.info(f"Created index for {field} ({schema_type})")
                except Exception as e:
                    if "already exists" in str(e):
                        logger.debug(f"Index for {field} already exists")
                    else:
                        logger.error(f"Failed to create index for {field}: {str(e)}")
            
            return True
        except Exception as e:
            logger.exception("Metadata index creation failed")
            return False
    
    def add_documents(self, collection_name: str, documents: List[str], 
                     metadata: List[Dict]) -> bool:
        """Add documents to collection with metadata and error handling"""
        try:
            if not documents or not metadata:
                logger.error("No documents or metadata provided")
                return False
                
            if len(documents) != len(metadata):
                logger.error("Documents and metadata length mismatch")
                return False
                
            if not self.qdrant_client.collection_exists(collection_name):
                logger.info(f"Creating collection: {collection_name}")
                if not self.create_collection(collection_name):
                    return False
            
            points = []
            success_count = 0
            
            for i, (doc, meta) in enumerate(zip(documents, metadata)):
                try:
                    # Generate embedding
                    embedding = self.embedding_model.encode(doc)
                    
                    # Prepare payload (text + metadata)
                    payload = {"text": doc, "metadata": meta}
                    
                    # Create point with nanosecond timestamp ID
                    point_id = int(time.time_ns()) + i
                    point = PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=payload
                    )
                    points.append(point)
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"Document {i} processing failed: {str(e)}")
                    continue
            
            # Batch upsert in chunks of 100
            for i in range(0, len(points), 100):
                batch = points[i:i+100]
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True
                )
            
            logger.info(f"Added {success_count}/{len(documents)} documents to '{collection_name}'")
            
            # After adding documents, create metadata indexes
            if success_count > 0:
                self.create_metadata_indexes(collection_name)
            
            return success_count > 0
            
        except Exception as e:
            logger.exception("Document addition failed")
            return False
    
    def enhanced_search(self, collection_name: str, query: str, 
                       limit: int = 10, filters: Optional[Dict] = None) -> Optional[List[Any]]:
        """
        Advanced hybrid search with:
        - Query expansion
        - Multiple query representations
        - Metadata filtering
        - Deduplication
        """
        try:
            if not self.qdrant_client.collection_exists(collection_name):
                logger.error(f"Collection '{collection_name}' not found")
                return None
            
            # Step 1: Extract structured parameters from query
            extracted_params = self._extract_query_params(query)
            if filters:
                extracted_params.update(filters)
            logger.info(f"Extracted parameters: {extracted_params}")
            
            # Step 2: Query expansion
            expanded_query = self._expand_query(query)
            logger.debug(f"Expanded query: {expanded_query}")
            
            # Step 3: Generate multiple query representations
            query_representations = self._generate_query_representations(expanded_query)
            logger.info(f"Generated {len(query_representations)} query representations")
            
            # Step 4: Build Qdrant filter
            qdrant_filter = self._build_qdrant_filter(extracted_params)
            
            all_results = []
            
            # Step 5: Search for each representation
            for q_rep in query_representations:
                try:
                    query_embedding = self.embedding_model.encode(q_rep).tolist()
                    
                    search_result = self.qdrant_client.search(
                        collection_name=collection_name,
                        query_vector=query_embedding,
                        query_filter=qdrant_filter,
                        with_payload=True,
                        limit=limit * 2,  # Get extra for deduplication
                        score_threshold=self.config.min_score_threshold
                    )
                    all_results.extend(search_result)
                except Exception as e:
                    logger.error(f"Search failed for '{q_rep}': {str(e)}")
            
            # Step 6: Deduplicate results
            unique_results = self._deduplicate_results(all_results)
            sorted_results = sorted(unique_results, key=lambda x: x.score, reverse=True)
            
            logger.info(f"Retrieved {len(sorted_results)} unique results after deduplication")
            return sorted_results[:limit]
            
        except Exception as e:
            logger.exception("Enhanced search with query_points failed")
            return None
    
    def list_collections(self) -> Optional[List[str]]:
        """List available collections"""
        try:
            collections = self.qdrant_client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {str(e)}")
            return None
        
    def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """Get detailed information about a collection"""
        try:
            if not self.qdrant_client.collection_exists(collection_name):
                logger.warning(f"Collection '{collection_name}' doesn't exist")
                return None
            
            # Get collection info from Qdrant
            collection_info = self.qdrant_client.get_collection(collection_name)
            
            # Get collection statistics
            stats = self.qdrant_client.get_collection_stats(collection_name)
            
            # Format the response
            return {
                "name": collection_name,
                "status": collection_info.status.name,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.name,
                "points_count": stats.points_count,
                "segments_count": stats.segments_count,
                "indexed_vectors_count": stats.indexed_vectors_count,
                "payload_schema": {
                    field: schema.dict() 
                    for field, schema in collection_info.payload_schema.items()
                } if collection_info.payload_schema else {}
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for '{collection_name}': {str(e)}")
            return None

    def _extract_query_params(self, query: str) -> Dict[str, Any]:
        """Robust parameter extraction for sales queries"""
        params = {}
        
        # 1. Date extraction
        date_patterns = [
            r'(\d{2}-\d{2}-\d{2})',  # MM-DD-YY
            r'(\d{2}/\d{2}/\d{2})',  # MM/DD/YY
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(today|yesterday|current)',
            r'last\s+(\d+)\s+(day|week|month)s?'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                date_str = match.group(1).lower() if match.lastindex >= 1 else match.group(0)
                if date_str == 'today':
                    params['date'] = datetime.now().strftime('%m-%d-%y')
                elif date_str == 'yesterday':
                    params['date'] = (datetime.now() - timedelta(days=1)).strftime('%m-%d-%y')
                elif date_str == 'current':
                    params['date'] = datetime.now().strftime('%m-%d-%y')
                elif pattern.startswith('last'):
                    num = int(match.group(2))
                    unit = match.group(3)
                    days = num * 7 if 'week' in unit else num * 30 if 'month' in unit else num
                    start_date = (datetime.now() - timedelta(days=days)).strftime('%m-%d-%y')
                    end_date = datetime.now().strftime('%m-%d-%y')
                    params['date_range'] = (start_date, end_date)
                else:
                    # Standardize to MM-DD-YY format
                    try:
                        dt = datetime.strptime(date_str, '%m-%d-%y')
                        params['date'] = dt.strftime('%m-%d-%y')
                    except ValueError:
                        try:
                            dt = datetime.strptime(date_str, '%m/%d/%y')
                            params['date'] = dt.strftime('%m-%d-%y')
                        except ValueError:
                            pass
                break
        
        # 2. Status extraction
        status_keywords = {
            'cancelled': ['cancel', 'cancelled', 'terminated'],
            'shipped': ['ship', 'delivered', 'sent'],
            'pending': ['pending', 'waiting', 'processing']
        }
        for status, keywords in status_keywords.items():
            if any(fr'\b{kw}\b' in query.lower() for kw in keywords):
                params['status'] = status
                break
        
        # 3. Numeric extraction - improved handling
        amount_pattern = r'[\$\₹\€]?\s*(\d+(?:,\d{3})*(?:\.\d+)?)\b'
        amount_matches = re.findall(amount_pattern, query)
        if amount_matches:
            try:
                # Take the last mentioned amount in the query
                amount_str = amount_matches[-1].replace(',', '')
                params['amount'] = float(amount_str)
            except ValueError:
                pass
        
        # 4. Location extraction
        location_map = {
            'ship_city': ['city', 'town'],
            'ship_state': ['state', 'province', 'region'],
            'ship_country': ['country', 'nation']
        }
        for field, terms in location_map.items():
            for term in terms:
                if term in query.lower():
                    # Extract next word as location
                    parts = query.lower().split(term)
                    if len(parts) > 1:
                        location = parts[1].split()[0].strip()
                        if location:
                            params[field] = location.upper()
                            break
        
        # 5. Order ID extraction
        order_id_match = re.search(r'order\s+([A-Z0-9-]+)', query, re.IGNORECASE)
        if order_id_match:
            params['order_id'] = order_id_match.group(1)
        
        logger.debug(f"Extracted parameters: {params}")
        return params
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and domain terms"""
        expansions = {
            'order': ['purchase', 'transaction', 'sale'],
            'sales': ['revenue', 'income', 'turnover'],
            'status': ['state', 'condition', 'stage'],
            'date': ['time', 'day', 'when'],
            'city': ['location', 'place', 'area'],
            'cancel': ['void', 'terminate', 'abort'],
            'amount': ['value', 'price', 'cost', 'total']
        }
        
        expanded = query.lower()
        for term, synonyms in expansions.items():
            if term in expanded:
                expanded += " " + " ".join(synonyms)
                
        # Add analytical terms for sales domain
        if any(word in expanded for word in ['trend', 'compare', 'growth', 'change', 'analysis']):
            expanded += " trend analysis comparison growth rate percentage change"
            
        return expanded
    
    def _generate_query_representations(self, query: str) -> List[str]:
        """Generate multiple representations of the query for different aspects"""
        representations = [query]
        
        # Numerical-focused
        if any(word in query for word in ['number', 'count', 'total', 'sum', 'amount']):
            representations.append(f"{query} numerical value exact match")
            
        # Temporal-focused
        if any(word in query for word in ['date', 'day', 'time', 'when']):
            representations.append(f"{query} date time chronological")
            
        # Status-focused
        if any(word in query for word in ['status', 'state', 'condition']):
            representations.append(f"{query} order status condition")
            
        # Location-focused
        if any(word in query for word in ['city', 'state', 'country', 'location']):
            representations.append(f"{query} geographic location place")
            
        return list(set(representations))  # Remove duplicates
    
    def _build_qdrant_filter(self, params: Dict) -> Optional[Filter]:
        """Build Qdrant filter from extracted parameters"""
        conditions = []
        
        # Date filter
        if 'date' in params:
            conditions.append(FieldCondition(
                key="metadata.date",
                match=MatchValue(value=params['date'])
            ))
        elif 'date_range' in params:
            start, end = params['date_range']
            conditions.append(FieldCondition(
                key="metadata.date",
                range=Range(gte=start, lte=end)
            ))
        
        # Status filter
        if 'status' in params:
            conditions.append(FieldCondition(
                key="metadata.status",
                match=MatchValue(value=params['status'])
            ))
        
        # Amount filter (±10%)
        if 'amount' in params:
            amount = params['amount']
            conditions.append(FieldCondition(
                key="metadata.amount",
                range=Range(gte=amount*0.9, lte=amount*1.1)
            ))
        
        # Location filters
        for loc_type in ['city', 'state', 'country']:
            key = f"ship_{loc_type}"
            if key in params:
                conditions.append(FieldCondition(
                    key=f"metadata.{key}",
                    match=MatchValue(value=params[key])
                ))
        
        # Order ID filter
        if 'order_id' in params:
            conditions.append(FieldCondition(
                key="metadata.order_id",
                match=MatchValue(value=params['order_id'])
            ))
        
        return Filter(must=conditions) if conditions else None
    
    def _deduplicate_results(self, results: List[Any]) -> List[Any]:
        """Deduplicate results based on order ID while keeping highest score"""
        unique_results = {}
        for result in results:
            try:
                order_id = result.payload['metadata']['order_id']
                if order_id not in unique_results or result.score > unique_results[order_id].score:
                    unique_results[order_id] = result
            except KeyError:
                # If no order_id, keep all results
                unique_results[id(result)] = result
        return list(unique_results.values())
    
    def generate_response(self, context: str, question: str, 
                         model: str = "llama3-70b-8192",
                         temperature: float = None,
                         max_tokens: int = None,
                         custom_prompt_template: str = None) -> Optional[str]:
        """Generate response with enhanced error handling"""
        try:
            temperature = temperature or self.config.default_temperature
            max_tokens = max_tokens or self.config.default_max_tokens
            prompt_template = custom_prompt_template or self.prompt_template
            
            filled_prompt = prompt_template.format(
                context=context,
                question=question
            )
            
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise sales data analyst."},
                    {"role": "user", "content": filled_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30  # Prepend long-running requests
            )
            
            answer = response.choices[0].message.content
            return answer
            
        except Exception as e:
            logger.error(f"Response generation failed: {str(e)}")
            return None
    
    def query(self, collection_name: str, question: str, 
              search_limit: int = None,
              model: str = "llama3-70b-8192",
              temperature: float = None,
              max_tokens: int = None) -> Dict[str, Any]:
        """Complete RAG query pipeline with enhanced search"""
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
            # Enhanced hybrid search
            search_results = self.enhanced_search(
                collection_name=collection_name,
                query=question,
                limit=search_limit or self.config.default_search_limit
            )
            
            if not search_results:
                result["error"] = "No relevant documents found"
                return result
            
            # Build context from top results
            context_texts = []
            for res in search_results:
                try:
                    context_texts.append(res.payload['text'])
                except KeyError:
                    logger.warning("Search result missing text payload")
            
            context = "\n\n".join(context_texts)
            
            # Generate response
            answer = self.generate_response(
                context=context,
                question=question,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            if not answer:
                result["error"] = "Response generation failed"
                return result
            
            # Prepare search results metadata
            search_meta = []
            for res in search_results:
                try:
                    search_meta.append({
                        "score": res.score,
                        "text": res.payload['text'][:200] + "..." if len(res.payload['text']) > 200 else res.payload['text'],
                        "metadata": res.payload['metadata']
                    })
                except KeyError:
                    continue
            
            # Successful result
            result.update({
                "success": True,
                "answer": answer,
                "context": context_texts,
                "search_results": search_meta,
                "execution_time": time.time() - start_time
            })
            
            return result
            
        except Exception as e:
            result["error"] = f"Query processing failed: {str(e)}"
            result["execution_time"] = time.time() - start_time
            logger.exception("Query processing error")
            return result

def preprocess_sales_data(df: pd.DataFrame) -> Tuple[List[str], List[Dict]]:
    """
    Preprocess sales data into documents and metadata
    Returns:
        documents: List of text descriptions
        metadata: List of metadata dictionaries
    """
    documents = []
    metadata_list = []
    
    # Clean and standardize data
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%m-%d-%y')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df.fillna('', inplace=True)
    
    for _, row in df.iterrows():
        try:
            # Create rich text description
            doc_text = (
                f"Order {row['Order ID']} ({row['Status']}) - "
                f"Date: {row['Date']}, "
                f"Amount: {row['Amount']:.2f} {row['currency']}, "
                f"Product: {row['Style']} ({row['SKU']}), "
                f"Category: {row['Category']}, Size: {row['Size']}, "
                f"Ship to: {row['ship-city']}, {row['ship-state']} {row['ship-postal-code']}, "
                f"Fulfilled by: {row['fulfilled-by']}, "
                f"Courier: {row['Courier Status']}"
            )
            
            # Create comprehensive metadata
            meta = {
                "order_id": row['Order ID'],
                "date": row['Date'],
                "status": row['Status'].lower(),
                "amount": float(row['Amount']) if not pd.isna(row['Amount']) else 0.0,
                "currency": row['currency'],
                "product_style": row['Style'],
                "sku": row['SKU'],
                "category": row['Category'],
                "size": row['Size'],
                "ship_city": row['ship-city'].upper(),
                "ship_state": row['ship-state'].upper(),
                "ship_country": row['ship-country'].upper(),
                "ship_postal_code": str(row['ship-postal-code']),
                "fulfilment": row['Fulfilment'],
                "fulfilled_by": row['fulfilled-by'],
                "b2b": bool(row['B2B']),
                "courier_status": row['Courier Status']
            }
            
            documents.append(doc_text)
            metadata_list.append(meta)
        except Exception as e:
            logger.error(f"Row processing failed: {str(e)}")
            continue
    
    logger.info(f"Preprocessed {len(documents)} sales records")
    return documents, metadata_list

def main():
    """Main execution with comprehensive error handling"""
    try:
        # Configuration
        config = RAGConfig(
            qdrant_url=os.getenv('QDRANT_URL'),
            qdrant_api_key=os.getenv('QDRANT_API_KEY'),
            groq_api_key=os.getenv('GROQ_API_KEY')
        )
        
        # Initialize RAG system
        rag = RAGSystem(config)
        logger.info("RAG system initialized")
        
        # Load and preprocess data
        try:
            sales_df = pd.read_csv('data/final/small_final_sales_data.csv')
            logger.info(f"Loaded sales data: {len(sales_df)} records")
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            return
        
        documents, metadata = preprocess_sales_data(sales_df)
        collection_name = "sales_data_v2"
        
        # Add documents to vector store
        if not rag.add_documents(collection_name, documents, metadata):
            logger.error("Document ingestion failed")
            return
        
        # Test queries with different types
        queries = [
            "Get all orders from 04-30-22",
            "Show me canceled orders from Mumbai",
            "What is the status of order 405-8078784-5731545?",
            "List shipped orders over Rs 400 from Bengaluru last week",
            "How many orders were cancelled yesterday?"
        ]
        
        for query in queries:
            print(f"\n{'='*50}")
            print(f"QUERY: {query}")
            start_time = time.time()
            
            result = rag.query(collection_name, query, search_limit=15)
            
            if result["success"]:
                print(f"\nANSWER ({result['execution_time']:.2f}s):")
                print(result["answer"])
                
                print("\nTOP CONTEXT:")
                for i, res in enumerate(result["search_results"][:3]):
                    print(f"{i+1}. [Score: {res['score']:.3f}] {res['text']}")
                    print(f"   Metadata: {res['metadata']}")
            else:
                print(f"ERROR: {result['error']}")
            
            print(f"{'='*50}\n")
        
        # Collection info
        collections = rag.list_collections()
        if collections:
            print(f"\nAvailable collections: {collections}")
        
    except Exception as e:
        logger.exception("Fatal error in main execution")

if __name__ == "__main__":
    main()