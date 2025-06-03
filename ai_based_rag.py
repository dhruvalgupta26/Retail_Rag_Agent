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
import json

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
    embedding_model_name: str = "hkunlp/instructor-large"
    vector_size: int = 768
    distance_metric: Distance = Distance.COSINE
    timeout: int = 300
    default_temperature: float = 0.1
    default_max_tokens: int = 1024
    default_search_limit: int = 10
    min_score_threshold: float = 0.3

class RAGSystem:
    """Production-ready RAG system with AI-enhanced query processing"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_model = None
        self.qdrant_client = None
        self.groq_client = None
        
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
        3. For summary questions, calculate from context
        4. If answer can't be determined, say "I couldn't find relevant data"
        5. NEVER invent order details or amounts

        Answer:
        """

        # Query processing prompt
        self.query_processing_prompt = """
        You are an expert in processing sales-related queries. Your task is to:
        1. Extract structured parameters from the query
        2. Generate expanded query versions
        3. Create multiple query representations
        4. Build Qdrant filter conditions

        Input Query: {query}

        Output JSON:
        ```json
        {{
            "parameters": {{
                "date": "MM-DD-YY",
                "date_range": ["MM-DD-YY", "MM-DD-YY"],
                "status": "status_value",
                "amount": float,
                "ship_city": "city_name",
                "ship_state": "state_name",
                "ship_country": "country_name",
                "order_id": "order_id_value"
            }},
            "expanded_query": "expanded query text",
            "query_representations": ["rep1", "rep2"],
            "qdrant_filter": {{
                "must": [
                    {{"key": "metadata.field", "match": {{"value": "value"}}}},
                    {{"key": "metadata.field", "range": {{"gte": "value", "lte": "value"}}}}
                ]
            }}
        }}
        Instructions:
        Return Only Valid Json
        For dates, convert to MM-DD-YY format
        For date ranges (e.g., "last week"), provide start and end dates
        For date ranges, use a single range condition in the format: {{"key": "metadata.date", "range": {{"gte": "MM-DD-YY", "lte": "MM-DD-YY"}}}}
        Do NOT create separate match conditions for each date in the range
        For status, use: cancelled, shipped, pending
        For amounts, extract numeric values (support $, ₹, €)
        For locations, normalize to uppercase
        Generate at least 2 query representations
        Create Qdrant filter conditions in the exact format shown
        If a parameter is not found, omit it from the parameters dictionary
        Handle relative dates (today, yesterday, last X days/weeks/months)
        Ensure the JSON output is valid:
        Include commas between fields in objects (e.g., {{"key1": "value1", "key2": "value2"}})
        Include commas between elements in arrays (e.g., ["item1", "item2"])
        Do NOT include trailing commas (e.g., ["item1", "item2",])
        Do NOT include newlines (\n) within string values; replace them with spaces
        Ensure all objects and arrays are properly closed
        Do NOT include any additional text or comments outside the JSON structure"""
        
        try:
            self._initialize_components()
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.critical(f"Critical failure during initialization: {str(e)}")
            raise RuntimeError("System initialization failed") from e
    
    def _initialize_components(self) -> None:
        """Initialize all system components with robust error handling"""
        try:
            logger.info(f"Loading embedding model: {self.config.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
            logger.info(f"Embedding model loaded. Vector size: {self.embedding_model.get_sentence_embedding_dimension()}")
            
            logger.info("Initializing Qdrant client")
            self.qdrant_client = QdrantClient(
                url=self.config.qdrant_url,
                api_key=self.config.qdrant_api_key,
                timeout=self.config.timeout
            )
            
            collections = self.qdrant_client.get_collections()
            logger.info(f"Qdrant connected. Collections: {len(collections.collections)}")
            
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
                    embedding = self.embedding_model.encode(doc)
                    payload = {"text": doc, "metadata": meta}
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
            
            for i in range(0, len(points), 100):
                batch = points[i:i+100]
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=batch,
                    wait=True
                )
            
            logger.info(f"Added {success_count}/{len(documents)} documents to '{collection_name}'")
            if success_count > 0:
                self.create_metadata_indexes(collection_name)
            
            return success_count > 0
            
        except Exception as e:
            logger.exception("Document addition failed")
            return False

    def _process_query_with_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM to process query for parameters, expansion, representations, and filters"""
        try:
            try:
                logger.info(f"Input Query: {query}")
                filled_prompt = self.query_processing_prompt.format(query=query)
            except Exception as e:
                logger.error(f"Error in filling prompt template: {str(e)}")
                return self._get_fallback_result(query)

            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a precise query processing AI. Always return valid JSON without markdown formatting."},
                    {"role": "user", "content": filled_prompt}
                ],
                temperature=self.config.default_temperature,
                max_tokens=self.config.default_max_tokens,
                timeout=30
            )
            
            raw_content = response.choices[0].message.content
            logger.debug(f"Raw LLM response: {raw_content}")
            
            # Extract JSON from response
            json_str = self._extract_json_from_response(raw_content)
            
            if not json_str:
                logger.error("No valid JSON found in response")
                return self._get_fallback_result(query)
            
            try:
                result = json.loads(json_str)
                logger.info(f"Successfully parsed JSON: {result}")
                return result
                
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON parsing failed: {json_error}")
                logger.error(f"Raw content that failed to parse: {repr(json_str)}")
                return self._get_fallback_result(query)

        except Exception as e:
            logger.error(f"LLM query processing failed: {str(e)}")
            return self._get_fallback_result(query)

    def _extract_json_from_response(self, raw_content: str) -> str:
        """Extract JSON from LLM response, handling various formatting"""
        if not raw_content:
            return ""
        
        content = raw_content.strip()
        
        # Remove markdown code blocks
        if content.startswith("```json"):
            content = content[7:]  # Remove ```json
        elif content.startswith("```"):
            content = content[3:]   # Remove ```
        
        if content.endswith("```"):
            content = content[:-3]  # Remove trailing ```
        
        content = content.strip()
        
        # Try to find JSON object boundaries
        json_start = -1
        json_end = -1
        
        # Look for opening brace
        for i, char in enumerate(content):
            if char == '{':
                json_start = i
                break
        
        if json_start == -1:
            return ""
        
        # Find matching closing brace
        brace_count = 0
        for i in range(json_start, len(content)):
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break
        
        if json_end == -1:
            return ""
        
        return content[json_start:json_end]

    def _get_fallback_result(self, query: str) -> Dict[str, Any]:
        """Return fallback structure when parsing fails"""
        return {
            "parameters": {},
            "expanded_query": query,
            "query_representations": [query],
            "qdrant_filter": None
        }
    # def _process_query_with_llm(self, query: str) -> Dict[str, Any]:
    #     """Use LLM to process query for parameters, expansion, representations, and filters"""
    #     try:
    #         try:
    #             logger.info(f"Input Query: {query}")
    #             filled_prompt = self.query_processing_prompt.format(query=query)
    #         except Exception as e:
    #             logger.error(f"Error in Filling prompt template: {str(e)}")

    #         response = self.groq_client.chat.completions.create(
    #             model="llama3-70b-8192",
    #             messages=[
    #                 {"role": "system", "content": "You are a precise query processing AI."},
    #                 {"role": "user", "content": filled_prompt}
    #             ],
    #             temperature=self.config.default_temperature,
    #             max_tokens=self.config.default_max_tokens,
    #             timeout=30
    #         )
    #         raw_content = response.choices[0].message.content

    #         # Safe JSON extraction
    #         json_str = raw_content.strip()
    #         if json_str.startswith("```json"):
    #             json_str = json_str.strip("```json").strip("```").strip()
    #         elif json_str.startswith("```"):
    #             json_str = json_str.strip("```").strip()

    #         # Sanitize the JSON string
    #         # Replace unescaped newlines within strings with spaces
    #         json_str = re.sub(r'(?<![\]\}]),\s*\n', ' ', json_str)
    #         # Fix missing commas between object fields (e.g., "key1": "value1" "key2": "value2" -> "key1": "value1", "key2": "value2")
    #         json_str = re.sub(r'("[^"]*"\s*:\s*"[^"]*")\s+("[^"]*"\s*:)', r'\1,\2', json_str)
    #         # Fix missing commas between array elements (e.g., "item1" "item2" -> "item1", "item2")
    #         json_str = re.sub(r'("[^"]*")\s+("[^"]*")', r'\1,\2', json_str)
    #         # Remove trailing commas
    #         json_str = re.sub(r',\s*([\]\}])', r'\1', json_str)
    #         try:
    #             result = json.loads(json_str)
    #             return result
    #         except json.JSONDecodeError as json_error:
    #             logger.error(f"JSON parsing failed: {json_error}")
    #             logger.error(f"Raw content that failed to parse: {json_str}")
    #             # Return a fallback structure
    #             return {
    #                 "parameters": {},
    #                 "expanded_query": query,
    #                 "query_representations": [query],
    #                 "qdrant_filter": None
    #             }
    
    #     except Exception as e:
    #         logger.error(f"LLM query processing failed: {str(e)}")
    #         return {
    #             "parameters": {},
    #             "expanded_query": query,
    #             "query_representations": [query],
    #             "qdrant_filter": None
    #         }
    
    def enhanced_search(self, collection_name: str, query: str, 
                       limit: int = 10, filters: Optional[Dict] = None) -> Optional[List[Any]]:
        """Advanced hybrid search with LLM-driven query processing"""
        try:
            if not self.qdrant_client.collection_exists(collection_name):
                logger.error(f"Collection '{collection_name}' not found")
                return None
            
            # Use LLM to process query
            query_info = self._process_query_with_llm(query)
            logger.info(f"LLM processed query: {query_info}")
            
            # Combine any provided filters with LLM-extracted parameters
            params = query_info["parameters"]
            if filters:
                params.update(filters)
            
            # Get expanded query and representations
            expanded_query = query_info["expanded_query"]
            query_representations = query_info["query_representations"]
            
            # Build Qdrant filter from LLM output
            qdrant_filter = None
            if query_info["qdrant_filter"] and query_info["qdrant_filter"].get("must"):
                conditions = []
                for condition in query_info["qdrant_filter"]["must"]:
                    try:
                        if "match" in condition:
                            conditions.append(FieldCondition(
                                key=condition["key"],
                                match=MatchValue(value=condition["match"]["value"])
                            ))
                        elif "range" in condition:
                            range_data = condition["range"]
                            
                            # Handle different field types appropriately
                            field_key = condition["key"]
                            
                            if "date" in field_key.lower():
                                # For date fields, use string matching instead of range
                                # Convert date range to individual date matches
                                if "gte" in range_data and "lte" in range_data:
                                    start_date = range_data["gte"]
                                    end_date = range_data["lte"]
                                    
                                    # For now, let's use match on the start date
                                    # You might want to implement more sophisticated date filtering
                                    conditions.append(FieldCondition(
                                        key=field_key,
                                        match=MatchValue(value=start_date)
                                    ))
                                    logger.info(f"Converted date range to match filter: {field_key} = {start_date}")
                            else:
                                # For numeric fields, ensure values are numeric
                                numeric_range = {}
                                for key, value in range_data.items():
                                    try:
                                        numeric_range[key] = float(value)
                                    except (ValueError, TypeError):
                                        logger.warning(f"Could not convert {value} to float for range filter")
                                        continue
                                
                                if numeric_range:
                                    conditions.append(FieldCondition(
                                        key=field_key,
                                        range=Range(**numeric_range)
                                    ))
                                    logger.info(f"Added numeric range filter: {field_key} = {numeric_range}")
                                
                    except Exception as e:
                        logger.error(f"Failed to process filter condition {condition}: {str(e)}")
                        continue
                
                if conditions:
                    qdrant_filter = Filter(must=conditions)
                    logger.info(f"Built Qdrant filter with {len(conditions)} conditions")
                else:
                    logger.warning("No valid filter conditions could be built")
                
            all_results = []
            
            # Search for each representation
            for q_rep in query_representations:
                try:
                    query_embedding = self.embedding_model.encode(q_rep).tolist()
                    
                    search_result = self.qdrant_client.search(
                        collection_name=collection_name,
                        query_vector=query_embedding,
                        query_filter=qdrant_filter,
                        with_payload=True,
                        limit=limit * 2,
                        score_threshold=self.config.min_score_threshold
                    )
                    all_results.extend(search_result)
                except Exception as e:
                    logger.error(f"Search failed for '{q_rep}': {str(e)}")
            
            # Deduplicate results
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
            
            collection_info = self.qdrant_client.get_collection(collection_name)
            
            return {
                "name": collection_name,
                "status": collection_info.status.name,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.name,
                "payload_schema": {
                    field: schema.dict() 
                    for field, schema in collection_info.payload_schema.items()
                } if collection_info.payload_schema else {}
            }
        except Exception as e:
            logger.error(f"Failed to get collection info for '{collection_name}': {str(e)}")
            return None
    
    def _deduplicate_results(self, results: List[Any]) -> List[Any]:
        """Deduplicate results based on order ID while keeping highest score"""
        unique_results = {}
        for result in results:
            try:
                order_id = result.payload['metadata']['order_id']
                if order_id not in unique_results or result.score > unique_results[order_id].score:
                    unique_results[order_id] = result
            except KeyError:
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
                timeout=30
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
            search_results = self.enhanced_search(
                collection_name=collection_name,
                query=question,
                limit=search_limit or self.config.default_search_limit
            )
            
            if not search_results:
                result["error"] = "No relevant documents found"
                return result
            
            context_texts = []
            for res in search_results:
                try:
                    context_texts.append(res.payload['text'])
                except KeyError:
                    logger.warning("Search result missing text payload")
            
            context = "\n\n".join(context_texts)
            
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
    """
    documents = []
    metadata_list = []
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%m-%d-%y')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df.fillna('', inplace=True)
    
    for _, row in df.iterrows():
        try:
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
        config = RAGConfig(
            qdrant_url=os.getenv('QDRANT_URL'),
            qdrant_api_key=os.getenv('QDRANT_API_KEY'),
            groq_api_key=os.getenv('GROQ_API_KEY')
        )
        
        rag = RAGSystem(config)
        logger.info("RAG system initialized")
        
        # try:
        #     sales_df = pd.read_csv('data/final/final_sales_data.csv')
        #     logger.info(f"Loaded sales data: {len(sales_df)} records")
        # except Exception as e:
        #     logger.error(f"Data loading failed: {str(e)}")
        #     return
        
        # documents, metadata = preprocess_sales_data(sales_df)
        collection_name = "sales_data_v3"
        
        # if not rag.add_documents(collection_name, documents, metadata):
        #     logger.error("Document ingestion failed")
        #     return
        
        queries = [
            "give me the count of all canceled orders in bangalore between 04-20-22 to 04-30-22",
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
        
        collections = rag.list_collections()
        if collections:
            print(f"\nAvailable collections: {collections}")
        
    except Exception as e:
        logger.exception("Fatal error in main execution")

if __name__ == "__main__":
    main()