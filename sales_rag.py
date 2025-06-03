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
        logging.FileHandler('rag_shoe_store.log'),
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
    """Production-ready RAG system for shoe store data"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_model = None
        self.qdrant_client = None
        self.groq_client = None
        
        self.prompt_template = """
        You are a shoe store sales analyst AI. Use the following context to answer the question accurately.

        Context: 
        {context}

        Question: {question}

        Instructions:
        1. Provide specific, data-driven answers based EXCLUSIVELY on the context
        2. For invoice data:
           - Include Invoice Numbers when available
           - Report exact amounts with currency (₹)
           - Specify dates in YYYY-MM-DD format
           - Include relevant product details (brand, category, size, color)
        3. For summary questions, calculate from context
        4. If answer can't be determined, say "I couldn't find relevant data"
        5. NEVER invent invoice details or amounts

        Answer:
        """

        self.query_processing_prompt = """
        You are an expert in processing shoe store sales queries. Your task is to:
        1. Extract structured parameters from the query
        2. Generate expanded query versions
        3. Create multiple query representations
        4. Build Qdrant filter conditions

        Input Query: {query}

        Output JSON:
        ```json
        {{
            "parameters": {{
                "date": "YYYY-MM-DD",
                "date_range": ["YYYY-MM-DD", "YYYY-MM-DD"],
                "receipt_type": "receipt_type_value",
                "amount": float,
                "store_code": "store_code_value",
                "brand": "brand_name",
                "category": "category_name",
                "size": "size_value",
                "color": "color_value",
                "invoice_no": "invoice_no_value"
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
        For dates, convert to YYYY-MM-DD format
        For date ranges (e.g., "last week"), provide start and end dates
        For date ranges, use a single range condition in the format: {{"key": "metadata.date", "range": {{"gte": "YYYY-MM-DD", "lte": "YYYY-MM-DD"}}}}
        Do NOT create separate match conditions for each date in the range
        For receipt_type, use: SALE, RETURN, EXCHANGE
        For amounts, extract numeric values (support ₹)
        For locations (store_code), normalize to uppercase
        For brands, categories, sizes, and colors, normalize to uppercase
        Generate at least 2 query representations
        Create Qdrant filter conditions in the exact format shown
        If a parameter is not found, omit it from the parameters dictionary
        Handle relative dates (today, yesterday, last X days/weeks/months)
        Ensure the JSON output is valid:
        Include commas between fields in objects
        Include commas between elements in arrays
        Do NOT include trailing commas
        Do NOT include newlines (\n) within string values; replace them with spaces
        Ensure all objects and arrays are properly closed
        Do NOT include any additional text or comments outside the JSON structure
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
                ("metadata.receipt_type", "keyword"),
                ("metadata.amount", "float"),
                ("metadata.store_code", "keyword"),
                ("metadata.brand", "keyword"),
                ("metadata.category", "keyword"),
                ("metadata.size", "keyword"),
                ("metadata.color", "keyword"),
                ("metadata.invoice_no", "keyword")
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
            logger.info(f"Input Query: {query}")
            filled_prompt = self.query_processing_prompt.format(query=query)
            
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
        
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        
        if content.endswith("```"):
            content = content[:-3]
        
        content = content.strip()
        
        json_start = -1
        json_end = -1
        
        for i, char in enumerate(content):
            if char == '{':
                json_start = i
                break
        
        if json_start == -1:
            return ""
        
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
    
    def enhanced_search(self, collection_name: str, query: str, 
                       limit: int = 10, filters: Optional[Dict] = None) -> Optional[List[Any]]:
        """Advanced hybrid search with LLM-driven query processing"""
        try:
            if not self.qdrant_client.collection_exists(collection_name):
                logger.error(f"Collection '{collection_name}' not found")
                return None
            
            query_info = self._process_query_with_llm(query)
            logger.info(f"LLM processed query: {query_info}")
            
            params = query_info["parameters"]
            if filters:
                params.update(filters)
            
            expanded_query = query_info["expanded_query"]
            query_representations = query_info["query_representations"]
            
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
                            field_key = condition["key"]
                            
                            if "date" in field_key.lower():
                                conditions.append(FieldCondition(
                                    key=field_key,
                                    range=Range(**range_data)
                                ))
                            else:
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
                    except Exception as e:
                        logger.error(f"Failed to process filter condition {condition}: {str(e)}")
                        continue
                
                if conditions:
                    qdrant_filter = Filter(must=conditions)
                    logger.info(f"Built Qdrant filter with {len(conditions)} conditions")
            
            all_results = []
            
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
            
            # unique_results = self._deduplicate_results(all_results)
            sorted_results = sorted(all_results, key=lambda x: x.score, reverse=True)
            
            logger.info(f"Retrieved {len(sorted_results)} unique results after deduplication")
            return sorted_results[:limit]
            
        except Exception as e:
            logger.exception("Enhanced search failed")
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
        """Deduplicate results based on invoice number while keeping highest score"""
        unique_results = {}
        for result in results:
            try:
                invoice_no = result.payload['metadata']['invoice_no']
                if invoice_no not in unique_results or result.score > unique_results[invoice_no].score:
                    unique_results[invoice_no] = result
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
                    {"role": "system", "content": "You are a precise shoe store sales analyst."},
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

def preprocess_shoe_data(df: pd.DataFrame) -> Tuple[List[str], List[Dict]]:
    """
    Preprocess shoe store data into documents and metadata
    """
    documents = []
    metadata_list = []
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    df['MRP'] = pd.to_numeric(df['MRP'], errors='coerce')
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df['NetAmount'] = pd.to_numeric(df['NetAmount'], errors='coerce')
    df['GrossAmount'] = pd.to_numeric(df['GrossAmount'], errors='coerce')
    df['Discount'] = pd.to_numeric(df['Discount'], errors='coerce')
    df.fillna('', inplace=True)
    
    for _, row in df.iterrows():
        try:
            doc_text = (
                f"Invoice {row['InvoiceNo']} ({row['ReceiptType']}) - "
                f"Store: {row['StoreCode']}, "
                f"Date: {row['Date']}, "
                f"Product: {row['Product_Desc']}, "
                f"Category: {row['P_Group']}, "
                f"Brand: {row['Brand']}, "
                f"Size: {row['Size']}, "
                f"Color: {row['Colour']}, "
                f"Quantity: {row['Quantity']}, "
                f"MRP: ₹{row['MRP']:.2f}, "
                f"Discount: ₹{row['Discount']:.2f}, "
                f"Net Amount: ₹{row['NetAmount']:.2f}, "
                f"Gross Amount: ₹{row['GrossAmount']:.2f}"
            )
            
            meta = {
                "invoice_no": row['InvoiceNo'],
                "store_code": row['StoreCode'].upper(),
                "date": row['Date'],
                "receipt_type": row['ReceiptType'].upper(),
                "brand": row['Brand'].upper(),
                "category": row['P_Group'].upper(),
                "size": row['Size'],
                "color": row['Colour'].upper(),
                "quantity": int(row['Quantity']) if not pd.isna(row['Quantity']) else 1,
                "mrp": float(row['MRP']) if not pd.isna(row['MRP']) else 0.0,
                "discount": float(row['Discount']) if not pd.isna(row['Discount']) else 0.0,
                "amount": float(row['Amount']) if not pd.isna(row['Amount']) else 0.0,
                "net_amount": float(row['NetAmount']) if not pd.isna(row['NetAmount']) else 0.0,
                "gross_amount": float(row['GrossAmount']) if not pd.isna(row['GrossAmount']) else 0.0,
                "ean": row['EAN'],
                "pos_item_id": row['POSItemID'],
                "hsn": row['HSN']
            }
            
            documents.append(doc_text)
            metadata_list.append(meta)
        except Exception as e:
            logger.error(f"Row processing failed: {str(e)}")
            continue
    
    logger.info(f"Preprocessed {len(documents)} shoe store records")
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
        
        try:
            sales_df = pd.read_csv('src/data_generation/shoe_store_dataset.csv')
            logger.info(f"Loaded shoe store data: {len(sales_df)} records")
        except Exception as e:
            logger.error(f"Data loading failed: {str(e)}")
            return
        
        # documents, metadata = preprocess_shoe_data(sales_df)
        collection_name = "shoe_store_data"
        
        # if not rag.add_documents(collection_name, documents, metadata):
        #     logger.error("Document ingestion failed")
        #     return
        
        queries = [
            "Give me the count of all RETURN invoices in store SH001 between 2024-04-01 to 2024-04-30",
            "Get all invoices from 2024-11-01",
            "Show me EXCHANGE invoices for Nike brand",
            "What is the status of invoice SH00000001?",
            "List SALE invoices over ₹2000 for Sneakers last month",
            "How many invoices were processed yesterday?",
            "Give me the Total number of invoices",
            "What is the total sales revenue",
            "Give me the list of top 5 brands by sales amount",
            "What is the average discount given on all invoices?",
            
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