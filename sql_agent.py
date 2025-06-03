import os
import logging
import pandas as pd
import sqlalchemy
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import quote
from dataclasses import dataclass
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sql_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SQLAgentConfig:
    """Configuration for the SQL Agent"""
    groq_api_key: str
    db_url: Optional[str] = None  # SQL database URL (e.g., 'sqlite:///data.db')
    csv_path: Optional[str] = None  # Path to CSV file
    excel_path: Optional[str] = None  # Path to Excel file
    default_model: str = "llama3-70b-8192"
    temperature: float = 0.1
    max_tokens: int = 1024

@dataclass
class AgentState:
    """State for the LangGraph workflow"""
    question: str
    sql_query: Optional[str] = None
    query_result: Optional[pd.DataFrame] = None
    answer: Optional[str] = None
    error: Optional[str] = None
    data_source: Optional[str] = None

class SQLAgent:
    """LangGraph-based SQL Agent for question answering"""
    
    def __init__(self, config: SQLAgentConfig):
        self.config = config
        self.groq_client = Groq(api_key=config.groq_api_key)
        self.engine = None
        self.data_source = None
        self.schema = None
        
        # Initialize data source
        self._initialize_data_source()
        
        # Define prompts - Using string templates for direct formatting
        self.sql_system_prompt = """
            You are an expert SQL query generator for a shoe store database. Your task is to:
            1. Generate a valid SQL query based on the user's natural language question.
            2. Use the provided schema to ensure the query is correct.
            3. Handle aggregations (count, sum, average, top N), joins, and filters appropriately.
            4. Return only the SQL query as a string, no explanations or markdown.

            Schema:
            {schema}

            Instructions:
            - Use table and column names exactly as in the schema.
            - For dates, use 'YYYY-MM-DD' format.
            - For relative dates (e.g., 'last month'), assume today is {today}.
            - Normalize categorical values (e.g., brand, category) to uppercase.
            - Handle joins if multiple tables are needed (e.g., invoices, products, stores).
            - For aggregations, use appropriate SQL functions (COUNT, SUM, AVG).
            - For top N queries, use ORDER BY and LIMIT.
            - If the question is ambiguous, make reasonable assumptions and log them.
            - If no relevant data can be queried, return an empty query.
            """
        
        self.answer_system_prompt = """
            You are a shoe store sales analyst AI. Use the query results to answer the question in a human-readable format.

            Query Results:
            {results}

            Question: {question}

            Instructions:
            - Provide a clear, concise answer based on the query results.
            - Include specific details (e.g., InvoiceNo, dates in YYYY-MM-DD, amounts in ₹).
            - For aggregations, format numbers appropriately (e.g., ₹1000.00).
            - If results are empty, say: "I couldn't find relevant data."
            - Do not invent data not present in the results.
            """
        
        # Build LangGraph workflow
        self.graph = self._build_graph()
    
    def _initialize_data_source(self):
        """Initialize the data source and load schema"""
        try:
            if self.config.db_url:
                self.engine = sqlalchemy.create_engine(self.config.db_url)
                self.data_source = "sql_db"
                self.schema = self._get_db_schema()
                logger.info(f"Connected to SQL database: {self.config.db_url}")
            elif self.config.csv_path:
                self.data_source = "csv"
                self.schema = self._get_csv_schema(self.config.csv_path)
                logger.info(f"Loaded CSV: {self.config.csv_path}")
            elif self.config.excel_path:
                self.data_source = "excel"
                self.schema = self._get_excel_schema(self.config.excel_path)
                logger.info(f"Loaded Excel: {self.config.excel_path}")
            else:
                raise ValueError("No valid data source provided")
        except Exception as e:
            logger.critical(f"Data source initialization failed: {str(e)}")
            raise RuntimeError("Failed to initialize data source") from e
    
    def _get_db_schema(self) -> str:
        """Get schema for SQL database"""
        try:
            inspector = sqlalchemy.inspect(self.engine)
            schema_desc = []
            for table_name in inspector.get_table_names():
                columns = inspector.get_columns(table_name)
                schema_desc.append(f"Table: {table_name}")
                schema_desc.append("Columns:")
                for col in columns:
                    schema_desc.append(f"  - {col['name']} ({col['type']})")
            return "\n".join(schema_desc)
        except Exception as e:
            logger.error(f"Failed to get database schema: {str(e)}")
            return ""
    
    def _get_csv_schema(self, path: str) -> str:
        """Get schema for CSV file"""
        try:
            df = pd.read_csv(path, nrows=1)
            schema_desc = ["Table: data"]
            schema_desc.append("Columns:")
            for col, dtype in df.dtypes.items():
                schema_desc.append(f"  - {col} ({dtype})")
            return "\n".join(schema_desc)
        except Exception as e:
            logger.error(f"Failed to get CSV schema: {str(e)}")
            return ""
    
    def _get_excel_schema(self, path: str) -> str:
        """Get schema for Excel file"""
        try:
            df = pd.read_excel(path, nrows=1)
            schema_desc = ["Table: data"]
            schema_desc.append("Columns:")
            for col, dtype in df.dtypes.items():
                schema_desc.append(f"  - {col} ({dtype})")
            return "\n".join(schema_desc)
        except Exception as e:
            logger.error(f"Failed to get Excel schema: {str(e)}")
            return ""
    
    def _generate_sql_query(self, state: AgentState) -> AgentState:
        """Generate SQL query from the question"""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Format the system prompt with schema and today's date
            system_content = self.sql_system_prompt.format(
                schema=self.schema,
                today=today
            )
            
            logger.info(f"Generating SQL query for question: {state.question}")
            logger.debug(f"System prompt: {system_content}")
            
            response = self.groq_client.chat.completions.create(
                model=self.config.default_model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": f"Question: {state.question}"}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=30
            )
            
            state.sql_query = response.choices[0].message.content.strip()
            logger.info(f"Generated SQL query: {state.sql_query}")
            return state
            
        except Exception as e:
            logger.error(f"SQL query generation failed: {str(e)}")
            state.error = f"Failed to generate SQL query: {str(e)}"
            return state
    
    def _execute_query(self, state: AgentState) -> AgentState:
        """Execute the SQL query on the data source"""
        try:
            if not state.sql_query:
                state.error = "No SQL query to execute"
                return state
            
            if self.data_source == "sql_db":
                state.query_result = pd.read_sql(state.sql_query, self.engine)
            elif self.data_source == "csv":
                df = pd.read_csv(self.config.csv_path)
                state.query_result = self._execute_sql_on_df(df, state.sql_query)
            elif self.data_source == "excel":
                df = pd.read_excel(self.config.excel_path)
                state.query_result = self._execute_sql_on_df(df, state.sql_query)
            
            logger.info(f"Query executed successfully, returned {len(state.query_result)} rows")
            return state
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}")
            state.error = f"Query execution failed: {str(e)}"
            return state
    
    def _execute_sql_on_df(self, df: pd.DataFrame, query: str) -> pd.DataFrame:
        """Execute SQL query on a DataFrame using pandasql"""
        try:
            from pandasql import sqldf
            # Create a lambda function to handle DataFrame as a table named 'data'
            pysqldf = lambda q: sqldf(q, {"data": df})
            return pysqldf(query)
        except Exception as e:
            logger.error(f"Failed to execute SQL on DataFrame: {str(e)}")
            raise
    
    def _generate_answer(self, state: AgentState) -> AgentState:
        """Generate human-readable answer from query results"""
        try:
            if state.error or state.query_result is None:
                state.answer = "I couldn't find relevant data due to an error."
                return state
                     
            results_str = state.query_result.to_string(index=False)
            if state.query_result.empty:
                state.answer = "I couldn't find relevant data."
                return state
                     
            # Format the system prompt with results and question
            system_content = self.answer_system_prompt.format(
                results=results_str,
                question=state.question
            )
            
            response = self.groq_client.chat.completions.create(
                model=self.config.default_model,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": "Answer:"}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                timeout=30
            )
            
            state.answer = response.choices[0].message.content.strip()
            logger.info(f"Generated answer: {state.answer}")
            return state
            
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            state.error = f"Answer generation failed: {str(e)}"
            state.answer = "I couldn't find relevant data due to an error."
            return state
            
        except Exception as e:
            logger.error(f"SQL query generation failed: {str(e)}")
            state.error = f"Failed to generate SQL query: {str(e)}"
            return state
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        graph = StateGraph(AgentState)
        
        # Add nodes
        graph.add_node("generate_sql", self._generate_sql_query)
        graph.add_node("execute_query", self._execute_query)
        graph.add_node("generate_answer", self._generate_answer)
        
        # Define edges
        graph.add_edge("generate_sql", "execute_query")
        graph.add_edge("execute_query", "generate_answer")
        graph.add_edge("generate_answer", END)
        
        # Set entry point
        graph.set_entry_point("generate_sql")
        
        return graph.compile()
    
    def query(self, question: str) -> Dict[str, Any]:
        """Execute the query pipeline"""
        start_time = datetime.now()
        state = AgentState(question=question, data_source=self.data_source)
        
        try:
            result = self.graph.invoke(state)
            result["execution_time"] = (datetime.now() - start_time).total_seconds()
            result["success"] = result["error"] is None
            logger.info(f"Query processed: {question}, Success: {result['success']}, Time: {result['execution_time']:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Query processing failed: {str(e)}")
            return {
                "success": False,
                "question": question,
                "sql_query": None,
                "query_result": None,
                "answer": "Query processing failed due to an error.",
                "error": str(e),
                "execution_time": (datetime.now() - start_time).total_seconds(),
                "data_source": self.data_source
            }
def main():
    """Main execution with example queries"""
    try:
        # Extract environment variables first
        username = os.getenv('DB_USERNAME', 'root')
        password = os.getenv('DB_PASSWORD', 'password')
        host = os.getenv('DB_HOST', 'localhost')
        port = os.getenv('DB_PORT', '3306')
        database_name = os.getenv('DB_NAME', 'mydatabase')
        
        # URL-encode the password to handle special characters
        password_encoded = quote(password)

        # Construct the DB URL
        db_url = f'mysql+pymysql://{username}:{password_encoded}@{host}:{port}/{database_name}'

        # Create the config
        config = SQLAgentConfig(
            groq_api_key=os.getenv('GROQ_API_KEY'),
            db_url=db_url,
            # csv_path="shoe_store_dataset.csv",
            # excel_path="shoe_store.xlsx"
        )

        agent = SQLAgent(config)
        logger.info("SQL Agent initialized")

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
            "What is the average discount given on all invoices?"
        ]

        for query in queries:
            print(f"\n{'='*50}")
            print(f"QUERY: {query}")
            result = agent.query(query)

            if result["success"]:
                print(f"\nSQL QUERY:\n{result['sql_query']}")
                print(f"\nRESULTS:\n{result['query_result'].to_string(index=False)}")
                print(f"\nANSWER ({result['execution_time']:.2f}s):\n{result['answer']}")
            else:
                print(f"\nERROR: {result['error']}")
            print(f"{'='*50}\n")

    except Exception as e:
        logger.exception(f"Fatal error in main execution: {str(e)}")

if __name__ == "__main__":
    main()