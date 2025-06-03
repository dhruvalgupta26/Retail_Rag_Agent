import streamlit as st
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
import time

# Load environment variables
load_dotenv()

# Configure logging for Streamlit
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SQLAgentConfig:
    """Configuration for the SQL Agent"""
    groq_api_key: str
    db_url: Optional[str] = None
    csv_path: Optional[str] = None
    excel_path: Optional[str] = None
    # default_model: str = "llama3-70b-8192"
    default_model: str = "qwen-qwq-32b"
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
You are an expert MYSQL query generator for a shoe store database. Your task is to:
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

# Streamlit UI
def main():
    st.set_page_config(
        page_title="SQL Agent Chat",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 SQL Agent Chat Interface")
    st.markdown("Chat with your shoe store database using natural language!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data source selection
        data_source_type = st.selectbox(
            "Select Data Source",
            ["Database", "CSV File", "Excel File"]
        )
        
        # Groq API Key
        groq_api_key = st.text_input(
            "Groq API Key", 
            type="password", 
            value=os.getenv('GROQ_API_KEY', ''),
            help="Enter your Groq API key"
        )
        
        # Model selection
        model = st.selectbox(
            "Select Model",
            ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "qwen-qwq-32b", "gemma2-9b-it"],
            index=0
        )
        
        # Temperature
        temperature = st.slider(
            "Temperature", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.1, 
            step=0.1,
            help="Controls randomness in responses"
        )
        
        # Data source specific inputs
        config_data = {}
        
        if data_source_type == "Database":
            st.subheader("Database Configuration")
            db_host = st.text_input("Host", value=os.getenv('DB_HOST', 'localhost'))
            db_port = st.text_input("Port", value=os.getenv('DB_PORT', '3306'))
            db_name = st.text_input("Database Name", value=os.getenv('DB_NAME', ''))
            db_username = st.text_input("Username", value=os.getenv('DB_USERNAME', ''))
            db_password = st.text_input("Password", type="password", value=os.getenv('DB_PASSWORD', ''))
            
            if all([db_host, db_port, db_name, db_username, db_password]):
                password_encoded = quote(db_password)
                config_data['db_url'] = f'mysql+pymysql://{db_username}:{password_encoded}@{db_host}:{db_port}/{db_name}'
        
        elif data_source_type == "CSV File":
            uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
            if uploaded_file:
                # Save uploaded file temporarily
                with open("temp_data.csv", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                config_data['csv_path'] = "temp_data.csv"
        
        elif data_source_type == "Excel File":
            uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
            if uploaded_file:
                # Save uploaded file temporarily
                with open("temp_data.xlsx", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                config_data['excel_path'] = "temp_data.xlsx"
        
        # Initialize agent button
        initialize_agent = st.button("🚀 Initialize Agent", type="primary")
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'agent_initialized' not in st.session_state:
        st.session_state.agent_initialized = False
    
    # Initialize agent
    if initialize_agent and groq_api_key:
        try:
            with st.spinner("Initializing SQL Agent..."):
                config = SQLAgentConfig(
                    groq_api_key=groq_api_key,
                    default_model=model,
                    temperature=temperature,
                    **config_data
                )
                st.session_state.agent = SQLAgent(config)
                st.session_state.agent_initialized = True
                st.success("✅ SQL Agent initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize agent: {str(e)}")
            st.session_state.agent_initialized = False
    
    # Main chat interface
    if st.session_state.agent_initialized and st.session_state.agent:
        # Display schema information
        with st.expander("📋 Database Schema", expanded=False):
            st.code(st.session_state.agent.schema, language="text")
        
        # Sample queries
        st.subheader("💡 Sample Queries")
        sample_queries = [
            "Give me the count of all RETURN invoices in store SH001 between 2024-04-01 to 2024-04-30",
            "What is the total sales revenue?",
            "Show me the top 5 brands by sales amount",
            "How many invoices were processed yesterday?",
            "What is the average discount given on all invoices?"
        ]
        
        cols = st.columns(len(sample_queries))
        for i, query in enumerate(sample_queries):
            if cols[i].button(f"Query {i+1}", help=query, key=f"sample_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": query})
                
                with st.spinner("Processing query..."):
                    result = st.session_state.agent.query(query)
                    st.session_state.chat_history.append({"role": "assistant", "content": result})
        
        # Chat input
        st.subheader("💬 Chat")
        user_query = st.text_input(
            "Ask a question about your data:", 
            placeholder="e.g., How many sales did we have last month?",
            key="user_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            send_button = st.button("Send", type="primary")
        with col2:
            clear_button = st.button("Clear History")
        
        if clear_button:
            st.session_state.chat_history = []
            st.rerun()
        
        if send_button and user_query:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Process query
            with st.spinner("🤔 Thinking..."):
                result = st.session_state.agent.query(user_query)
                st.session_state.chat_history.append({"role": "assistant", "content": result})
            
            # Clear input and rerun
            st.rerun()
        
        # Display chat history
        st.subheader("📝 Chat History")
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                result = message["content"]
                with st.chat_message("assistant"):
                    if result["success"]:
                        st.write(f"**Answer:** {result['answer']}")
                        
                        # Show SQL query in expander
                        with st.expander("🔍 SQL Query", expanded=False):
                            st.code(result["sql_query"], language="sql")
                        
                        # Show results table if available
                        if result["query_result"] is not None and not result["query_result"].empty:
                            with st.expander("📊 Query Results", expanded=False):
                                st.dataframe(result["query_result"])
                        
                        # Show execution time
                        st.caption(f"⏱️ Execution time: {result['execution_time']:.2f}s")
                    else:
                        st.error(f"❌ Error: {result['error']}")
    
    else:
        st.info("👆 Please configure your data source and initialize the agent in the sidebar to start chatting!")
        
        # Show example configuration
        st.subheader("🚀 Getting Started")
        st.markdown("""
        1. **Configure your data source** in the sidebar:
           - For Database: Enter your MySQL/PostgreSQL connection details
           - For CSV/Excel: Upload your data file
        
        2. **Enter your Groq API key** (get one from [Groq Console](https://console.groq.com))
        
        3. **Click "Initialize Agent"** to connect to your data
        
        4. **Start asking questions** in natural language!
        
        ### Example Questions:
        - "How many sales did we have last month?"
        - "What are our top-selling products?"
        - "Show me all returns for Nike brand"
        - "What's the average order value?"
        """)

if __name__ == "__main__":
    main()