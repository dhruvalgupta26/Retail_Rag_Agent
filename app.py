import streamlit as st
import requests
import pandas as pd
import time
import json
from datetime import datetime
import plotly.express as px

# Configuration
FASTAPI_URL = "http://localhost:8001"  # Update with your FastAPI URL
DEFAULT_COLLECTION = "sales_data_v2"

# Initialize session state
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "current_query" not in st.session_state:
        st.session_state.current_query = None
    if "current_response" not in st.session_state:
        st.session_state.current_response = None

# API Helper Functions
def get_collections():
    try:
        response = requests.get(f"{FASTAPI_URL}/collections")
        if response.status_code == 200:
            return response.json().get("collections", [])
    except requests.exceptions.RequestException:
        return []
    return []

def get_collection_info(collection_name):
    try:
        response = requests.get(f"{FASTAPI_URL}/collection/{collection_name}")
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        return None
    return None

def query_rag(question, collection_name, search_limit=10):
    payload = {
        "question": question,
        "collection_name": collection_name,
        "search_limit": search_limit
    }
    try:
        response = requests.post(f"{FASTAPI_URL}/query", json=payload)
        return response.json()
    except requests.exceptions.RequestException:
        return None

def ingest_data(file_path, collection_name):
    payload = {
        "file_path": file_path,
        "collection_name": collection_name
    }
    try:
        response = requests.post(f"{FASTAPI_URL}/ingest", json=payload)
        return response.json()
    except requests.exceptions.RequestException:
        return None

# UI Components
def render_sidebar():
    with st.sidebar:
        st.title("RAG Agent Configuration")
        st.divider()
        
        # Collection selection
        collections = get_collections()
        selected_collection = st.selectbox(
            "Select Collection",
            collections,
            index=collections.index(DEFAULT_COLLECTION) if DEFAULT_COLLECTION in collections else 0
        )
        
        # Collection info
        if selected_collection:
            info = get_collection_info(selected_collection)
            if info and "error" not in info:
                st.subheader("Collection Info")
                st.metric("Documents", info.get("points_count", "N/A"))
                st.metric("Vector Size", info.get("vector_size", "N/A"))
                st.metric("Distance Metric", info.get("distance_metric", "N/A"))
        
        st.divider()
        
        # Data ingestion
        st.subheader("Data Ingestion")
        uploaded_file = st.file_uploader("Upload Sales CSV", type="csv")
        if uploaded_file is not None:
            # Save to temp file
            file_path = f"./data/uploaded_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("Ingest Data", use_container_width=True):
                with st.spinner("Ingesting data..."):
                    result = ingest_data(file_path, selected_collection)
                    if result and result.get("success"):
                        st.success(f"Ingested {result.get('document_count')} documents!")
                    else:
                        st.error("Ingestion failed: " + result.get("message", "Unknown error"))
        
        st.divider()
        
        # System info
        st.subheader("System Status")
        st.info("RAG system connected" if collections else "⚠️ RAG system not connected")
        st.caption(f"API: {FASTAPI_URL}")
        
        return selected_collection

def render_chat(collection_name):
    st.title("Sales Data RAG Agent")
    
    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask about your sales data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.current_query = prompt
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get RAG response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing sales data..."):
                response = query_rag(
                    question=prompt,
                    collection_name=collection_name,
                    search_limit=15
                )
            
            st.session_state.current_response = response
            
            if response and response.get("success"):
                # Add to conversation history
                st.session_state.conversation_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "query": prompt,
                    "response": response["answer"],
                    "context": response["context"],
                    "metadata": [res["metadata"] for res in response["search_results"]]
                })
                
                # Display assistant response
                st.markdown(response["answer"])
                st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            else:
                error = response.get("error", "Failed to get response from RAG system") if response else "Connection error"
                st.error(f"Error: {error}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {error}"})

def render_query_details():
    if st.session_state.current_response and st.session_state.current_response.get("success"):
        response = st.session_state.current_response
        query = st.session_state.current_query
        
        st.divider()
        st.subheader("Query Analysis")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Response Time", f"{response['execution_time']:.2f}s")
        col2.metric("Context Chunks", len(response['context']))
        col3.metric("LLM Model", "llama3-70b-8192")
        
        # Tabs for detailed view
        tab1, tab2, tab3 = st.tabs(["Retrieved Context", "Search Results", "Conversation History"])
        
        with tab1:
            st.subheader("Retrieved Context")
            for i, context in enumerate(response['context']):
                with st.expander(f"Context Chunk {i+1}", expanded=i==0):
                    st.markdown(f"```\n{context}\n```")
        
        with tab2:
            st.subheader("Search Results & Metadata")
            results = response['search_results']
            
            # Create dataframe for visualization
            result_data = []
            for res in results:
                meta = res['metadata']
                result_data.append({
                    "Score": res['score'],
                    "Order ID": meta.get('order_id', ''),
                    "Date": meta.get('date', ''),
                    "Amount": meta.get('amount', 0),
                    "City": meta.get('ship_city', ''),
                    "Status": meta.get('status', '')
                })
            
            df = pd.DataFrame(result_data)
            
            if not df.empty:
                # Summary stats
                st.dataframe(df, use_container_width=True)
                
                # Visualizations
                st.subheader("Result Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    if 'Status' in df.columns:
                        status_counts = df['Status'].value_counts()
                        fig1 = px.pie(
                            status_counts, 
                            values=status_counts.values,
                            names=status_counts.index,
                            title="Result Status Distribution"
                        )
                        st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    if 'Amount' in df.columns:
                        fig2 = px.histogram(
                            df, 
                            x='Amount',
                            nbins=20,
                            title="Amount Distribution in Results"
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                    date_counts = df.groupby(df['Date'].dt.date).size()
                    fig3 = px.bar(
                        date_counts,
                        x=date_counts.index,
                        y=date_counts.values,
                        title="Result Distribution by Date"
                    )
                    fig3.update_layout(xaxis_title="Date", yaxis_title="Count")
                    st.plotly_chart(fig3, use_container_width=True)
            else:
                st.warning("No metadata available for search results")
        
        with tab3:
            st.subheader("Conversation History")
            if st.session_state.conversation_history:
                # Show conversation timeline
                for conv in reversed(st.session_state.conversation_history):
                    with st.expander(f"{conv['query']}", expanded=False):
                        st.markdown(f"**Query:** {conv['query']}")
                        st.markdown(f"**Response:** {conv['response']}")
                        st.caption(f"Timestamp: {conv['timestamp']}")
                
                # Export option
                if st.button("Export Conversation History"):
                    json_data = json.dumps(st.session_state.conversation_history, indent=2)
                    st.download_button(
                        label="Download as JSON",
                        data=json_data,
                        file_name="rag_conversation_history.json",
                        mime="application/json"
                    )
            else:
                st.info("No conversation history yet")

# Main App
def main():
    st.set_page_config(
        page_title="Sales Data RAG Agent",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for styling
    # st.markdown("""
    # <style>
    #     .stApp { background-color: #f0f2f6; }
    #     .stChatInput { bottom: 20px; }
    #     .stExpander .st-emotion-cache-1hynsf2 { background-color: white; }
    #     .block-container { padding-top: 2rem; }
    #     .stDataFrame { border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    #     .metric-card { background-color: white; border-radius: 10px; padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    # </style>
    # """, unsafe_allow_html=True)
    
    init_session_state()
    
    # Render UI components
    collection_name = render_sidebar()
    render_chat(collection_name)
    
    # Show query details if available
    if st.session_state.current_response:
        render_query_details()

if __name__ == "__main__":
    main()