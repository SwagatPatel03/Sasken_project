# src/app.py

import streamlit as st
import requests
import json

# Page config
st.set_page_config(
    page_title="3GPP Change Detection QA",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-container {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .example-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e1e5e9;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main header
st.markdown("""
<div class="main-header">
    <h1>🔍 3GPP Change Detection QA System</h1>
    <p style="margin: 0; font-size: 1.1em; opacity: 0.9;">
        Ask intelligent questions about changes between 3GPP specification versions (Rel-15 vs Rel-16)
    </p>
</div>
""", unsafe_allow_html=True)

# API endpoint
API_URL = "http://localhost:8000"

def query_api(question: str, top_k: int = 5):
    """Query the FastAPI backend"""
    try:
        response = requests.post(
            f"{API_URL}/qa",
            json={"question": question, "top_k": top_k},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["answer"]
        else:
            return f"⚠️ **Error {response.status_code}**: {response.text}"
    except requests.exceptions.ConnectionError:
        return "🔌 **Connection Error**: Could not connect to the API server. Please ensure it's running on http://localhost:8000"
    except requests.exceptions.Timeout:
        return "⏱️ **Timeout Error**: The request took too long. Try simplifying your question."
    except Exception as e:
        return f"❌ **Unexpected Error**: {str(e)}"

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    # Chat interface
    st.markdown("### 💬 Chat Interface")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        if st.session_state.messages:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("👋 Welcome! Start by asking a question about 3GPP changes below.")

    # Chat input
    if prompt := st.chat_input("💭 Ask a question about 3GPP changes..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing specifications and searching for relevant changes..."):
                response = query_api(prompt, 5)
            
            st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

with col2:
    # Settings panel
    st.markdown("### ⚙️ Settings")
    
    with st.expander("🎛️ Advanced Options", expanded=False):
        top_k = st.slider(
            "Number of results to analyze",
            min_value=1,
            max_value=20,
            value=5,
            help="Higher values provide more comprehensive analysis but may take longer"
        )
    
    # Quick Actions
    st.markdown("### 🚀 Quick Actions")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col_b:
        if st.button("📊 New Session", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Example questions
    st.markdown("### 💡 Example Questions")
    
    example_questions = [
        "What changed in section 5.5?",
        "Summarize changes in section 8.2.7",
        "How many subsections are in section 4?",
        "What are the main security changes?",
        "List all modifications to authentication procedures",
        "What new features were added in Rel-16?",
        "Compare protocol changes between versions"
    ]
    
    for i, question in enumerate(example_questions):
        if st.button(f"📝 {question}", key=f"example_{i}", use_container_width=True):
            # Add to chat and get response
            st.session_state.messages.append({"role": "user", "content": question})
            
            with st.spinner("Processing example question..."):
                response = query_api(question, top_k if 'top_k' in locals() else 5)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# Alternative input section (below the main interface)
st.markdown("---")
st.markdown("### 📝 Alternative Input Method")

with st.expander("✍️ Detailed Question Form", expanded=False):
    with st.form("question_form", clear_on_submit=True):
        question = st.text_area(
            "Enter your detailed question:",
            placeholder="Type your question about 3GPP specification changes here...\n\nFor example:\n- What are the key differences in authentication procedures?\n- Summarize all changes in a specific section\n- List new security features",
            height=120,
            help="You can ask complex, multi-part questions here"
        )
        
        col_form1, col_form2, col_form3 = st.columns([1, 1, 2])
        
        with col_form1:
            submitted = st.form_submit_button("🚀 Ask Question", use_container_width=True)
        
        with col_form2:
            if st.form_submit_button("🔄 Reset Form", use_container_width=True):
                st.rerun()

        if submitted and question.strip():
            # Add to chat history
            st.session_state.messages.append({"role": "user", "content": question})
            
            # Get response
            with st.spinner("🔍 Processing your detailed question..."):
                response = query_api(question, top_k if 'top_k' in locals() else 5)
            
            # Add response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Rerun to show the new messages
            st.rerun()

# Statistics and info
if st.session_state.messages:
    st.markdown("---")
    col_stats1, col_stats2, col_stats3 = st.columns(3)
    
    with col_stats1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #667eea;">{len(st.session_state.messages)}</h3>
            <p style="margin: 0;">Total Messages</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stats2:
        user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #667eea;">{user_messages}</h3>
            <p style="margin: 0;">Questions Asked</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stats3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; color: #667eea;">Active</h3>
            <p style="margin: 0;">Session Status</p>
        </div>
        """, unsafe_allow_html=True)

# Footer with tips
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <h4 style="color: #667eea; margin-bottom: 0.5rem;">💡 Pro Tips</h4>
    <p style="margin: 0;">
        • Be specific about sections or features you're interested in<br>
        • Ask for summaries, comparisons, or detailed explanations<br>
        • Use the example questions to get started quickly
    </p>
</div>
""", unsafe_allow_html=True)