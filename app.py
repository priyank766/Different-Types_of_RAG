import streamlit as st
import os
from langgraph_workflow.corrective_rag import get_corrective_rag_app
from langgraph_workflow.agentic_rag import get_agentic_rag_app
from langgraph_workflow.adaptive_rag import get_adaptive_rag_app
from langgraph_workflow.hybrid_rag import get_hybrid_rag_app
from langchain_core.messages import HumanMessage, AIMessage

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="RAG Explorer",
    layout="wide",
)

# --- Custom CSS ---
st.markdown(
    """
    <style>
    /* General background */
    body {
        background-color: #f8fafc;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a, #1e293b);
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
        color: #e2e8f0 !important;
    }
    .rag-tip {
        background: #1e293b;
        padding: 12px;
        border-radius: 8px;
        color: #f1f5f9;
        margin-top: 8px;
        font-size: 0.9rem;
    }
    
    /* Chat bubbles */
    .stChatMessage[data-testid="stChatMessage-human"] {
        background: #2563eb20;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }
    .stChatMessage[data-testid="stChatMessage-ai"] {
        background: #10b98120;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 8px;
    }

    /* Expander for snippets */
    .streamlit-expanderHeader {
        font-weight: 600;
        color: #334155;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# --- Sidebar: API Key and RAG Type ---
with st.sidebar:
    st.header("⚙️ Configuration")
    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Get your API key from Google AI Studio.",
    )
    if not gemini_api_key:
        st.warning("Please enter your Gemini API key.")
        #os.environ["GOOGLE_API_KEY"] = ""
    else:
        pass
        #os.environ["GOOGLE_API_KEY"] = gemini_api_key

    # Removed PDF selection as RAGs now handle multiple PDFs internally

    st.subheader("🔍 Choose a RAG Type")
    rag_type = st.selectbox(
        "RAG Variant", ["Corrective", "Agentic", "Adaptive", "Hybrid"]
    )

    # Short structured tips
    if rag_type == "Corrective":
        st.markdown(
            '<div class="rag-tip"><b>Corrective RAG</b><br>Runs multiple passes where the LLM validates its generated answer against retrieved documents. If contradictions or unsupported claims are found, it attempts to correct or refine the answer. Great for enhancing factual precision and reducing hallucinations.</div>',
            unsafe_allow_html=True,
        )
    elif rag_type == "Agentic":
        st.markdown(
            '<div class="rag-tip"><b>Agentic RAG</b><br>Uses the LLM as an agent capable of planning, reasoning, and calling external tools (like a retriever) based on the query. Best for multi-step logic, dynamic decisions, and sophisticated problem-solving.</div>',
            unsafe_allow_html=True,
        )
    elif rag_type == "Adaptive":
        st.markdown(
            '<div class="rag-tip"><b>Adaptive RAG</b><br>Adapts its retrieval strategy or answer generation based on factors like chat history, user preferences, or the perceived complexity/type of the query. Useful for providing more personalized and context-aware answers.</div>',
            unsafe_allow_html=True,
        )
    # elif rag_type == "Hybrid":
    #     st.markdown(
    #         '<div class="rag-tip"><b>Hybrid RAG</b><br>Combines adaptiveness, validation, and agentic tool use for complex workflows. It can dynamically switch between strategies, offering a robust and flexible solution for diverse information retrieval tasks.</div>',
    #         unsafe_allow_html=True,
    #     )


# --- Main Chat Interface ---
col1, col2, col3 = st.columns([0.3, 3, 0.3])
with col2:
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.subheader(f"💬 {rag_type} RAG Chat")

    # Init chat history
    if "chat_histories" not in st.session_state:
        st.session_state.chat_histories = {
            "Corrective": [],
            "Agentic": [],
            "Adaptive": []
            # "Hybrid": [],
        }
    
    # Get current RAG type's messages
    current_messages = st.session_state.chat_histories[rag_type]

    # Display past messages
    for message in current_messages:
        with st.chat_message(message.type):
            st.markdown(message.content)

    # Handle new input
    if query := st.chat_input("Ask Mentioned PDFs or AIML/LLM concepts..."):
        if not gemini_api_key:
            st.error("Enter your Gemini API key first.")
        else:
            current_messages.append(HumanMessage(content=query, type="human"))
            with st.chat_message("human"):
                st.markdown(query)

            with st.spinner("Thinking..."):
                rag_apps = {
                    "Corrective": get_corrective_rag_app,
                    "Agentic": get_agentic_rag_app,
                    "Adaptive": get_adaptive_rag_app
                    # "Hybrid": get_hybrid_rag_app,
                }

                if rag_type not in st.session_state or not st.session_state.get(
                    rag_type
                ):
                    try:
                        st.session_state[rag_type] = rag_apps[rag_type](gemini_api_key)
                    except Exception as e:
                        st.error(f"Failed to load {rag_type} RAG: {e}")
                        st.stop()

                app = st.session_state[rag_type]

                def get_state(query, chat_history, rag_type):
                    if rag_type in ["Agentic"]:
                        return {"messages": [HumanMessage(content=query)]}
                    elif rag_type == "Adaptive":
                        return {"query": query, "chat_history": chat_history}
                    else:
                        return {"query": query}

                initial_state = get_state(query, current_messages, rag_type)

                try:
                    result = app.invoke(initial_state)

                    answer = "No answer."
                    if isinstance(result, dict) and result.get("answer"):
                        answer = result["answer"]
                    elif isinstance(result, dict) and isinstance(
                        result.get("messages"), list
                    ):
                        last = result["messages"][-1]
                        if hasattr(last, "content"):
                            answer = last.content
                        elif isinstance(last, dict) and last.get("content"):
                            answer = last["content"]
                    elif isinstance(result, str):
                        answer = result

                    current_messages.append(
                        AIMessage(content=answer, type="ai")
                    )
                    with st.chat_message("ai"):
                        st.markdown(answer)

                    st.session_state.context = result.get("context", [])
                except Exception as e:
                    st.error(f"Error: {e}")
                    current_messages.append(
                        AIMessage(content=f"Error: {e}", type="ai")
                    )
                    with st.chat_message("ai"):
                        st.markdown(f"Error: {e}")


