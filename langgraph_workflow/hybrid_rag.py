import os
import json
from typing import TypedDict, Annotated, Sequence, List
import operator

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

INDEX_DIR = "indices"


# Define the state for the Hybrid RAG workflow
class HybridRagState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    context: List[str]
    answer: str
    is_relevant: bool
    claims_supported: bool


def get_hybrid_rag_app(api_key: str):
    """Creates and returns the Hybrid RAG LangGraph app."""
    # Initialize the LLM with the provided API key
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", temperature=0, google_api_key=api_key
    )

    # Discover and load all available vector stores
    loaded_vectorstores = {}
    if os.path.exists(INDEX_DIR):
        for item in os.listdir(INDEX_DIR):
            item_path = os.path.join(INDEX_DIR, item)
            if os.path.isdir(item_path) and os.path.exists(
                os.path.join(item_path, "index.faiss")
            ):
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/embedding-001"
                    )
                    vectorstore = FAISS.load_local(
                        item_path, embeddings, allow_dangerous_deserialization=True
                    )
                    loaded_vectorstores[item] = vectorstore
                    print(f"Loaded vector store from: {item_path}")
                except Exception as e:
                    print(f"Error loading vector store from {item_path}: {e}")

    if not loaded_vectorstores:
        print("No vector stores found. Please run ingestion first.")

        @tool
        def dummy_multi_pdf_retriever(query: str, pdf_name: str = "any"):
            """A dummy retriever for when no vector stores are found."""
            return "No vector stores found. Please run ingestion first."

        tools = [dummy_multi_pdf_retriever]
    else:

        @tool
        def multi_pdf_retriever(query: str, pdf_name: str = "any") -> List[str]:
            """Retrieves relevant document chunks for a given query from a specified PDF's knowledge base.
            If pdf_name is 'any', it will query all available PDFs.
            Available PDFs: {available_pdfs}
            """.format(available_pdfs=list(loaded_vectorstores.keys()))

            retrieved_docs_content = []
            if pdf_name == "any":
                for vs_name, vs in loaded_vectorstores.items():
                    retriever = vs.as_retriever()
                    docs = retriever.invoke(query)
                    retrieved_docs_content.extend([d.page_content for d in docs])
            elif pdf_name in loaded_vectorstores:
                retriever = loaded_vectorstores[pdf_name].as_retriever()
                docs = retriever.invoke(query)
                retrieved_docs_content.extend([d.page_content for d in docs])
            else:
                return f"PDF '{pdf_name}' not found in available knowledge bases. Available PDFs: {list(loaded_vectorstores.keys())}"

            return retrieved_docs_content

        tools = [multi_pdf_retriever]

    tool_node = ToolNode(tools)
    llm_with_tools = llm.bind_tools(tools)

    # Define the nodes
    def call_agent_hybrid(state: HybridRagState):
        print("---CALLING AGENT (HYBRID)---")
        # Add a system message to guide the agent
        system_message = SystemMessage(
            content="""You are an expert assistant that can answer questions by retrieving information from specific PDF documents. 
        When asked a question, identify which PDF document is most relevant to the query. 
        If the user's question is general or doesn't specify a PDF, you can query all available PDFs by setting pdf_name to 'any'.
        Available PDFs: {available_pdfs}
        """.format(available_pdfs=list(loaded_vectorstores.keys()))
        )

        messages = [system_message] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        state["messages"] = state["messages"] + [response]
        return state

    def grade_documents_hybrid(state: HybridRagState):
        state["is_relevant"] = True
        return state

    def generate_hybrid(state: HybridRagState):
        state["answer"] = "This is a hybrid answer."
        return state

    def check_claims_hybrid(state: HybridRagState):
        state["claims_supported"] = True
        return state

    # Define conditional edges
    def should_continue_hybrid(state):
        if state["messages"][-1].tool_calls:
            return "tool"
        else:
            return "generate"

    def decide_to_finish_hybrid(state: HybridRagState):
        return END if state["claims_supported"] else "generate"

    # Build the graph
    workflow = StateGraph(HybridRagState)
    workflow.add_node("agent", call_agent_hybrid)
    workflow.add_node("tools", tool_node)
    workflow.add_node("grade_documents", grade_documents_hybrid)
    workflow.add_node("generate", generate_hybrid)
    workflow.add_node("check_claims", check_claims_hybrid)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", should_continue_hybrid, {"tool": "tools", "generate": "generate"}
    )
    workflow.add_edge("tools", "grade_documents")
    workflow.add_edge("grade_documents", "generate")
    workflow.add_edge("generate", "check_claims")
    workflow.add_conditional_edges(
        "check_claims", decide_to_finish_hybrid, {"generate": "generate", "end": END}
    )

    # Compile the graph
    return workflow.compile()
