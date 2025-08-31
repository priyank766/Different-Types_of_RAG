import os
from typing import TypedDict, Annotated, Sequence, List
import operator

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document

INDEX_DIR = "indices"


# Define the state for the Agentic RAG workflow
class AgenticRagState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


def get_agentic_rag_app(api_key: str):
    """Creates and returns the Agentic RAG LangGraph app."""
    # Initialize the LLM with the provided API key
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", temperature=0, google_api_key=api_key
    )

    # Discover and load all available vector stores
    loaded_vectorstores = {}
    if os.path.exists(INDEX_DIR):
        for item in os.listdir(INDEX_DIR):
            item_path = os.path.join(INDEX_DIR, item)
            if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, "index.faiss")):
                try:
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    vectorstore = FAISS.load_local(
                        item_path, embeddings, allow_dangerous_deserialization=True
                    )
                    loaded_vectorstores[item] = vectorstore
                    print(f"Loaded vector store from: {item_path}")
                except Exception as e:
                    print(f"Error loading vector store from {item_path}: {e}")

    if not loaded_vectorstores:
        print("No vector stores found. Please run ingestion first.")
        # Define a dummy tool if no vector stores are found
        @tool(description="A dummy retriever for when no vector stores are found.")
        def dummy_multi_pdf_retriever(query: str, pdf_name: str = "any"):
            """A dummy retriever for when no vector stores are found."""
            return "No vector stores found. Please run ingestion first."
        tools = [dummy_multi_pdf_retriever]
    else:
        @tool(description="Retrieves relevant document chunks for a given query from a specified PDF's knowledge base. If pdf_name is 'any', it will query all available PDFs.")
        def multi_pdf_retriever(query: str, pdf_name: str = "any") -> List[str]:
            """Retrieves relevant document chunks for a given query from a specified PDF's knowledge base.
            If pdf_name is 'any', it will query all available PDFs.
            """
            
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

    # Define the nodes for the Agentic RAG workflow
    def call_agent(state):
        """Calls the agent to decide the next action."""
        print("---CALLING AGENT---")
        # Add a system message to guide the agent
        system_message = SystemMessage(content="""You are an expert assistant that can answer questions by retrieving information from specific PDF documents. 
        When asked a question, identify which PDF document is most relevant to the query. 
        If the user's question is general or doesn't specify a PDF, you can query all available PDFs by setting pdf_name to 'any'.
        Available PDFs: {available_pdfs}
        """.format(available_pdfs=list(loaded_vectorstores.keys())))
        
        messages = [system_message] + list(state["messages"])
        response = llm_with_tools.invoke(messages)
        return {"messages": state["messages"] + [response]}

    # Define the conditional edge
    def should_continue(state):
        """Decides whether to continue or end the workflow."""
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "continue"
        else:
            return "end"

    # Build the graph
    workflow = StateGraph(AgenticRagState)
    workflow.add_node("agent", call_agent)
    workflow.add_node("tools", tool_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
        },
    )
    workflow.add_edge("tools", "agent")

    # Compile the graph
    return workflow.compile()
