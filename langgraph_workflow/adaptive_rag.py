import os
from typing import TypedDict, List
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document


INDEX_DIR = "indices"


# Define the state for the Adaptive RAG workflow
class AdaptiveRagState(TypedDict):
    query: str
    chat_history: List[BaseMessage]
    context: List[str]
    answer: str


def get_adaptive_rag_app(api_key: str):
    """Creates and returns the Adaptive RAG LangGraph app."""
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
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
                    vectorstore = FAISS.load_local(
                        item_path, embeddings, allow_dangerous_deserialization=True
                    )
                    loaded_vectorstores[item] = vectorstore
                    print(f"Loaded vector store from: {item_path}")
                except Exception as e:
                    print(f"Error loading vector store from {item_path}: {e}")

    # Create a combined retriever
    combined_retriever = None
    if loaded_vectorstores:
        def custom_combined_retriever(query: str) -> List[Document]:
            all_docs = []
            for vs_name, vs in loaded_vectorstores.items():
                retriever = vs.as_retriever()
                docs = retriever.invoke(query)
                all_docs.extend(docs)
            return all_docs
        combined_retriever = custom_combined_retriever
    else:
        print("No vector stores found. Combined retriever will be empty.")


    # Define the nodes for the Adaptive RAG workflow
    def retrieve_with_history(state: AdaptiveRagState):
        """Retrieve documents considering the chat history."""
        print("---RETRIEVE WITH HISTORY---")
        if combined_retriever is None:
            state["context"] = []
            return state

        # Create a new query that incorporates the chat history
        if state["chat_history"]:
            history = "\n".join([msg.content for msg in state["chat_history"]])
            new_query = f"Based on the following chat history, answer the question. History: {history}\n\nQuestion: {state['query']}"
        else:
            new_query = state["query"]

        # Use the combined retriever directly
        retrieved_docs_lc = combined_retriever(new_query) # Call the function
        retrieved_docs_content = [doc.page_content for doc in retrieved_docs_lc]
        
        state["context"] = retrieved_docs_content
        return state

    def generate_with_history(state: AdaptiveRagState):
        """Generate an answer considering the chat history."""
        print("---GENERATE WITH HISTORY---")
        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know. 
            Use three sentences maximum and keep the answer concise.
            Chat History: {chat_history}

            Question: {question}

            Context: {context}

            Answer:""",
            input_variables=["question", "context", "chat_history"],
        )
        generator = prompt | llm
        answer = generator.invoke(
            {
                "question": state["query"],
                "context": "\n".join(state["context"]),
                "chat_history": "\n".join(
                    [msg.content for msg in state["chat_history"]]
                ),
            }
        ).content
        state["answer"] = answer
        return state

    # Build the graph
    workflow = StateGraph(AdaptiveRagState)
    workflow.add_node("retrieve", retrieve_with_history)
    workflow.add_node("generate", generate_with_history)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile the graph
    return workflow.compile()
