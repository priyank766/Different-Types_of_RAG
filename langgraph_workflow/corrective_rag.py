import os
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document


INDEX_DIR = "indices"


# Define the state for the Corrective RAG workflow
class CorrectiveRagState(TypedDict):
    query: str
    context: List[str]
    answer: str
    is_relevant: bool
    claims_supported: bool


# Define the data model for the grader
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


def get_corrective_rag_app(api_key: str):
    """Creates and returns the Corrective RAG LangGraph app."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", temperature=0, google_api_key=api_key
    )
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

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


    def retrieve(state: CorrectiveRagState):
        if combined_retriever is None:
            state["context"] = []
            return state
        
        # Use the combined retriever directly
        retrieved_docs_lc = combined_retriever(state["query"])
        retrieved_docs_content = [doc.page_content for doc in retrieved_docs_lc]
        
        state["context"] = retrieved_docs_content
        return state

    def grade_documents(state: CorrectiveRagState):
        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of a retrieved document to a user question.
            Here is the retrieved document: 

             {context} 

             Here is the user question: {question} 

            If the document contains keywords related to the user question, grade it as relevant.
            It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
            Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.""",
            input_variables=["context", "question"],
        )
        grader = prompt | structured_llm_grader
        relevance_grades = []
        for doc in state["context"]:
            grade = grader.invoke({"context": doc, "question": state["query"]})
            relevance_grades.append(grade.binary_score)

        state["is_relevant"] = "no" not in relevance_grades
        return state

    def generate(state: CorrectiveRagState):
        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks.
            Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know.
            Use three sentences maximum and keep the answer concise.

            Question: {question}

            Context: {context}

            Answer:""",
            input_variables=["question", "context"],
        )
        generator = prompt | llm
        answer = generator.invoke(
            {"question": state["query"], "context": "\n".join(state["context"])}
        ).content
        state["answer"] = answer
        return state

    def check_claims(state: CorrectiveRagState):
        prompt = PromptTemplate(
            template="""You are a grader assessing whether an answer is supported by the evidence.
            Here is the answer: 

            {answer} 

             Here is the evidence: 

            {context}

            Give a binary score 'yes' or 'no' to indicate whether the answer is supported by the evidence.""",
            input_variables=["answer", "context"],
        )
        grader = prompt | structured_llm_grader
        grade = grader.invoke(
            {"answer": state["answer"], "context": "\n".join(state["context"])}
        ).binary_score
        state["claims_supported"] = grade == "yes"
        return state

    def decide_to_generate(state: CorrectiveRagState):
        return "generate" if state["is_relevant"] else END

    def decide_to_finish(state: CorrectiveRagState):
        return END if state["claims_supported"] else "generate"

    workflow = StateGraph(CorrectiveRagState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)
    workflow.add_node("check_claims", check_claims)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents", decide_to_generate, {"generate": "generate", "end": END}
    )
    workflow.add_edge("generate", "check_claims")
    workflow.add_conditional_edges(
        "check_claims", decide_to_finish, {"generate": "generate", "end": END}
    )

    return workflow.compile()