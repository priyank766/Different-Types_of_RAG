from langchain.graphs import StateGraph
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict, Annotated, Sequence
import operator


# Define the state for our graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# Define the nodes of the graph
def call_model(state):
    messages = state["messages"]
    # In a real implementation, you would call your LLM here
    response = HumanMessage(content="This is a response from the model.")
    return {"messages": [response]}


def call_tool(state):
    messages = state["messages"]
    # In a real implementation, you would call your tool here
    response = HumanMessage(content="This is a response from a tool.")
    return {"messages": [response]}


# Define the edges of the graph
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    if "tool_calls" in last_message.additional_kwargs:
        return "tool"
    else:
        return "end"


# Build the graph
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tool", call_tool)
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tool": "tool",
        "end": "__end__",
    },
)
workflow.add_edge("tool", "agent")
workflow.set_entry_point("agent")

# Compile the graph
app = workflow.compile()
