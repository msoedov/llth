import operator
from typing import Annotated, TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from lessons import MODEL
from lessons.l06_graph_agent.problem import SYSTEM_PROMPT, add, divide, multiply

tools = {fn.__name__: fn for fn in (add, multiply, divide)}
model = init_chat_model(f"openai:{MODEL}").bind_tools(list(tools.values()))


class State(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def llm_call(state: State) -> dict:
    return {"messages": [model.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])]}


def tool_node(state: State) -> dict:
    calls = state["messages"][-1].tool_calls
    return {"messages": [ToolMessage(str(tools[c["name"]](**c["args"])), tool_call_id=c["id"]) for c in calls]}


def should_continue(state: State) -> str:
    return "tool_node" if state["messages"][-1].tool_calls else END


builder = StateGraph(State)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)
builder.add_edge(START, "llm_call")
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
builder.add_edge("tool_node", "llm_call")
agent = builder.compile()


def run(question: str) -> str:
    return agent.invoke({"messages": [HumanMessage(question)]})["messages"][-1].text
