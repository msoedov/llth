from typing import Annotated, TypedDict
from uuid import uuid4

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from lessons import MODEL
from lessons.l07_human_in_the_loop.problem import SYSTEM_PROMPT, send_email

model = init_chat_model(f"openai:{MODEL}").bind_tools([send_email], tool_choice="send_email")


class State(TypedDict):
    messages: Annotated[list, add_messages]
    result: str


def draft(state: State) -> dict:
    return {"messages": [model.invoke([SystemMessage(SYSTEM_PROMPT), *state["messages"]])]}


def human_gate(state: State) -> dict:
    call = state["messages"][-1].tool_calls[0]
    if not interrupt(call["args"]):
        return {"result": "rejected"}
    return {"result": send_email(**call["args"])}


builder = StateGraph(State)
builder.add_node("draft", draft)
builder.add_node("human_gate", human_gate)
builder.add_edge(START, "draft")
builder.add_edge("draft", "human_gate")
builder.add_edge("human_gate", END)
graph = builder.compile(checkpointer=InMemorySaver())


def run(case: dict) -> dict:
    config = {"configurable": {"thread_id": uuid4().hex}}
    first = graph.invoke({"messages": [HumanMessage(case["request"])], "result": ""}, config)
    final = graph.invoke(Command(resume=case["approve"]), config)
    return {"paused": bool(first.get("__interrupt__")), "result": final["result"]}
