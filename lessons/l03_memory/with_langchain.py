from uuid import uuid4

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from lessons import MODEL
from lessons.l03_memory.problem import SYSTEM_PROMPT

agent = create_agent(model=f"openai:{MODEL}", tools=[], system_prompt=SYSTEM_PROMPT, checkpointer=InMemorySaver())


def run(turns: list[tuple[str, str]]) -> list[str]:
    conversation = uuid4().hex
    replies = []
    for thread, message in turns:
        config = {"configurable": {"thread_id": f"{conversation}-{thread}"}}
        result = agent.invoke({"messages": [{"role": "user", "content": message}]}, config)
        replies.append(result["messages"][-1].text)
    return replies
