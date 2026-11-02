from langchain.agents import create_agent

from lessons import MODEL
from lessons.l01_tool_agent.problem import SYSTEM_PROMPT, get_weather

agent = create_agent(model=f"openai:{MODEL}", tools=[get_weather], system_prompt=SYSTEM_PROMPT)


def run(question: str) -> str:
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].text
