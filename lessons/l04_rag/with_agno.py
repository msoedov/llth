import tempfile
from functools import cache

from agno.agent import Agent
from agno.knowledge.embedder.openai import OpenAIEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.lancedb import LanceDb

from lessons import EMBEDDING_MODEL, MODEL
from lessons.l04_rag.problem import DOCS, SYSTEM_PROMPT


@cache
def agent() -> Agent:
    vector_db = LanceDb(
        uri=tempfile.mkdtemp(),
        table_name="handbook",
        embedder=OpenAIEmbedder(id=EMBEDDING_MODEL),
    )
    knowledge = Knowledge(vector_db=vector_db, max_results=4)
    for name, text in DOCS.items():
        knowledge.insert(name=name, text_content=text)
    return Agent(model=f"openai:{MODEL}", knowledge=knowledge, search_knowledge=True, instructions=SYSTEM_PROMPT, telemetry=False)


def run(question: str) -> str:
    return agent().run(question).content
