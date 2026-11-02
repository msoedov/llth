from functools import cache

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lessons import EMBEDDING_MODEL, MODEL
from lessons.l04_rag.problem import DOCS, SYSTEM_PROMPT


@cache
def vector_store() -> InMemoryVectorStore:
    documents = [Document(page_content=text, metadata={"source": name}) for name, text in DOCS.items()]
    chunks = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40).split_documents(documents)
    store = InMemoryVectorStore(embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL))
    store.add_documents(chunks)
    return store


@tool
def search_handbook(query: str) -> str:
    """Search the handbook and return the matching passages."""
    return "\n\n".join(doc.page_content for doc in vector_store().similarity_search(query, k=4))


agent = create_agent(model=f"openai:{MODEL}", tools=[search_handbook], system_prompt=SYSTEM_PROMPT)


def run(question: str) -> str:
    return agent.invoke({"messages": [{"role": "user", "content": question}]})["messages"][-1].text
