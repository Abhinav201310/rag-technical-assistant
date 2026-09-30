# LangGraph Core Concepts and StateGraph

## What is LangGraph?
LangGraph is a library for building stateful, multi-actor applications with LLMs, used to create agent and multi-agent workflows. Compared to traditional LLM chains, LangGraph allows adding cycles, state persistence, conditional branching, and human-in-the-loop steps.

## StateGraph
The core abstraction in LangGraph is `StateGraph`. A `StateGraph` is parameterized by a custom `TypedDict` or `Pydantic` model representing the state.

### Key Components of a LangGraph Workflow:
1. **State**: A data structure that represents the current snapshot of your application. Every node in the graph reads from and writes updates to this state.
2. **Nodes**: Python functions or runnables that perform work (e.g., query analysis, retrieval, document grading, response generation). Each node receives the current state and returns a dict with updated state keys.
3. **Edges**: Connections between nodes:
   - **Normal Edges**: Direct transition from node A to node B.
   - **Conditional Edges**: Conditional functions that evaluate the state and return the name of the next node to execute.

## Corrective RAG (CRAG) Workflow
Corrective RAG is a pattern where an agent evaluates retrieved documents before generating an answer:
1. **Retrieve**: Fetch documents from vector store.
2. **Grade Documents**: LLM evaluates relevance of each document chunk.
3. **Branching Decision**:
   - If relevant documents exist $\to$ Proceed to generate answer.
   - If documents are irrelevant $\to$ Rewrite query and re-retrieve (up to a max retry limit).
   - If maximum retries exceeded $\to$ Fallback response or web search.

## Example Code for StateGraph
```python
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    retry_count: int

def retrieve(state: GraphState):
    print("---RETRIEVAL NODE---")
    return {"documents": ["Doc 1 content"]}

builder = StateGraph(GraphState)
builder.add_node("retrieve", retrieve)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", END)
graph = builder.compile()
```
