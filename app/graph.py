import json
import logging
from typing import List, Dict, Any, Optional, TypedDict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END

from app.config import settings
from app.ingestion import ingestion_manager

logger = logging.getLogger("rag_graph")

# Graph State Schema
class GraphState(TypedDict):
    raw_question: str
    processed_question: str
    query_type: Optional[str]
    documents: List[Document]
    relevant_documents: List[Document]
    generation: str
    citations: List[str]
    retry_count: int
    is_fallback: bool
    status: str
    trace: List[str]

def get_llm():
    if not settings.groq_api_key:
        raise ValueError("GROQ_API_KEY is not set. Please add it to your .env file or environment.")
    return ChatGroq(
        groq_api_key=settings.groq_api_key,
        model_name=settings.llm_model,
        temperature=0.0
    )

# --- Node 1: Query Analysis & Classification ---
def query_analysis_node(state: GraphState) -> Dict[str, Any]:
    trace = list(state.get("trace", []))
    trace.append("Executing Node 1: Query Analysis & Expansion")
    
    question = state.get("processed_question") or state["raw_question"]
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert AI technical assistant query processor.
Analyze the user question:
1. Classify the query into one of: 'conceptual', 'how-to', 'troubleshooting', 'api_reference', or 'general'.
2. Rewrite or expand the question to optimize semantic retrieval from a technical documentation vector store (add clear keywords, synonyms, technical terminology).

Return ONLY a valid JSON object in the exact format:
{{
  "query_type": "category",
  "expanded_query": "rephrased query string"
}}"""),
        ("human", "Question: {question}")
    ])
    
    try:
        chain = prompt | llm
        response = chain.invoke({"question": question})
        content = response.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
            
        data = json.loads(content)
        query_type = data.get("query_type", "general")
        expanded_query = data.get("expanded_query", question)
    except Exception as e:
        logger.warning(f"Query analysis fallback due to error: {e}")
        query_type = "general"
        expanded_query = question

    trace.append(f"Query classified as '{query_type}', expanded: '{expanded_query}'")
    return {
        "processed_question": expanded_query,
        "query_type": query_type,
        "trace": trace
    }

# --- Node 2: Retrieval ---
def retrieval_node(state: GraphState) -> Dict[str, Any]:
    trace = list(state.get("trace", []))
    query = state.get("processed_question", state["raw_question"])
    trace.append(f"Executing Node 2: Retrieval for query '{query}'")
    
    retriever = ingestion_manager.get_retriever(k=settings.top_k_retrieval)
    docs = retriever.invoke(query)
    
    trace.append(f"Retrieved {len(docs)} candidate chunks from vector database")
    return {
        "documents": docs,
        "trace": trace
    }

# --- Node 3: Document Grading (Self-Corrective) ---
def document_grading_node(state: GraphState) -> Dict[str, Any]:
    trace = list(state.get("trace", []))
    trace.append("Executing Node 3: Document Grading (Self-Corrective evaluation)")
    
    question = state["raw_question"]
    documents = state.get("documents", [])
    
    if not documents:
        trace.append("No documents retrieved to grade.")
        return {
            "relevant_documents": [],
            "trace": trace
        }
        
    llm = get_llm()
    grading_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a grader assessing relevance of a retrieved technical document chunk to a user question.
If the document chunk contains keywords, concepts, or information related to the question, reply with 'YES'.
Otherwise, reply with 'NO'.
Respond with only 'YES' or 'NO'."""),
        ("human", "User Question: {question}\n\nDocument Chunk:\n{context}")
    ])
    chain = grading_prompt | llm
    
    relevant_docs = []
    for i, doc in enumerate(documents):
        try:
            res = chain.invoke({"question": question, "context": doc.page_content})
            c = res.content.strip().upper()
            if "YES" in c:
                relevant_docs.append(doc)
        except Exception:
            relevant_docs.append(doc)
            
    trace.append(f"Grading complete: {len(relevant_docs)} / {len(documents)} chunks deemed relevant")
    return {
        "relevant_documents": relevant_docs,
        "trace": trace
    }

# --- Conditional Router / Edge Logic ---
def route_after_grading(state: GraphState) -> str:
    relevant_docs = state.get("relevant_documents", [])
    retry_count = state.get("retry_count", 0)
    
    if relevant_docs:
        return "generation"
    elif retry_count < settings.max_retries:
        return "rewrite_query"
    else:
        return "fallback"

# --- Node: Query Rewriter (On Grade Failure) ---
def rewrite_query_node(state: GraphState) -> Dict[str, Any]:
    trace = list(state.get("trace", []))
    current_retry = state.get("retry_count", 0) + 1
    trace.append(f"Executing Node: Query Rewriting (Attempt {current_retry}/{settings.max_retries})")
    
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query optimizer. The previous query failed to retrieve relevant documents from the technical knowledge base.
Rewrite the question into an alternative search query that focuses strictly on core keywords and technical concepts.
Output ONLY the rewritten query string with no explanation."""),
        ("human", "Original Question: {question}\nPrevious Query: {prev_query}")
    ])
    
    chain = prompt | llm
    res = chain.invoke({
        "question": state["raw_question"],
        "prev_query": state.get("processed_question", state["raw_question"])
    })
    rewritten = res.content.strip()
    
    trace.append(f"Rewritten query for re-retrieval: '{rewritten}'")
    return {
        "processed_question": rewritten,
        "retry_count": current_retry,
        "trace": trace
    }

# --- Node 4: Generation ---
def generation_node(state: GraphState) -> Dict[str, Any]:
    trace = list(state.get("trace", []))
    trace.append("Executing Node 4: Answer Generation with Context & Citations")
    
    question = state["raw_question"]
    docs = state.get("relevant_documents", [])
    
    context_blocks = []
    citations = set()
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", f"Document {i}")
        citations.add(source)
        context_blocks.append(f"--- Document [{source}] ---\n{doc.page_content}\n")
        
    context_text = "\n".join(context_blocks)
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert technical documentation assistant.
Answer the user's question accurately and concisely based ONLY on the provided context documents.
- Provide step-by-step code examples if applicable.
- Explicitly cite source documents using format [Source: filename/URL].
- If the context does not contain enough information, state what is missing instead of hallucinating."""),
        ("human", "Question: {question}\n\nRetrieved Technical Context:\n{context}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"question": question, "context": context_text})
    
    trace.append("Answer successfully generated and grounded in source context.")
    return {
        "generation": response.content.strip(),
        "citations": list(citations),
        "is_fallback": False,
        "status": "success",
        "trace": trace
    }

# --- Node: Graceful Fallback ---
def fallback_node(state: GraphState) -> Dict[str, Any]:
    trace = list(state.get("trace", []))
    trace.append("Executing Fallback Node: Retrieval limit reached with no relevant documentation found.")
    
    fallback_message = (
        "I'm sorry, but I could not find relevant information in the indexed technical documentation "
        f"to answer your question: '{state['raw_question']}'. "
        "Please check if the required documentation is indexed or try rephrasing your question."
    )
    
    return {
        "generation": fallback_message,
        "citations": [],
        "is_fallback": True,
        "status": "fallback",
        "trace": trace
    }

# --- Build LangGraph StateGraph ---
def build_rag_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("document_grading", document_grading_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generation", generation_node)
    workflow.add_node("fallback", fallback_node)
    
    workflow.add_edge(START, "query_analysis")
    workflow.add_edge("query_analysis", "retrieval")
    workflow.add_edge("retrieval", "document_grading")
    
    workflow.add_conditional_edges(
        "document_grading",
        route_after_grading,
        {
            "generation": "generation",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback"
        }
    )
    
    workflow.add_edge("rewrite_query", "retrieval")
    workflow.add_edge("generation", END)
    workflow.add_edge("fallback", END)
    
    return workflow.compile()

rag_graph = build_rag_graph()
