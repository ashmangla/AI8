"""LangGraph agent integration with production features."""

from typing import Dict, Any, List, Optional
import os

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from .models import get_openai_model
from .rag import ProductionRAGChain
from .guardrails import create_guardrails_guard, create_guardrails_node
from guardrails import Guard


class AgentState(TypedDict):
    """State schema for agent graphs."""
    messages: Annotated[List[BaseMessage], add_messages]
    validation_results: Optional[Dict[str, Any]]


def create_rag_tool(rag_chain: ProductionRAGChain):
    """Create a RAG tool from a ProductionRAGChain."""
    
    @tool
    def retrieve_information(query: str) -> str:
        """Use Retrieval Augmented Generation to retrieve information from the student loan documents."""
        try:
            result = rag_chain.invoke(query)
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
    
    return retrieve_information


def get_default_tools(rag_chain: Optional[ProductionRAGChain] = None) -> List:
    """Get default tools for the agent.
    
    Args:
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        List of tools
    """
    tools = []
    
    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=5))
    
    # Add Arxiv tool
    tools.append(ArxivQueryRun())
    
    # Add RAG tool if provided
    if rag_chain:
        tools.append(create_rag_tool(rag_chain))
    
    return tools


def create_langgraph_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    input_guard: Optional[Guard] = None,
    output_guard: Optional[Guard] = None
):
    """Create a simple LangGraph agent with guardrails.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        input_guard: Optional Guard instance for input validation
        output_guard: Optional Guard instance for output validation
        
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Create default guards if not provided
    if input_guard is None:
        input_guard = create_guardrails_guard(
            enable_jailbreak_detection=True,
            enable_pii_protection=True,
            enable_profanity_check=True
        )
    
    if output_guard is None:
        output_guard = create_guardrails_guard(
            enable_jailbreak_detection=False,
            enable_pii_protection=True,
            enable_profanity_check=True
        )
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return "output_guard"
    
    # Create guard nodes
    input_guard_node = create_guardrails_node(
        input_guard=input_guard,
        output_guard=None,
        strict_mode=True
    )
    
    output_guard_node = create_guardrails_node(
        input_guard=None,
        output_guard=output_guard,
        strict_mode=True
    )
    
    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    # Add nodes
    graph.add_node("input_guard", input_guard_node)
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("output_guard", output_guard_node)
    
    # Set entry point to input_guard
    graph.set_entry_point("input_guard")
    
    # Flow: input_guard -> agent
    graph.add_edge("input_guard", "agent")
    
    # Conditional from agent: tool calls -> action, otherwise -> output_guard
    graph.add_conditional_edges("agent", should_continue, {
        "action": "action",
        "output_guard": "output_guard"
    })
    
    # Action loops back to agent
    graph.add_edge("action", "agent")
    
    # Output guard leads to END
    graph.add_edge("output_guard", END)
    
    return graph.compile()

graph = create_langgraph_agent()

